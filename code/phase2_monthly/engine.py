"""Monthly cognition loop with durable first choices and same-state correction.

Model.generate(stage, request) returns raw JSON text; model.identity is frozen.
The engine has no provider credentials/client and cannot launch a paid run itself.
"""
from dataclasses import asdict, dataclass
import json
from pathlib import Path

from .checks import check_claims, check_decision, check_understanding, compare_plan, environment_truth
from .contracts import COMMON_PROMPT, STAGE_PROMPTS, integer, output_contract, validate
from .ledger import validate_state
from .memory import month_episode, previous_plan, recall
from .store import RecordStore, clone, digest


class UnknownCallOutcome(RuntimeError):
    """Request exists without a response. Never redispatch automatically."""


class InvalidModelOutput(RuntimeError):
    pass


@dataclass(frozen=True)
class EngineConfig:
    # Explicit run choices, not final experiment parameters or fitted thresholds.
    months: int
    plan_horizon_months: int
    recall_window_months: int
    recall_limit: int
    max_reselections: int
    remember_constraint_errors: bool

    def __post_init__(self):
        for field in ('months', 'plan_horizon_months', 'recall_window_months', 'recall_limit'):
            integer(getattr(self, field), 1)
        integer(self.max_reselections)
        if type(self.remember_constraint_errors) is not bool:
            raise ValueError('explicit error-memory policy required')
        if self.remember_constraint_errors:
            raise ValueError('correction history is excluded from economic memory')
        if self.recall_window_months < self.months:
            raise ValueError('recall must cover all available past months')

    @classmethod
    def study(cls, months=24):
        """Adopted cognition settings only, not payment rules or live approval."""
        return cls(months, 3, months, 5, 2, False)


class MonthlyEngine:
    def __init__(self, root, namespace, profile, initial_state, config, environment, model):
        validate_state(initial_state)
        self.store = RecordStore(root, namespace)
        self.config, self.environment, self.model = config, environment, model
        # Freeze implementation as well as all runtime rules and provider settings.
        implementation = {p.name: p.read_text() for p in sorted(Path(__file__).parent.glob('*.py'))}
        manifest = dict(engine='phase2_monthly_v2', implementation_hash=digest(implementation),
                        config=asdict(config), environment=environment.contract(),
                        model=clone(model.identity), reflection_schedule='simulation_month_3_6_9_etc_after_execution',
                        failure_disposition='stop_path_without_automatic_fallback')
        with self.store.lock():
            self.store.put('config', 'run', 0, manifest)
            self.store.put('profile', 'fixed', 0, profile)
            self.store.put('state', 'initial', 0, initial_state)

    def _call(self, stage, key, month, payload):
        horizon_end = min(self.config.months, month + self.config.plan_horizon_months - 1)
        request = dict(stage=stage, month=month, system=COMMON_PROMPT + STAGE_PROMPTS[stage],
                       output_contract=output_contract(stage, month, horizon_end),
                       context=clone(payload))
        existing = self.store.get('request', key)
        saved = self.store.get('response', key)
        if existing and existing['data'] != request:
            raise ValueError('resume request changed')
        if saved and not existing:
            raise ValueError('orphan model response')
        if saved is None:
            if existing:
                raise UnknownCallOutcome(f'{key}: pending/unknown outcome; inspect journal, no replay')
            self.store.put('request', key, month, request)
            try:
                raw = self.model.generate(stage, clone(request))
            except Exception as exc:
                self.store.put('failure', key, month,
                               {'status': 'MODEL_OUTCOME_UNKNOWN', 'exception_type': type(exc).__name__})
                raise UnknownCallOutcome(f'{key}: provider failed; no automatic retry') from exc
            if not isinstance(raw, str):
                self.store.put('failure', key, month, {'status': 'PROVIDER_CONTRACT_ERROR'})
                raise UnknownCallOutcome('provider must return raw text; request will not be replayed')
            saved = self.store.put('response', key, month, {'raw_text': raw})
        try:
            # Disallow NaN/Infinity and duplicate keys instead of silently accepting them.
            def pairs(items):
                obj = {}
                for key_, value in items:
                    if key_ in obj:
                        raise ValueError('duplicate JSON key')
                    obj[key_] = value
                return obj
            def invalid_constant(value):
                raise ValueError(f'invalid JSON constant {value}')
            parsed = json.loads(saved['data']['raw_text'], object_pairs_hook=pairs,
                                parse_constant=invalid_constant)
            validate(stage, parsed, month, horizon_end)
        except (ValueError, TypeError, KeyError) as exc:
            self.store.put('failure', key + '_schema', month, {'status': 'INVALID_MODEL_SCHEMA'})
            raise InvalidModelOutput(f'{key}: invalid output; raw preserved, path stopped') from exc
        return parsed

    def _memory(self, month, observed):
        return recall(self.store, month, observed, self.config.recall_limit)

    def _prior_plan(self, month):
        return previous_plan(self.store, month)

    def _save_plan(self, stage, key, month, payload, previous):
        plan = self._call(stage, key, month, payload)
        record = self.store.put('plan', key, month, plan)
        self.store.put('audit', key, month,
                       dict(factual_memory=check_claims(plan['claims'], payload['sources']),
                            plan_revision=compare_plan(plan, previous, month)))
        return record

    def step(self, observed):
        """One month's *already available* information only, never an entire future path."""
        month = observed['month']
        integer(month, 1)
        if month > self.config.months:
            raise ValueError('outside configured horizon')
        with self.store.lock():
            return self._step_locked(month, clone(observed))

    def _step_locked(self, month, observed):
        key = f'm{month:03d}'
        if month > 1 and self.store.get('event', f'm{month-1:03d}_complete') is None:
            raise ValueError('previous month is not complete; no time advance')
        previous = self.store.get('state', 'initial' if month == 1 else f'm{month-1:03d}_close')
        opened = self.environment.open_month(previous['data'], month, observed)
        environment_truth(observed)  # Reject contradictory public facts before generation.
        environment_record = self.store.put('event', key + '_environment', month,
                                            dict(type='economic_environment', observed=observed,
                                                 tags=[t for e in observed['events'] for t in e['tags']]))
        opening = self.store.put('state', key + '_open', month, opened)
        complete = self.store.get('event', key + '_complete')
        if complete:
            return clone(complete['data'])
        memories, recall_audit = self._memory(month, observed)
        self.store.put('recall', key, month, recall_audit)
        sources = [self.store.get('profile', 'fixed'), opening, environment_record] + memories
        # Deduplicate while retaining deterministic ordering.
        sources = list({r['id']: r for r in sources}.values())
        prior_plan = self._prior_plan(month)
        payload = dict(sources=sources, previous_plan=prior_plan,
                       environment_contract=self.environment.contract(),
                       plan_horizon_end=min(self.config.months, month + self.config.plan_horizon_months - 1))
        plan = self._save_plan('plan', key + '_p000', month, payload, prior_plan)
        first_valid = None
        for attempt in range(self.config.max_reselections + 1):
            attempt_key = f'{key}_a{attempt:03d}'
            decision_payload = dict(payload, active_plan=plan)
            decision = self._call('decide', attempt_key, month, decision_payload)
            choice_record = self.store.put('choice', attempt_key, month, decision)
            preview = self.environment.preview(opened, observed, decision['action'])
            audit = check_decision(decision, plan, payload['sources'], month)
            self.store.put('audit', attempt_key, month,
                           dict(consistency=audit, feasibility=preview,
                                environment_understanding=check_understanding(decision['environment_understanding'], observed)))
            if attempt == 0:
                first_valid = preview['valid']
            if preview['valid']:
                execution = self.store.put('execution', key, month,
                                           dict(choice_id=choice_record['id'], plan_id=plan['id'], **preview))
                self.store.put('state', key + '_close', month, preview['closing_state'])
                self.store.put('audit', key + '_execution', month,
                               check_decision(decision, plan, payload['sources'], month, preview['action']))
                break
            feedback = self.store.put('event', attempt_key + '_feedback', month,
                                       dict(type='constraint_feedback', choice_id=choice_record['id'],
                                            violations=preview['violations'], calculation=preview['calculation'],
                                            unchanged_opening_state_id=opening['id'],
                                            unchanged_opening_state_hash=opening['data_hash'],
                                            executed=False, tags=['constraint_feedback']))
            if attempt == self.config.max_reselections:
                result = dict(status='BLOCKED_FEASIBILITY', month=month,
                              first_choice_feasible=first_valid, attempts=attempt + 1,
                              executed=False, month_advanced=False)
                self.store.put('failure', key + '_exhausted', month, result)
                return result
            payload = dict(payload, previous_plan=plan, feedback=feedback,
                           sources=payload['sources'] + [feedback])
            plan = self._save_plan('revise_plan', f'{key}_p{attempt+1:03d}', month, payload, plan)
        # End-of-quarter synthesis can see this quarter's completed economic experience.
        # It is stored separately, available starting next month, and cannot alter balances.
        reflection_id = None
        if month % 3 == 0:
            quarter_sources = [self.store.get('profile', 'fixed')]
            for source_month in range(month - 2, month + 1):
                quarter_sources.extend(month_episode(self.store, source_month))
            reflection_payload = dict(sources=quarter_sources, quarter=month // 3,
                                      start_month=month - 2, end_month=month,
                                      environment_contract=self.environment.contract())
            reflection = self._call('reflect', key + '_reflection', month, reflection_payload)
            saved = self.store.put('interpretation', key + '_reflection', month, reflection, month + 1)
            self.store.put('audit', key + '_reflection', month,
                           check_claims(reflection['claims'], quarter_sources))
            reflection_id = saved['id']
        result = dict(status='COMPLETE', month=month, first_choice_feasible=first_valid,
                      attempts=attempt + 1, executed=True, month_advanced=True,
                      execution_id=execution['id'], reflection_id=reflection_id)
        self.store.put('event', key + '_complete', month, result)
        return result
