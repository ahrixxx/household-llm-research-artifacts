import json
from pathlib import Path
import tempfile
import unittest

from phase2_monthly.checks import check_claims, check_decision, leaf_claims, retrieve
from phase2_monthly.contracts import ITEMS, zero_action
from phase2_monthly.demo import SyntheticModel, demo_rules, run_demo
from phase2_monthly.engine import EngineConfig, InvalidModelOutput, MonthlyEngine, UnknownCallOutcome
from phase2_monthly.ledger import ReferenceLedger
from phase2_monthly.report import summarize
from phase2_monthly.scenarios import b_duration_month, duration_panel_month
from phase2_monthly.store import canonical, clone, digest


def observed(month=1, duration=6):
    return b_duration_month(month, duration, 100_000, 50_000, 10_000, ui_recipient=True)


def initial():
    return dict(cash_krw=100_000, deposit_krw=50_000, etf_units=3, cbdc_lots=[])


class MonthlyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def make(self, model=None, namespace=None, retries=1):
        return MonthlyEngine(self.root, namespace or {'household': 'synthetic', 'condition': 'B', 'repeat': 0},
                             {'source': 'synthetic', 'occupation': {'code': '312', 'label': '경영 관련 사무원'}},
                             initial(), EngineConfig(24, 3, 24, 5, retries, False),
                             ReferenceLedger(demo_rules()), model or SyntheticModel())

    def test_reselection_preserves_first_and_unchanged_state(self):
        model = SyntheticModel()
        engine = self.make(model)
        result = engine.step(observed())
        self.assertEqual(result['status'], 'COMPLETE')
        self.assertFalse(result['first_choice_feasible'])
        self.assertEqual(result['attempts'], 2)
        first = engine.store.get('choice', 'm001_a000')
        second = engine.store.get('choice', 'm001_a001')
        self.assertNotEqual(first['data']['action'], second['data']['action'])
        self.assertEqual(len(engine.store.list('execution')), 1)
        openings = [next(s for s in c['context']['sources'] if s['id'] == 'state/m001_open')
                    for c in model.calls if c['stage'] == 'decide']
        self.assertEqual(openings[0], openings[1])
        self.assertEqual(engine.store.get('state', 'm001_close')['data']['cash_krw'], 150_000)
        self.assertEqual(engine.store.get('plan', 'm001_p001')['data']['supersedes'], 'plan/m001_p000')
        self.assertLessEqual(engine.store.get('plan', 'm001_p000')['created_at'], first['created_at'])
        report = summarize(engine.store)
        self.assertEqual(report['first_choices_infeasible'], 1)
        self.assertEqual(report['months_corrected'], 1)

    def test_resume_no_model_calls_no_double_credit_or_execution(self):
        engine = self.make()
        engine.step(observed())
        before = {str(p): p.read_bytes() for p in engine.store.root.rglob('*.json')}
        resumed_model = SyntheticModel()
        resumed = self.make(resumed_model)
        resumed.step(observed())
        self.assertEqual(resumed_model.calls, [])
        self.assertEqual(before, {str(p): p.read_bytes() for p in engine.store.root.rglob('*.json')})
        with self.assertRaises(ValueError):
            resumed.step(dict(observed(), income_krw=1))

    def test_exhaustion_never_executes_or_advances(self):
        engine = self.make(retries=0)
        result = engine.step(observed())
        self.assertEqual(result['status'], 'BLOCKED_FEASIBILITY')
        self.assertEqual(engine.store.list('execution'), [])
        self.assertIsNone(engine.store.get('state', 'm001_close'))
        with self.assertRaises(ValueError):
            engine.step(observed(2))
        calls_before = len(engine.model.calls)
        self.assertEqual(engine.step(observed()), result)
        self.assertEqual(len(engine.model.calls), calls_before)

    def test_quarter_reflection_available_next_month_only(self):
        model = SyntheticModel(False)
        engine = self.make(model)
        for month in range(1, 5):
            engine.step(observed(month))
        reflection = engine.store.get('interpretation', 'm003_reflection')
        self.assertEqual(reflection['available_month'], 4)
        self.assertEqual(len(engine.store.list('interpretation')), 1)
        month3 = [c for c in model.calls if c['month'] == 3 and c['stage'] == 'decide'][0]
        month4 = [c for c in model.calls if c['month'] == 4 and c['stage'] == 'plan'][0]
        self.assertNotIn(reflection['id'], [s['id'] for s in month3['context']['sources']])
        self.assertIn(reflection['id'], [s['id'] for s in month4['context']['sources']])
        self.assertEqual(month4['context']['previous_plan']['id'], 'plan/m003_p000')
        reflected = [c for c in model.calls if c['stage'] == 'reflect'][0]
        self.assertEqual((reflected['context']['start_month'], reflected['context']['end_month']), (1, 3))
        self.assertFalse(any(s['month'] > 3 for s in reflected['context']['sources']))

    def test_namespace_isolates_household_condition_and_repeat(self):
        base = self.make(SyntheticModel(False))
        for month in range(1, 4):
            base.step(observed(month))
        for key, value in [('household', 'other'), ('condition', 'C'), ('repeat', 1)]:
            ns = dict(base.store.namespace, **{key: value})
            model = SyntheticModel(False)
            engine = self.make(model, ns)
            engine.step(observed())
            self.assertNotEqual(base.store.root, engine.store.root)
            self.assertEqual(engine.store.get('recall', 'm001')['data']['source_ids'], [])
            for call in model.calls:
                self.assertTrue(all(s['namespace'] == ns for s in call['context']['sources']))

    def test_transport_failure_journal_blocks_automatic_replay(self):
        class TimeoutModel(SyntheticModel):
            def generate(self, stage, request):
                self.calls.append(request)
                raise TimeoutError('synthetic')
        model = TimeoutModel()
        engine = self.make(model)
        for _ in range(2):
            with self.assertRaises(UnknownCallOutcome):
                engine.step(observed())
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(len(engine.store.list('request')), 1)
        self.assertEqual(engine.store.list('execution'), [])

    def test_invalid_schema_archives_raw_and_stops(self):
        class BadModel(SyntheticModel):
            def generate(self, stage, request):
                self.calls.append(request)
                return '{"x": NaN, "x": 1}'
        model = BadModel()
        engine = self.make(model)
        for _ in range(2):
            with self.assertRaises(InvalidModelOutput):
                engine.step(observed())
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(engine.store.get('response', 'm001_p000')['data']['raw_text'], '{"x": NaN, "x": 1}')

    def test_reflection_failure_does_not_duplicate_committed_month(self):
        class ReflectionTimeout(SyntheticModel):
            def generate(self, stage, request):
                if stage == 'reflect':
                    self.calls.append(request)
                    raise TimeoutError()
                return super().generate(stage, request)
        model = ReflectionTimeout(False)
        engine = self.make(model)
        for month in (1, 2):
            engine.step(observed(month))
        for _ in range(2):
            with self.assertRaises(UnknownCallOutcome):
                engine.step(observed(3))
        self.assertEqual(len(engine.store.list('execution')), 3)
        self.assertEqual(len([c for c in model.calls if c['stage'] == 'reflect']), 1)
        self.assertIsNotNone(engine.store.get('state', 'm003_close'))
        self.assertIsNone(engine.store.get('event', 'm003_complete'))
        with self.assertRaises(ValueError):
            engine.step(observed(4))

    def test_future_event_rejected_before_model_call(self):
        engine = self.make()
        obs = observed()
        obs['events'][0]['announced_month'] = 2
        with self.assertRaises(ValueError):
            engine.step(obs)
        self.assertEqual(engine.model.calls, [])

    def test_fabricated_memory_and_plan_mismatch_logged_without_steering_action(self):
        class ContradictionModel(SyntheticModel):
            def generate(self, stage, request):
                result = json.loads(super().generate(stage, request))
                if stage == 'decide':
                    result['explanation']['claims'][0]['value'] += 1
                    result['plan_id'] = 'plan/does_not_exist'
                    result['action_claims'][0]['value'] += 1
                return canonical(result)
        engine = self.make(ContradictionModel(False))
        engine.step(observed())
        audit = engine.store.get('audit', 'm001_a000')['data']['consistency']
        self.assertEqual(audit['factual_memory']['mismatch_count'], 1)
        self.assertFalse(audit['plan_id_matches'])
        self.assertEqual(audit['explanation_action']['invalid_or_duplicate_count'], 1)
        self.assertEqual(len(engine.store.list('choice')), 1)
        self.assertEqual(engine.store.get('state', 'm001_close')['data']['cash_krw'], 150_000)

    def test_immutable_records_and_config(self):
        engine = self.make()
        with self.assertRaises(ValueError):
            engine.store.put('profile', 'fixed', 0, {'changed': True})
        with self.assertRaises(ValueError):
            self.make(retries=2)
        with self.assertRaises(ValueError):
            engine.store.put('profile', '../bad', 0, {})

    def test_full_24_month_three_duration_paths_and_zero_call_resume(self):
        first = run_demo(self.root)
        self.assertEqual([r['completed_months'] for r in first['summaries']], [24] * 9)
        self.assertEqual([r['reflections'] for r in first['summaries']], [8] * 9)
        self.assertEqual([r['first_choice_violations'] for r in first['summaries']], [1] * 9)
        self.assertEqual([r['newly_generated_calls'] for r in first['summaries']], [58] * 9)
        second = run_demo(self.root)
        self.assertEqual([r['newly_generated_calls'] for r in second['summaries']], [0] * 9)

    def test_environment_factual_error_scored_without_retry(self):
        class WrongTiming(SyntheticModel):
            def generate(self, stage, request):
                value = json.loads(super().generate(stage, request))
                if stage == 'decide':
                    value['environment_understanding']['first_ui_unpaid_month'] = 99
                return canonical(value)
        engine = self.make(WrongTiming(False))
        engine.step(observed())
        self.assertEqual(len(engine.store.list('choice')), 1)
        self.assertEqual(summarize(engine.store)['first_environment_mismatches'], 1)

    def test_error_prose_cannot_reenter_future_memory_or_reflection(self):
        class PoisonCorrection(SyntheticModel):
            def generate(self, stage, request):
                result = json.loads(super().generate(stage, request))
                if stage == 'revise_plan':
                    result['reason'] = 'CORRECTION_ONLY_SENTINEL'
                    result['claims'] = [dict(source_id=request['context']['feedback']['id'],
                                             path='/executed', value=False)]
                if stage == 'decide' and 'feedback' in request['context']:
                    result['explanation']['text'] = 'CORRECTION_ONLY_SENTINEL'
                return canonical(result)
        model = PoisonCorrection()
        engine = self.make(model, retries=2)
        for month in range(1, 5):
            engine.step(observed(month))
        self.assertEqual(engine.store.get('plan', 'm001_p001')['data']['reason'],
                         'CORRECTION_ONLY_SENTINEL')
        self.assertTrue(engine.store.get('event', 'm001_a000_feedback'))
        later_calls = [c for c in model.calls if c['month'] > 1]
        for call in later_calls:
            self.assertNotIn('CORRECTION_ONLY_SENTINEL', canonical(call))
            self.assertNotIn('constraint_feedback', canonical(call))
            self.assertNotIn('_feedback', canonical(call))
        month2 = next(c for c in model.calls if c['month'] == 2 and c['stage'] == 'plan')
        plan = month2['context']['previous_plan']
        self.assertEqual(set(plan['data']), {'schedule'})
        self.assertEqual(plan['source_data_hash'], engine.store.get('plan', 'm001_p001')['data_hash'])
        ids = {r['id'] for r in month2['context']['sources']}
        self.assertTrue({'execution/m001', 'state/m001_close', 'event/m001_environment'} <= ids)

    def test_episode_limit_no_forced_fill_and_all_past_relevance(self):
        engine = self.make(SyntheticModel(False))
        for month in range(1, 13):
            row = observed(month)
            if month == 1:
                row['events'][0]['tags'].append('rare_early_event')
            engine.step(row)
        sources, audit = engine._memory(13, observed(13))
        self.assertEqual([r['month'] for r in audit['additional_episodes']], [11, 10, 9, 8, 7])
        self.assertEqual(len({s['id'] for s in sources}), len(sources))
        self.assertIn('interpretation/m012_reflection', audit['mandatory_source_ids'])
        self.assertIn('execution/m012', audit['mandatory_source_ids'])
        for m in (7, 8, 9, 10, 11):
            self.assertIn(f'event/m{m:03d}_environment', audit['source_ids'])
            self.assertIn(f'execution/m{m:03d}', audit['source_ids'])
            self.assertIn(f'state/m{m:03d}_close', audit['source_ids'])
        row = observed(13)
        row['events'][0]['tags'] = ['rare_early_event']
        _, audit = engine._memory(13, row)
        self.assertEqual([r['month'] for r in audit['additional_episodes']], [1])
        row['events'][0]['tags'] = ['no_match']
        calls = len(engine.model.calls)
        _, audit = engine._memory(13, row)
        self.assertEqual(audit['additional_episodes'], [])
        self.assertIn('execution/m012', audit['source_ids'])
        self.assertEqual(calls, len(engine.model.calls))

    def test_adopted_two_reselections_and_final_failure(self):
        class RepeatedViolation(SyntheticModel):
            def __init__(self, fail_last):
                super().__init__(False)
                self.fail_last = fail_last
                self.identity['fail_last'] = fail_last
            def generate(self, stage, request):
                result = json.loads(super().generate(stage, request))
                attempts = sum(c['stage'] == 'decide' for c in self.calls)
                if stage == 'decide' and (attempts < 3 or self.fail_last):
                    opening = next(s for s in request['context']['sources'] if s['id'] == 'state/m001_open')
                    result['action']['consumption_krw']['food'] = opening['data']['cash_krw'] + 1
                return canonical(result)
        for fail_last in (False, True):
            model = RepeatedViolation(fail_last)
            engine = self.make(model, {'household': 'synthetic', 'condition': 'B', 'repeat': int(fail_last)}, retries=2)
            result = engine.step(observed())
            self.assertEqual(result['attempts'], 3)
            self.assertEqual(len(engine.store.list('choice')), 3)
            self.assertEqual(len(model.calls), 6)  # initial plan/choice +2 revision/choice pairs
            self.assertEqual(len(engine.store.list('execution')), 0 if fail_last else 1)
            self.assertEqual(summarize(engine.store)['corrections_by_attempt']['1']['feasible'], 0)
            self.assertEqual(summarize(engine.store)['corrections_by_attempt']['2']['feasible'], int(not fail_last))
            self.assertEqual(result['status'], 'BLOCKED_FEASIBILITY' if fail_last else 'COMPLETE')
            self.assertEqual(engine.step(observed()), result)
            self.assertEqual(len(model.calls), 6)

    def test_study_config_excludes_error_memory_and_short_window(self):
        config = EngineConfig.study()
        self.assertEqual((config.plan_horizon_months, config.recall_limit, config.max_reselections), (3, 5, 2))
        with self.assertRaises(ValueError):
            EngineConfig(24, 3, 24, 5, 2, True)
        with self.assertRaises(ValueError):
            EngineConfig(24, 3, 6, 5, 2, False)


class LedgerAndChecksTests(unittest.TestCase):
    def setUp(self):
        self.ledger = ReferenceLedger(demo_rules())

    def test_same_month_sale_and_deposit_withdrawal_fund_consumption(self):
        state = dict(cash_krw=0, deposit_krw=20_000, etf_units=3, cbdc_lots=[])
        a = zero_action()
        a.update(sell_units=2, deposit_delta_krw=-20_000)
        a['consumption_krw']['food'] = 40_000
        before = clone(state)
        result = self.ledger.preview(state, observed(), a)
        self.assertTrue(result['valid'])
        self.assertEqual(result['closing_state'], dict(cash_krw=0, deposit_krw=0, etf_units=1, cbdc_lots=[]))
        self.assertEqual(state, before)

    def test_invalid_transactions_never_return_partial_settlement(self):
        for mutate, code in [
            (lambda a: a.update(sell_units=4), 'ETF_OVERSELL'),
            (lambda a: a.update(deposit_delta_krw=-50_001), 'DEPOSIT_OVERDRAW'),
            (lambda a: a['consumption_krw'].update(food=100_001), 'CASH_BUDGET'),
            (lambda a: a['cbdc_use_krw'].update(food=1), 'CBDC_BALANCE')]:
            with self.subTest(code=code):
                a = zero_action()
                mutate(a)
                result = self.ledger.preview(initial(), observed(), a)
                self.assertFalse(result['valid'])
                self.assertIn(code, [v['code'] for v in result['violations']])
                self.assertNotIn('closing_state', result)

    def test_cbdc_eligibility_allocation_expiry_and_freed_cash_investment(self):
        state = dict(cash_krw=10_000, deposit_krw=0, etf_units=0,
                     cbdc_lots=[dict(id='later', amount_krw=10_000, expiry_month=2),
                                dict(id='now', amount_krw=10_000, expiry_month=1)])
        a = zero_action()
        a['consumption_krw']['food'] = a['cbdc_use_krw']['food'] = 5_000
        a['buy_units'] = 1
        result = self.ledger.preview(state, observed(), a)
        self.assertTrue(result['valid'])
        self.assertEqual(result['closing_state']['cash_krw'], 0)
        self.assertEqual(result['closing_state']['etf_units'], 1)
        self.assertEqual(result['cbdc_expired'], [{'id': 'now', 'expired_krw': 5_000}])
        self.assertEqual(result['closing_state']['cbdc_lots'], [state['cbdc_lots'][0]])
        a['consumption_krw']['durables'] = a['cbdc_use_krw']['durables'] = 1
        self.assertIn('CBDC_INELIGIBLE:durables', [v['code'] for v in self.ledger.preview(state, observed(), a)['violations']])
        no_cash = dict(state, cash_krw=0)
        self.assertFalse(self.ledger.preview(no_cash, observed(), zero_action() | {'buy_units': 1})['valid'])

    def test_factual_claims_require_available_fact_not_interpretation(self):
        sources = [dict(id='state/a', kind='state', data={'cash': 5}),
                   dict(id='interpretation/a', kind='interpretation', data={'cash': 9})]
        claims = [dict(source_id='state/a', path='/cash', value=5),
                  dict(source_id='state/a', path='/cash', value=6),
                  dict(source_id='state/future', path='/cash', value=1),
                  dict(source_id='interpretation/a', path='/cash', value=9)]
        result = check_claims(claims, sources)
        self.assertEqual(result['claim_count'], 4)
        self.assertEqual(result['mismatch_count'], 3)
        self.assertEqual(check_claims([], sources)['coverage'], 'NO_FACT_CLAIMS')

    def test_plan_deviation_reason_and_explanation_execution_mismatch(self):
        target = zero_action()
        actual = clone(target)
        actual['consumption_krw']['food'] = 1
        plan = dict(id='plan/a', data={'schedule': {'1': target}})
        decision = dict(plan_id='plan/a', action=actual, explanation={'claims': []},
                        action_claims=leaf_claims(actual), deviation_reason=None)
        result = check_decision(decision, plan, [], 1, target)
        self.assertTrue(result['deviates_from_plan'])
        self.assertFalse(result['deviation_documented'])
        self.assertFalse(result['choice_equals_execution'])
        food = next(r for r in result['explanation_action']['details'] if r['claim']['path'] == '/consumption_krw/food')
        self.assertFalse(food['matches_execution'])
        decision['deviation_reason'] = '새로운 지출 필요'
        self.assertTrue(check_decision(decision, plan, [], 1)['deviation_documented'])

    def test_retrieval_time_and_tag_order(self):
        records = [dict(id=str(m), month=m, available_month=m+1, data={'tags': ['ui'] if m == 1 else []})
                   for m in (1, 2, 3, 4)]
        found = retrieve(records, 3, ['ui'], 24, 8)
        self.assertEqual([r['id'] for r in found], ['1', '2'])

    def test_b_durations_end_in_4_7_10_and_total_benefits_differ(self):
        for duration in (3, 6, 9):
            rows = [b_duration_month(m, duration, 100, 20, 10, ui_recipient=True) for m in range(1, 25)]
            self.assertEqual(rows[duration-1]['income_krw'], 120)
            self.assertEqual(rows[duration]['income_krw'], 100)
            self.assertEqual(sum(r['income_krw'] - 100 for r in rows), 20 * duration)
            self.assertTrue(all(r['cash_transfer_krw'] == 0 and not r['cbdc_grants'] for r in rows))
            self.assertEqual(rows[0]['events'][0]['facts']['first_unpaid_month'], duration + 1)
        with self.assertRaises(ValueError):
            b_duration_month(1, 7, 100, 20, 10, ui_recipient=True)
        with self.assertRaises(ValueError):
            b_duration_month(1, 3, 100, 20, 10, ui_recipient=False)

    def test_consumption_schema_matches_phase1(self):
        source = Path(__file__).parents[2] / 'phase1_rag_llm' / 'phase1_rag_v1.py'
        import ast
        tree = ast.parse(source.read_text())
        items = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == 'ITEMS' for t in n.targets))
        self.assertEqual(list(ITEMS), items)

    def test_bcd_panel_holds_payment_and_market_fixed_across_durations(self):
        rows = duration_panel_month(4, 100, 20, 10, ui_recipient=True,
                                    extra_payment_krw=30, cbdc_expiry_month=6)
        self.assertEqual(len(rows), 9)
        for d in (3, 6, 9):
            self.assertEqual(rows[f'B_UI_{d}']['cash_transfer_krw'], 0)
            self.assertEqual(rows[f'C_UI_{d}']['cash_transfer_krw'], 30)
            self.assertEqual(rows[f'D_UI_{d}']['cash_transfer_krw'], 0)
            self.assertEqual(rows[f'D_UI_{d}']['cbdc_grants'], [{'id': 'policy_m004', 'amount_krw': 30, 'expiry_month': 6}])
            self.assertEqual(len({rows[f'{c}_UI_{d}']['income_krw'] for c in 'BCD'}), 1)
        self.assertEqual(rows['C_UI_3']['income_krw'], 100)
        self.assertEqual(rows['C_UI_6']['income_krw'], 120)


if __name__ == '__main__':
    unittest.main()
