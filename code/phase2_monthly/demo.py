"""Synthetic integration fixture, not an economic baseline or LLM evaluation.

PYTHONPATH=code .venv/bin/python -m phase2_monthly.demo --output /tmp/ubi-phase2-demo
"""
import argparse
import json
from pathlib import Path

from .checks import environment_truth, leaf_claims
from .contracts import zero_action
from .engine import EngineConfig, MonthlyEngine
from .ledger import LedgerRules, ReferenceLedger
from .scenarios import BENEFIT_DURATIONS, duration_panel_month
from .store import canonical, clone


class SyntheticModel:
    def __init__(self, inject_first_violation=True):
        self.inject_first_violation = inject_first_violation
        self.identity = {'model': 'SYNTHETIC_ONLY', 'version': 1,
                         'inject_first_violation': inject_first_violation}
        self.calls = []

    def generate(self, stage, request):
        self.calls.append(clone(request))
        ctx = request['context']
        month = request['month']
        if stage == 'reflect':
            last = [s for s in ctx['sources'] if s['kind'] == 'execution'][-1]
            result = dict(text='합성 분기 경제 경험을 요약한다. 행동 타당성의 증거가 아니다.',
                          claims=[dict(source_id=last['id'], path='/closing_state/cash_krw',
                                       value=last['data']['closing_state']['cash_krw'])],
                          lessons=['경제 경험의 사실과 해석을 분리한다.'],
                          next_plan_considerations=['다음 달 가용 자원을 다시 확인한다.'])
        else:
            opening = next(s for s in ctx['sources'] if s['id'] == f'state/m{month:03d}_open')
            fact = dict(source_id=opening['id'], path='/cash_krw', value=opening['data']['cash_krw'])
            target = zero_action()
            target['consumption_krw']['food'] = min(100_000, opening['data']['cash_krw'])
            if stage in ('plan', 'revise_plan'):
                previous = ctx['previous_plan']
                result = dict(supersedes=previous['id'] if previous else None,
                              reason='현재 현금과 경제 경험을 확인하여 합성 계획을 작성한다.', claims=[fact],
                              schedule={str(m): clone(target) for m in range(month, ctx['plan_horizon_end'] + 1)})
            else:
                target = clone(ctx['active_plan']['data']['schedule'][str(month)])
                bad = self.inject_first_violation and month == 1 and 'feedback' not in ctx
                if bad:
                    target['consumption_krw']['food'] = opening['data']['cash_krw'] + 1
                result = dict(plan_id=ctx['active_plan']['id'], action=target,
                              explanation=dict(text='합성 선택: 현재 예산에 근거한다.', claims=[fact]),
                              action_claims=leaf_claims(target),
                              deviation_reason='제약 검사 시험용으로 의도적으로 초과했다.' if bad else None,
                              environment_understanding=environment_truth(next(
                                  s['data']['observed'] for s in ctx['sources'] if s['id'] == f'event/m{month:03d}_environment')))
        return canonical(result)


def demo_rules():
    return LedgerRules(cbdc_eligible_items=('food', 'housing'), buy_fee_bps=0, sell_fee_bps=0,
                       borrowing='none', etf_units='integer',
                       settlement='inflows_sell_transfer_spend_buy_expire',
                       cbdc_allocation='earliest_expiry_first', cbdc_expiry='end_of_expiry_month',
                       money_unit='integer_krw')


def run_demo(root):
    summaries = []
    for condition, duration in [(c, d) for c in ('B', 'C', 'D') for d in BENEFIT_DURATIONS]:
        model = SyntheticModel()
        engine = MonthlyEngine(root, {'household': 'SYNTHETIC_001', 'condition': f'{condition}_UI_{duration}', 'repeat': 0},
                               {'source': 'synthetic', 'household_size': 2},
                               dict(cash_krw=300_000, deposit_krw=100_000, etf_units=0, cbdc_lots=[]),
                               EngineConfig.study(),
                               ReferenceLedger(demo_rules()), model)
        results = []
        for month in range(1, 25):
            panel = duration_panel_month(month, 150_000, 80_000, 10_000, ui_recipient=True,
                                         extra_payment_krw=50_000, cbdc_expiry_month=month + 2)
            observed = panel[f'{condition}_UI_{duration}']
            results.append(engine.step(observed))
        summaries.append(dict(condition=condition, duration_months=duration, completed_months=sum(r['status'] == 'COMPLETE' for r in results),
                              first_choice_violations=sum(not r['first_choice_feasible'] for r in results),
                              reflections=len(engine.store.list('interpretation')), newly_generated_calls=len(model.calls)))
    return dict(status='SYNTHETIC_ONLY', paid_calls=0, summaries=summaries)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run_demo(args.output), ensure_ascii=False, indent=2))
