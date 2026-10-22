"""Pure reference environment with explicit rules, not a certified KLIPS adapter.

No implicit credit, asset mapping, benefits, or market path. The caller supplies
already-resolved flows and initial liquid resources. This implementation supports
integer ETF units only and refuses unsupported settlement conventions.
"""
from dataclasses import asdict, dataclass
from .contracts import ITEMS, action, exact, integer, string
from .store import clone


@dataclass(frozen=True)
class LedgerRules:
    cbdc_eligible_items: tuple
    buy_fee_bps: int
    sell_fee_bps: int
    borrowing: str
    etf_units: str
    settlement: str
    cbdc_allocation: str
    cbdc_expiry: str
    money_unit: str

    def __post_init__(self):
        if len(set(self.cbdc_eligible_items)) != len(self.cbdc_eligible_items) or not set(self.cbdc_eligible_items) <= set(ITEMS):
            raise ValueError('invalid CBDC eligibility')
        for fee in (self.buy_fee_bps, self.sell_fee_bps):
            integer(fee)
            if fee > 10000:
                raise ValueError('fee exceeds 100%')
        supported = dict(borrowing='none', etf_units='integer',
                         settlement='inflows_sell_transfer_spend_buy_expire',
                         cbdc_allocation='earliest_expiry_first', cbdc_expiry='end_of_expiry_month',
                         money_unit='integer_krw')
        for key, value in supported.items():
            if getattr(self, key) != value:
                raise ValueError(f'unsupported explicit rule {key}')


def validate_state(state):
    exact(state, {'cash_krw', 'deposit_krw', 'etf_units', 'cbdc_lots'})
    for key in ('cash_krw', 'deposit_krw', 'etf_units'):
        integer(state[key])
    if not isinstance(state['cbdc_lots'], list):
        raise ValueError('CBDC lots must be a list')
    ids = []
    for lot in state['cbdc_lots']:
        exact(lot, {'id', 'amount_krw', 'expiry_month'})
        string(lot['id'])
        integer(lot['amount_krw'])
        integer(lot['expiry_month'], 1)
        ids.append(lot['id'])
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate CBDC lot IDs')


class ReferenceLedger:
    def __init__(self, rules):
        self.rules = rules

    def contract(self):
        return {'version': 'reference_ledger_v1', 'rules': asdict(self.rules)}

    def open_month(self, previous, month, observed):
        validate_state(previous)
        exact(observed, {'month', 'income_krw', 'cash_transfer_krw', 'deposit_interest_krw',
                         'other_cashflow_krw', 'cbdc_grants', 'etf_price_krw', 'events'})
        if observed['month'] != month:
            raise ValueError('wrong environment month')
        for key in ('income_krw', 'cash_transfer_krw', 'deposit_interest_krw'):
            integer(observed[key])
        integer(observed['etf_price_krw'], 1)
        if type(observed['other_cashflow_krw']) is not int:
            raise ValueError('other_cashflow_krw must be signed integer')
        if not isinstance(observed['events'], list):
            raise ValueError('events must be a list')
        for event in observed['events']:
            exact(event, {'id', 'announced_month', 'text', 'tags', 'facts'})
            string(event['id'])
            string(event['text'])
            integer(event['announced_month'], 1)
            if event['announced_month'] > month:
                raise ValueError('future information is not yet available')
            if not isinstance(event['tags'], list) or not all(isinstance(t, str) for t in event['tags']):
                raise ValueError('invalid event tags')
            if not isinstance(event['facts'], dict):
                raise ValueError('event facts must be an object')
        if len({e['id'] for e in observed['events']}) != len(observed['events']):
            raise ValueError('duplicate monthly event IDs')
        opened = clone(previous)
        # Expired balances should have been removed at the preceding settlement.
        if any(lot['expiry_month'] < month for lot in opened['cbdc_lots']):
            raise ValueError('stale expired CBDC in opening state')
        opened['cash_krw'] += observed['income_krw'] + observed['cash_transfer_krw'] + observed['other_cashflow_krw']
        opened['deposit_krw'] += observed['deposit_interest_krw']
        if not isinstance(observed['cbdc_grants'], list):
            raise ValueError('cbdc_grants must be a list')
        opened['cbdc_lots'].extend(clone(observed['cbdc_grants']))
        validate_state(opened)
        if any(lot['expiry_month'] < month for lot in opened['cbdc_lots']):
            raise ValueError('new CBDC grant already expired')
        return opened

    def preview(self, opened, observed, choice):
        """No mutation. Return a complete feasible settlement OR violations only."""
        action(choice)
        validate_state(opened)
        violations = []

        def violate(code, lhs, relation, rhs):
            violations.append(dict(code=code, lhs=lhs, relation=relation, rhs=rhs,
                                   excess=max(0, lhs - rhs)))

        sell, buy = choice['sell_units'], choice['buy_units']
        if sell > opened['etf_units']:
            violate('ETF_OVERSELL', sell, '<=', opened['etf_units'])
        price = observed['etf_price_krw']
        sell_gross, buy_gross = sell * price, buy * price
        sell_fee = (sell_gross * self.rules.sell_fee_bps + 9999) // 10000
        buy_fee = (buy_gross * self.rules.buy_fee_bps + 9999) // 10000
        cbdc_total = sum(choice['cbdc_use_krw'].values())
        consumption = sum(choice['consumption_krw'].values())
        for item in ITEMS:
            use = choice['cbdc_use_krw'][item]
            if use > choice['consumption_krw'][item]:
                violate(f'CBDC_EXCEEDS_CONSUMPTION:{item}', use, '<=', choice['consumption_krw'][item])
            if use and item not in self.rules.cbdc_eligible_items:
                violate(f'CBDC_INELIGIBLE:{item}', use, '<=', 0)
        cbdc_available = sum(lot['amount_krw'] for lot in opened['cbdc_lots'])
        if cbdc_total > cbdc_available:
            violate('CBDC_BALANCE', cbdc_total, '<=', cbdc_available)
        delta = choice['deposit_delta_krw']
        deposits = opened['deposit_krw'] + delta
        if deposits < 0:
            violate('DEPOSIT_OVERDRAW', -delta, '<=', opened['deposit_krw'])
        cash_available = opened['cash_krw'] + sell_gross - sell_fee - delta
        cash_required = consumption - cbdc_total + buy_gross + buy_fee
        if cash_required > cash_available:
            violate('CASH_BUDGET', cash_required, '<=', cash_available)
        calculation = dict(opening_cash_krw=opened['cash_krw'], sale_gross_krw=sell_gross,
                           sell_fee_krw=sell_fee, deposit_delta_krw=delta,
                           consumption_cash_krw=consumption - cbdc_total,
                           buy_gross_krw=buy_gross, buy_fee_krw=buy_fee,
                           cash_available_krw=cash_available, cash_required_krw=cash_required,
                           cbdc_available_krw=cbdc_available, cbdc_used_krw=cbdc_total)
        if violations:
            return dict(valid=False, violations=violations, calculation=calculation)
        remaining, redeemed, expired, lots = cbdc_total, [], [], []
        for original in sorted(opened['cbdc_lots'], key=lambda x: (x['expiry_month'], x['id'])):
            lot = clone(original)
            used = min(remaining, lot['amount_krw'])
            remaining -= used
            lot['amount_krw'] -= used
            redeemed.append({'id': lot['id'], 'used_krw': used})
            if lot['expiry_month'] == observed['month']:
                expired.append({'id': lot['id'], 'expired_krw': lot['amount_krw']})
            elif lot['amount_krw']:
                lots.append(lot)
        closing = dict(cash_krw=cash_available - cash_required, deposit_krw=deposits,
                       etf_units=opened['etf_units'] - sell + buy, cbdc_lots=lots)
        validate_state(closing)
        return dict(valid=True, violations=[], calculation=calculation, action=clone(choice),
                    closing_state=closing, cbdc_redeemed=redeemed, cbdc_expired=expired,
                    etf_value_krw=closing['etf_units'] * price)
