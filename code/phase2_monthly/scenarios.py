"""UI-duration diagnostic: 3/6/9 months separately within B, C and D.

All monetary inputs are explicitly resolved by the caller. This does not infer
legal UI eligibility/amounts or extract household data. Durations vary total UI.
"""
from .contracts import integer

BENEFIT_DURATIONS = (3, 6, 9)


def b_duration_month(month, duration, income_after_loss_krw, monthly_ui_krw,
                     etf_price_krw, *, ui_recipient):
    integer(month, 1)
    if month > 24 or duration not in BENEFIT_DURATIONS:
        raise ValueError('B diagnostic requires months 1..24 and duration 3/6/9')
    if ui_recipient is not True:
        raise ValueError('duration comparison is for explicitly designated UI recipients')
    integer(income_after_loss_krw)
    integer(monthly_ui_krw, 1)
    integer(etf_price_krw, 1)
    benefit = monthly_ui_krw if month <= duration else 0
    return dict(month=month, income_krw=income_after_loss_krw + benefit,
                cash_transfer_krw=0, deposit_interest_krw=0, other_cashflow_krw=0,
                cbdc_grants=[], etf_price_krw=etf_price_krw,
                events=[dict(id='announced_ui_schedule', announced_month=1,
                             text=f'실업급여는 1~{duration}개월 지급되며 {duration+1}개월부터 지급되지 않는다. '
                                  '대상자 근로소득은 24개월 동안 없고 추가 정책 지급은 없다.',
                             tags=['income', 'unemployment_benefit'],
                             facts=dict(duration_months=duration, last_paid_month=duration,
                                        first_unpaid_month=duration + 1,
                                        monthly_ui_krw=monthly_ui_krw, current_ui_krw=benefit,
                                        income_after_loss_krw=income_after_loss_krw))])


def duration_panel_month(month, income_after_loss_krw, monthly_ui_krw, etf_price_krw,
                         *, ui_recipient, extra_payment_krw, cbdc_expiry_month):
    """Nine matched current-month environments, no future market data.

    C and D receive the same explicitly supplied amount this month; all three
    UI durations keep that payment/expiry identical. Repeated monthly calls can
    encode any externally specified payment schedule. No payment default exists.
    The caller separately supplies all known future policy announcements.
    """
    integer(extra_payment_krw)
    if extra_payment_krw:
        integer(cbdc_expiry_month, month)
    elif cbdc_expiry_month is not None:
        raise ValueError('no grant: expiry must be null')
    panel = {}
    for condition in ('B', 'C', 'D'):
        for duration in BENEFIT_DURATIONS:
            row = b_duration_month(month, duration, income_after_loss_krw, monthly_ui_krw,
                                   etf_price_krw, ui_recipient=ui_recipient)
            if condition != 'B':
                row['events'][0]['text'] = (
                    f'실업급여는 1~{duration}개월 지급되며 {duration+1}개월부터 지급되지 않는다. '
                    '대상자 근로소득은 24개월 동안 없다.')
                row['cash_transfer_krw'] = extra_payment_krw if condition == 'C' else 0
                if condition == 'D' and extra_payment_krw:
                    row['cbdc_grants'] = [dict(id=f'policy_m{month:03d}', amount_krw=extra_payment_krw,
                                              expiry_month=cbdc_expiry_month)]
                row['events'].append(dict(
                    id=f'{condition}_current_payment', announced_month=month,
                    text=f'당월 {condition} 정책 지급 정보. 이후 지급 일정은 별도 공지 정보만 따른다.',
                    tags=['policy_payment'],
                    facts=dict(amount_krw=extra_payment_krw, medium='cash' if condition == 'C' else 'restricted_cbdc',
                               expiry_month=cbdc_expiry_month if condition == 'D' else None)))
            panel[f'{condition}_UI_{duration}'] = row
    return panel
