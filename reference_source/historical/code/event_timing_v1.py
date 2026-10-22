"""Input-only calendar arithmetic. Observed inter-job gaps are not unemployment."""
from datetime import date, timedelta

VERSION = 'event_timing_v1_20261015'
GUIDE = '''event_timing_summary와 reference_event_timing_summaries는 원래 날짜에서 코드로 계산한 달력 요약이다.
days_after_termination은 퇴직일 다음 날부터 해당 연도 말까지의 날짜 수이며 실업 일수가 아니다.
observed_start_gap_days는 [퇴직일 다음 날, 처음 관측된 이후 일자리 시작일)과 각 연도가 겹치는 날짜 수이다. 두 경계일은 제외한다.
min/max는 날짜 불확실성의 범위이다. 이 간격은 계속 실업했거나 다른 일을 하지 않았다는 증거가 아니다.
이후 일자리 기록이 없거나 중첩·미확인 기록이 있으면 간격은 null이다. 이를 연말까지 실업, 0일 또는 재취업 없음으로 바꾸지 말라.
연말 퇴직과 연초 퇴직, 같은 해와 다음 해 첫 관측 일자리 시작을 구분하여 연간 월평균 소비를 추정하라.
실현된 연간 가구소득에 실직 손실을 다시 빼거나 실업급여를 다시 더하지 말라. 급여 지급 달력은 확인되지 않았으므로 수급 일수를 임의 월에 배분하지 말라.'''


def bounds(value):
    if not value or not value.get('earliest') or not value.get('latest'):
        return None
    lo, hi = date.fromisoformat(value['earliest']), date.fromisoformat(value['latest'])
    if lo > hi:
        raise ValueError('Reversed date bounds')
    return lo, hi


def overlap(start, end, year):
    """Days in the half-open interval intersected with a calendar year."""
    return max(0, (min(end, date(year + 1, 1, 1)) - max(start, date(year, 1, 1))).days)


def summary(x):
    shock = x['supplied_realized_jobloss_path']
    year = x['event_year']
    if shock['event_year'] != year or x['target_years'] != [year, year + 1, year + 2]:
        raise ValueError('Event/target years disagree')
    end = bounds(shock.get('termination_date'))
    if end and (end[0].year != year or end[1].year != year):
        raise ValueError('Termination bounds cross the declared event year')
    path = shock.get('employment_path') or {}
    first = path.get('first_reemployment') or {}
    start = bounds(first.get('start'))
    # Conservative: unresolved overlaps can hide an earlier/continuing job.
    reliable = bool(end and start and path.get('status') == 'observed_reemployment'
                    and path.get('unresolved_episode_count', 0) == 0)
    if reliable and start[0] <= end[1]:
        raise ValueError('Observed subsequent job must start after termination bounds')
    same = 'unknown'
    if reliable:
        same = ('yes' if start[1].year == year else
                'no' if start[0].year > year else 'date_range_crosses_year')
    rows = []
    for y in x['target_years']:
        after = None if not end else dict(
            min=overlap(end[1] + timedelta(days=1), date(y + 1, 1, 1), y),
            max=overlap(end[0] + timedelta(days=1), date(y + 1, 1, 1), y))
        gap = None if not reliable else dict(
            min=overlap(end[1] + timedelta(days=1), start[0], y),
            max=overlap(end[0] + timedelta(days=1), start[1], y))
        rows.append(dict(reference_year=y, days_in_year=(date(y + 1, 1, 1) - date(y, 1, 1)).days,
                         days_after_termination=after, observed_start_gap_days=gap))
    return dict(version=VERSION, provenance='derived_from_input_dates_only',
                first_observed_start_same_event_year=same,
                gap_status='observed_interjob_interval_not_confirmed_unemployment' if reliable
                else 'unknown_due_to_missing_or_unresolved_job_dates',
                unemployment_insurance_payment_calendar='not_known', years=rows)


def retrieval_features(x):
    row = summary(x)['years'][0]
    return {f'{field}_fraction_{bound}': None if row[field] is None else row[field][bound] / row['days_in_year']
            for field in ('days_after_termination', 'observed_start_gap_days') for bound in ('min', 'max')}


def probe_messages(x):
    """Separate factual check, never mixed with consumption estimation/scoring."""
    import json
    expected = summary(x)
    spec = dict(first_observed_start_same_event_year='yes/no/date_range_crosses_year/unknown',
                gap_status='간격 계산 가능: observed_interjob_interval_not_confirmed_unemployment; 그 외: unknown_due_to_missing_or_unresolved_job_dates',
                years='목표연도순 {reference_year,days_in_year,days_after_termination:{min,max} 또는 null,observed_start_gap_days:{min,max} 또는 null}')
    return [dict(role='system', content=GUIDE + '\n소비를 예측하지 말고 지정한 JSON 필드만 답하라.'),
            dict(role='user', content=json.dumps(dict(target_household=x, output_specification=spec), ensure_ascii=False))], {
                k: expected[k] for k in spec}
