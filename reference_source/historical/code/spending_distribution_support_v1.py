"""Mean/probability guidance and summaries of the SAME five training examples."""
from statistics import mean, median
import math
from phase1_rag_v1 import ITEMS

VERSION = 'spending_distribution_support_v1_20261015'
GUIDE = '''두 갈래 분포의 각 값을 같은 연도·단위에 맞춰 따로 추정하라.
occurrence_probability는 그해 항목별 연간 월평균 소비가 양수일 확률이다. 특정 월의 구매 확률이 아니다.
conditional_positive_mean_krw는 그 값이 양수인 경우의 조건부 평균이며, 구매 1회의 가격이나 양수 중앙값이 아니다. 연간 지출은 12로 나눈 월평균 단위로 생각하라.
전체 기대 소비는 발생 확률 × 양수 조건부 평균이다. 항목별 전체 중앙값의 합은 전체 기대 소비가 아니므로 평균 소비 변화와 혼동하지 말라.
낮은 발생 확률을 큰 양수 평균으로 임의 보상하거나, 고액 지출이 가능하다는 이유만으로 양수 평균을 부풀리지 말라. 음식·내구재를 포함한 각 항목의 과거 수준, 소득·자산·가구 정보에 근거하라.
평균과 중앙값이 같거나 일정 비율이라는 가정은 하지 말라. 과거 0도 자동 고정하지 않으며 기존 p≤0.5일 때 전체 중앙값 0 규칙을 유지한다.
실현된 연간 가구소득에 소득 손실을 또 빼거나 급여를 또 더하지 말라. 특정 감소액·감소율에 맞추지 말라.'''

REFERENCE_GUIDE = '''reference_spending_distribution은 이미 제공된 학습 사례 5개만 요약한 참고 자료다. 추가 사례나 대상 가구의 정답을 사용하지 않았다.
source_year 금액을 CPI로 대상 target_year의 명목 원화 월평균 단위에 맞췄다. 각 항목은 [사례수,양수사례수,양수평균,양수중앙값,양수최대값] 순서다.
5개 사례의 빈도는 대상 가구의 확정 발생 확률이 아니다. 양수사례 0개이면 양수 금액은 null이며 미래 지출을 0으로 강제하지 않는다.
양수사례가 적을수록 분포를 확인할 근거가 약하다. 최대값은 예측 상한이 아니며 평균·중앙값·빈도를 그대로 복사하거나 기계적으로 축소하지 말라.
금액 차이의 원인이 기간·연도·가구 구성·소득·자산 차이인지 함께 보고, 원래 사례와 대상 가구 정보를 종합하라.'''


def summary(task, query, references, deflator):
    if len(references) != 5:
        raise ValueError('Summarize exactly the five displayed references')
    if deflator.nominal:
        raise ValueError('CPI required for cross-year reference summary')
    target_years = [query['target_year']] if task == '1a' else query['target_years']
    if len(target_years) != (1 if task == '1a' else 3):
        raise ValueError('Target year count mismatch')
    years = []
    for t, year in enumerate(target_years):
        columns = {k: [] for k in ITEMS}
        source_years = []
        for rank, ref in enumerate(references, 1):
            if ref['rank'] != rank:
                raise ValueError('Reference rank mismatch')
            x, output = ref['input'], ref['observed_consumption']
            blocks = [output] if task == '1a' else output['years']
            expected_years = [x['target_year']] if task == '1a' else x['target_years']
            if len(blocks) != len(target_years) or [b['reference_year'] for b in blocks] != expected_years:
                raise ValueError('Reference outcome year alignment mismatch')
            block = blocks[t]
            if block['unit'] != 'nominal_KRW_per_month_annual_average' or set(block['items']) != set(ITEMS):
                raise ValueError('Reference outcome units/items mismatch')
            factor = deflator(1, block['reference_year']) / deflator(1, year)
            source_years.append(block['reference_year'])
            for k in ITEMS:
                v = block['items'][k]
                if type(v) not in (int, float) or not math.isfinite(v) or v < 0:
                    raise ValueError('Invalid reference consumption')
                columns[k].append(v * factor)
        items = {}
        for k, values in columns.items():
            positive = [v for v in values if v > 0]
            items[k] = [len(values), len(positive), *(
                [round(mean(positive), 2), round(median(positive), 2), round(max(positive), 2)]
                if positive else [None, None, None])]
        years.append(dict(target_year=year, source_years_by_rank=source_years, items=items))
    return dict(version=VERSION, unit='target_year_nominal_KRW_per_month_annual_average',
                columns=['n', 'n_positive', 'positive_mean', 'positive_median', 'positive_max'],
                scope='same_five_training_examples_only', years=years)
