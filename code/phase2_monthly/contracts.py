"""Strict JSON contracts. Money is integer KRW; prose is never ledger truth."""
ITEMS = ('food', 'dining_out', 'public_education', 'private_education', 'housing',
         'healthcare', 'recreation', 'durables', 'communication', 'clothing',
         'public_transport', 'daily_necessities')
UNDERSTANDING_FIELDS = ('income_krw', 'cash_transfer_krw', 'cbdc_grant_krw', 'etf_price_krw',
                        'last_ui_paid_month', 'first_ui_unpaid_month', 'current_ui_krw')


def integer(value, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f'expected integer >= {minimum}')


def exact(value, keys):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f'expected exactly {sorted(keys)}')


def string(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('nonempty string required')


def evidence(value):
    if not isinstance(value, list):
        raise ValueError('evidence must be a list')
    for claim in value:
        exact(claim, {'source_id', 'path', 'value'})
        string(claim['source_id'])
        if not isinstance(claim['path'], str) or not claim['path'].startswith('/'):
            raise ValueError('claim path must be a JSON pointer')


def action(value):
    exact(value, {'consumption_krw', 'cbdc_use_krw', 'buy_units', 'sell_units', 'deposit_delta_krw'})
    for field in ('consumption_krw', 'cbdc_use_krw'):
        exact(value[field], ITEMS)
        for amount in value[field].values():
            integer(amount)
    integer(value['buy_units'])
    integer(value['sell_units'])
    if type(value['deposit_delta_krw']) is not int:
        raise ValueError('deposit_delta_krw must be signed integer KRW')


def narrative(value):
    exact(value, {'text', 'claims'})
    string(value['text'])
    evidence(value['claims'])


def validate(stage, value, month, horizon_end):
    if stage in ('plan', 'revise_plan'):
        exact(value, {'supersedes', 'reason', 'claims', 'schedule'})
        if value['supersedes'] is not None:
            string(value['supersedes'])
        string(value['reason'])
        evidence(value['claims'])
        expected = [str(m) for m in range(month, horizon_end + 1)]
        exact(value['schedule'], expected)
        for target in value['schedule'].values():
            action(target)
    elif stage == 'decide':
        exact(value, {'plan_id', 'action', 'explanation', 'action_claims', 'deviation_reason', 'environment_understanding'})
        string(value['plan_id'])
        action(value['action'])
        narrative(value['explanation'])
        exact(value['environment_understanding'], UNDERSTANDING_FIELDS)
        for field, amount in value['environment_understanding'].items():
            if amount is not None:
                integer(amount)
        if not isinstance(value['action_claims'], list):
            raise ValueError('action_claims must be a list')
        for claim in value['action_claims']:
            exact(claim, {'path', 'value'})
            string(claim['path'])
        if value['deviation_reason'] is not None:
            string(value['deviation_reason'])
    elif stage == 'reflect':
        exact(value, {'text', 'claims', 'lessons', 'next_plan_considerations'})
        string(value['text'])
        evidence(value['claims'])
        for field in ('lessons', 'next_plan_considerations'):
            if not isinstance(value[field], list):
                raise ValueError('reflection suggestions must be lists')
            for entry in value[field]:
                string(entry)
    else:
        raise ValueError('unknown model stage')


def zero_action():
    return dict(consumption_krw=dict.fromkeys(ITEMS, 0), cbdc_use_krw=dict.fromkeys(ITEMS, 0),
                buy_units=0, sell_units=0, deposit_delta_krw=0)


COMMON_PROMPT = '''당신은 제공된 가구의 월별 경제 선택을 수행한다. 제공된 사실만 사용한다.
sources의 profile/state/event/execution은 관측 또는 모형 장부의 사실이다.
interpretation은 생성된 해석이며 사실이 아니다. plan은 의도이며 집행 사실이 아니다.
economic_memory_v2 표시는 원기록에서 필요한 경제 정보만 추출한 보기다.
과거 plan의 schedule은 당시 최종 의도이며 실제 행동은 execution으로 확인한다.
과거 금액·제약을 현재 한도로 사용하지 말고 현재 상태와 제도를 적용한다.
기억에서 인용하는 사실은 source_id, JSON pointer path, value로 원기록에 연결한다.
새 성격·감정·인생 이야기, 관측하지 않은 사건, 미공개 미래 수익을 만들지 않는다.
소비 감소·저축·CBDC 소진·정책 우열을 목표로 맞추지 않는다.
금액 단위는 원이며 consumption_krw와 cbdc_use_krw는 동일한 12항목 전체를 포함한다.
CBDC 사용액은 총소비에 포함된 결제수단별 부분이며 추가 소비가 아니다.
반환 형식은 output_contract를 따른 단일 JSON 객체다. 설명은 짧은 선택 근거로 쓴다.
'''

STAGE_PROMPTS = {
    'plan': '현재 상태와 회상한 경제 경험을 참고해 제공된 계획 기간의 월별 계획을 작성한다. '
            '이전 계획이 있으면 supersedes에 그 ID를 쓰고 유지/변경 이유와 근거를 남긴다. '
            '미래 계획은 기대·의도이며 미래에 실현된 사실로 서술하지 않는다.',
    'revise_plan': '집행 전 제약 위반 피드백을 참고해 계획을 유지 또는 수정한다. '
                   '동일 월·동일 상태이며 앞선 행동은 집행되지 않았다. 기존 계획 ID를 supersedes로 남긴다.',
    'decide': '저장된 계획을 참고하여 이번 달 행동을 선택한다. 계획과 달라지면 deviation_reason에 '
              '이유를 명시한다. plan_id를 연결하고 action_claims에 소비·CBDC·매수·매도·예금이체 '
              '각 필드의 값을 JSON pointer로 명시한다. 계획 변경으로 과거 계획을 덮어쓰지 않는다. '
              'environment_understanding에 현재 가구소득(실업급여 포함), 별도 현금 지급, 당월 CBDC 지급, '
              'ETF 가격, 공지된 급여 마지막 지급 월·최초 미지급 월·현재 급여를 기입한다. '
              '제공되지 않은 정보는 추측하지 말고 null로 둔다. 이 사실 응답은 오답이어도 자동 교정하지 않는다.',
    'reflect': '방금 끝난 시뮬레이션 분기 3개월의 경제 환경·계획·집행·결과를 돌아본다. '
               '경제적 경험에서 얻은 해석과 이후 계획의 고려사항을 제시한다. 예상은 사실과 구별한다. '
               '회계 오류를 고치는 절차가 아니며 장부를 수정하지 않는다. 다음 달부터 참고된다.',
}


def output_contract(stage, month, horizon_end):
    claim = {'source_id': 'state/m001_open', 'path': '/cash_krw', 'value': 0}
    if stage in ('plan', 'revise_plan'):
        return dict(supersedes='previous plan ID or null', reason='유지/변경 이유', claims=[claim],
                    schedule={str(m): zero_action() for m in range(month, horizon_end + 1)})
    if stage == 'decide':
        return dict(plan_id='saved plan ID', action=zero_action(),
                    explanation={'text': '선택 근거', 'claims': [claim]},
                    action_claims=[{'path': '/buy_units', 'value': 0}],
                    deviation_reason='계획 이탈 이유 또는 null',
                    environment_understanding=dict.fromkeys(UNDERSTANDING_FIELDS, None))
    return dict(text='분기의 경제 경험에 대한 해석', claims=[claim],
                lessons=['근거를 구별한 해석'], next_plan_considerations=['다음 달 계획 고려사항'])
