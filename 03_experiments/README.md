# 산출물 #2 · LLM 소비 재현 실험 및 평가 구현

## 개선 과정과 실제 결과

[개선 방법 → 사용 파일 → 개발 결과 대응표](RESULT_LINKS.md)

- 초기 RAG 비교와 시점·분포 보강 후의 결과를 별도 정리
- 역할·절차 안내, 가구 정보 요약, 전항목 설명의 비교 결과 수록
- 현재 1-A·1-B 입력과 기준선 결과, CatBoost 보조 실험으로 연결
- [과거 프롬프트·보강 코드](../reference_source/historical/README.md)와 [과거 실험 집계 CSV](historical_metrics.csv) 제공

가구자료 구성·개발용 표본 구축을 마쳤으며, 본실험용 최종 표본 배정은 진행 중입니다.

## 기준선

| 모형 | 비교 목적·구현 |
|---|---|
| 전년 실질소비 유지 | 변화 예측을 하지 않는 기준 |
| Ridge | L2 규제 선형 회귀 |
| Lasso | 변수 선택을 포함하는 L1 규제 선형 회귀 |
| 중앙값 회귀 | 0.5 분위수 회귀 |
| RF-MAE | 절대오차형 나무의 말단 중앙값, 나무별 예측은 평균 집계 |
| CatBoost-MAE | 수치·범주형 정보를 활용하는 부스팅 회귀 |
| RF-MSE, HistGBM | 보조 비교. RF-MSE는 말단 평균 및 나무 평균 집계 |

통계모형은 학습자료로 적합했고 LLM에는 같은 자료로 추가 가중치 학습을 하지 않았습니다. LLM 프롬프트는 개발 결과를 보며 수정했습니다. 이는 동일한 학습 조건의 구조 비교가 아닌 실용적 성능 비교입니다.

## 평가

항목 평균 MAE는 각 가구·응답에서 12항목의 절대오차를 평균합니다. 총소비 MAE는 먼저 12항목을 합한 뒤 실제 총소비와의 절대차를 계산합니다. 항목 오차의 상쇄 때문에 둘을 구분합니다. 반복 예측을 먼저 평균한 뒤 채점하는 것과 각 반복의 오차를 평균하는 것은 다릅니다.

평균 소비 변화와 가구 간 변화 SD는 변화의 방향·다양성을 보는 보조 분석입니다. 같은 입력에 대한 반복 응답 SD는 생성 일관성으로 별도 평가합니다. 두 모형의 가족별 오차 차이를 짝지어 재표집한 구간은 비교의 불확실성입니다. 다중비교 보정 없는 개발 탐색 구간이며 자동 승자 판정으로 사용하지 않습니다.

## 개선 시도

1. 가구 구성·교육·취업·개인소득 및 목표연도 상태 보완
2. 지출 발생과 금액 조정의 문제를 구분하고 단일 소비금액 출력으로 정리
3. 관측 사실 → 변화 방향 → 조정 폭 → 최종금액의 구조화된 설명 요청
4. 유사 가구 RAG와 문헌 경향 비교, 사건 시점·지출 분포 보강 후 재비교
5. CatBoost의 대상 가구별 12항목 예측을 참고로 제공하고 유지·수정 선택 허용

설명 출력은 모델 내부 추론의 증명이 아닙니다. RAG의 개선과 CatBoost 예측 제공 효과를 2차 정책 반응의 검증으로 해석하지 않습니다.

## 실제 사용 파일

- [1-A 공통 프롬프트](../reference_source/code/phase1_inputs/household_decision_1a_v3_uncertainty.txt)
- [1-B 공통 프롬프트](../reference_source/code/phase1_inputs/household_decision_1b_v2_target_state.txt)
- [CatBoost 참고 추가 지침](../reference_source/code/phase1_inputs/household_decision_1a_catboost_addendum_v1.txt)
- [구조화 응답·채점 계약](../reference_source/code/phase1_inputs/household_decision_v2.py)
- [B40 기준선 비교 구현](../reference_source/code/phase1_inputs/compare_v6_newcohort_baselines_20261020.py)
- [A152 기준선 구현](../reference_source/code/phase1_inputs/a152_baselines_20261020.py)
- [CatBoost 참고 입력 준비](../reference_source/code/phase1_inputs/prepare_catboost_llm_a98_20261021.py)
- [CatBoost 참고 비교 분석](../reference_source/code/phase1_inputs/analyze_catboost_llm_a98_20261021.py)

이는 원본 발췌로서 코드·프롬프트 증빙입니다. 모든 의존 코드와 비공개 중간자료를 포함한 재실행 패키지가 아니며, 이 공개본은 외부 LLM 호출을 실행하지 않습니다.

- [사례 검색 RAG 원본 구현](../code/phase1_rag_llm/phase1_rag_v1.py): 과거 검색 설정을 기록한 구현 증빙. 기본 설정은 당시 초안이며 최신 최종 조건으로 간주하지 않음
