# LLM의 가구 소비 재현과 지급조건별 시뮬레이션

한국노동패널(KLIPS)의 가구자료로 LLM이 실제 소비를 얼마나 재현하는지 살펴보는 연구입니다. 가구의 소득·자산·구성·과거 소비를 입력하고, 12개 항목의 소비금액을 관측값 및 통계·머신러닝 모형과 비교합니다. 이를 바탕으로 소득 상실 이후 현금과 소비 전용 지급금에 따라 소비·금융자산 경로가 어떻게 달라지는지 탐색하고자 합니다.

이 저장소에는 도전학기제 중간보고 시점까지의 자료 처리 코드, 실험 방법, 개발 결과와 월별 시뮬레이션 구현을 정리했습니다. 결과는 2026년 10월 21일까지의 개발 실험을 기준으로 하며, 본실험용 최종 표본 배정은 진행 중입니다.

## 연구 구성

| 단계 | 분석 내용 | 현재 진행 상황 |
|---|---|---|
| 1차 · 소비 재현 | 일반 가구(1-A)와 퇴직 관련 가구(1-B)의 소비를 재현하고 기준선과 비교 | 입력·프롬프트 개선 및 개발 실험 완료, 본실험 준비 |
| 2차 · 정책 시나리오 | 소득 상실 상황에서 지급조건별 월별 소비·금융자산 경로 비교 | 기초 엔진 구현 및 합성자료 실행, 최신 정책조건 연결 준비 |

1차는 목표연도의 소득과 가구 상태도 제공하는 **조건부 소비 재현** 과제입니다. 2차에서는 관측된 정답이 없는 상황을 다루므로, 1차에서 확인한 LLM의 성능과 한계를 바탕으로 결과를 해석할 예정입니다.

## 자료 안내

| 폴더 | 내용 |
|---|---|
| [01_data_pipeline](01_data_pipeline/README.md) | 가구자료 구성, 변수 처리, 과제별 표본과 자료 구축 코드 |
| [02_experiments](02_experiments/README.md) | 평가지표, 기준선, LLM 개선 과정과 사용 프롬프트 |
| [03_results](03_results/README.md) | 1-A·1-B 비교 결과, CatBoost 보조 실험, 오류 분석과 그래프 |
| [04_policy_simulation](04_policy_simulation/README.md) | 2차 연구 설계, 월별 계획·실행·반성 구조와 합성 실행 예시 |

실험에 사용한 코드와 프롬프트는 `reference_source/`에, 독립 실행이 가능한 월별 엔진은 `code/phase2_monthly/`에 있습니다. [산출물별 파일 목록](EVIDENCE_INDEX.md)에서 관련 파일을 모아 볼 수 있습니다.

## 실행 방법

Python 3.10 이상과 macOS/Linux 환경을 기준으로 합니다. Windows에서는 파일 잠금에 사용하는 `fcntl` 때문에 WSL 등 POSIX 환경이 필요합니다.

```sh
# 집계 결과로 표 생성
python3 scripts/render_results.py

# 월별 엔진 검사
PYTHONPATH=code python3 -m unittest discover -s code/phase2_monthly/tests -v

# 합성자료로 월별 시뮬레이션 실행
PYTHONPATH=code python3 -m phase2_monthly.demo --output outputs/synthetic-demo

# 계획·실행·반성 연결 예시 생성
python3 scripts/build_monthly_walkthrough.py
```

위 명령은 표준 라이브러리만 사용하며, 실제 가구자료나 LLM API 키가 필요하지 않습니다. 합성 실행은 정해진 시험용 응답으로 엔진의 동작을 확인하는 예시입니다.

그래프는 `python3 scripts/plot_results.py`로 다시 그릴 수 있습니다. `matplotlib`과 한글 글꼴이 필요하며, macOS 외 환경에서는 스크립트의 글꼴 설정을 조정해야 합니다. 생성된 PNG·SVG·PDF는 [결과 폴더](03_results/figures)에 포함했습니다.

## 자료 이용

가구별 원자료와 개별 LLM 응답은 공개하지 않습니다. 따라서 이 저장소에서 실행할 수 있는 범위는 집계 결과의 표·그래프 생성과 합성자료를 이용한 엔진 검사입니다. KLIPS 기반 실험 전체를 다시 실행하려면 별도의 원자료와 의존 코드가 필요합니다.

[공개 범위](DATA_POLICY.md) · [실행 확인 기록](VALIDATION.md) · [원본 파일 출처와 해시](PROVENANCE.json)

별도의 오픈소스 라이선스는 부여하지 않았습니다. 외부 연구·데이터의 권리는 원저작자와 제공기관에 있습니다.
