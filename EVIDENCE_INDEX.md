# 산출물별 파일 목록

중간보고서의 산출물 네 항목에 해당하는 문서와 코드를 정리했습니다.

| 번호 | 산출물 | 주요 파일 | 진행 상태 |
|---|---|---|---|
| 1 | [가구 실험자료 구축](01_data_pipeline/README.md) | [1-A 자료 구성](reference_source/code/build_phase1a_cases_v1.py), [1-B 자료 구성](reference_source/code/build_phase1b_cases_v1.py) | 가구자료·개발용 표본 구축 완료, 본실험용 최종 표본 배정 중 |
| 2 | [LLM 소비 재현 실험 및 평가](02_experiments/README.md) | [개선 과정과 사용 파일](02_experiments/RESULT_LINKS.md), [프롬프트·평가 코드](reference_source/code/phase1_inputs) | 개발 실험 및 기준선 비교 완료 |
| 3 | [개발 실험 결과 및 오류 분석](03_results/README.md) | [결과표](03_results/TABLES.md), [기준선 비교](03_results/figures/baseline_comparison.png), [CatBoost 보조 실험](03_results/figures/catboost_assisted.png) | 개발 결과 정리, 본실험 준비 |
| 4 | [월별 시뮬레이션 기초 구현](04_policy_simulation/README.md) | [엔진 코드](code/phase2_monthly/README.md), [합성 실행 예시](04_policy_simulation/EXECUTION_EXAMPLE.md), [검사 결과](VALIDATION.md) | 기초 구현 완료, 최신 정책조건 연결 및 실제 LLM 예비 실행 준비 |
