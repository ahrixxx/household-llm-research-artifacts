# 월별 시뮬레이션 엔진

월별 계획, 소비·저축·투자 선택, 예산 검사와 기록 전달을 처리하는 엔진입니다. 원 연구에서 사용한 코드와 테스트를 그대로 공개했으며, 파일별 출처와 해시는 [PROVENANCE.json](../../PROVENANCE.json)에 있습니다.

```sh
PYTHONPATH=code python3 -m unittest discover -s code/phase2_monthly/tests -v
PYTHONPATH=code python3 -m phase2_monthly.demo --output outputs/synthetic-demo
```

저장소 루트에서 실행합니다. 표준 라이브러리만 사용하며 LLM API를 호출하지 않습니다. B/C/D와 3·6·9개월 기간을 조합한 9개 합성 경로를 각각 24개월 실행합니다. 첫 선택에 예산 위반을 넣어 수정 과정도 확인합니다.

완료된 출력 디렉터리를 다시 지정하면 기존 기록을 재사용합니다. 코드나 설정을 바꾸었다면 새 출력 디렉터리를 지정하세요. 출력 폴더는 합성 실행용으로 사용하고 실제 가구자료는 별도로 관리해야 합니다.

[최신 정책조건 설계](../../04_policy_simulation/README.md)에서는 이 시험 조건과 본실험 설계의 차이를 설명합니다. 월별로 어떤 정보가 전달되는지는 [4개월 실행 예시](../../04_policy_simulation/EXECUTION_EXAMPLE.md)에서 볼 수 있습니다.
