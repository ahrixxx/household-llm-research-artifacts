# 월별 시뮬레이션 엔진

원 연구의 엔진과 테스트 파일을 그대로 보존한 폴더입니다. 각 파일의 출처와 해시는 [PROVENANCE.json](../../PROVENANCE.json)에 있습니다.

```sh
PYTHONPATH=code python3 -m unittest discover -s code/phase2_monthly/tests -v
PYTHONPATH=code python3 -m phase2_monthly.demo --output outputs/synthetic-demo
```

저장소 루트에서 실행합니다. 표준 라이브러리만 사용하며 LLM API를 호출하지 않습니다. B/C/D와 3·6·9개월 기간을 조합한 9개 합성 경로를 각각 24개월 실행합니다. 첫 선택에 예산 위반을 넣어 수정 과정도 확인합니다.

완료된 출력 디렉터리를 다시 지정하면 기존 기록을 재사용합니다. 코드나 설정을 바꾸었다면 새 출력 디렉터리를 지정하세요. 출력 폴더는 합성 실행용으로 사용하고 실제 가구자료는 별도로 관리해야 합니다.

이 데모는 엔진 점검용입니다. [최신 정책조건 설계](../../04_policy_simulation/README.md)와 [4개월 실행 예시](../../04_policy_simulation/EXECUTION_EXAMPLE.md)를 함께 참고할 수 있습니다.
