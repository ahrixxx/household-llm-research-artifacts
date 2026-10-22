# 월별 기초 엔진 — 합성자료 실행

이 폴더의 Python 파일과 테스트는 원 연구의 동일 파일을 그대로 복사했습니다. 원본 해시는 루트 PROVENANCE.json 참조

```sh
PYTHONPATH=code python3 -m unittest discover -s code/phase2_monthly/tests -v
PYTHONPATH=code python3 -m phase2_monthly.demo --output outputs/synthetic-demo
```

표준 라이브러리만 사용하며 API를 호출하지 않습니다. B/C/D와3/6/9개월 조건의9개 합성 경로,각24개월을 점검합니다. 일부러 첫 예산 위반을 발생시켜 수정 경로도 검사합니다. 인간 행동이나 정책효과의 증거가 아닙니다.

같은 완료 디렉터리의 재실행은 기존 기록을 재사용합니다. 코드·설정이 바뀌면 새로운 출력 디렉터리를 사용하세요. 실제 가구 데이터를 이 공개 디렉터리에 넣지 마세요.

최신5조건 설계는 [설계 안내](../../05_policy_simulation/README.md)와 구분합니다.
