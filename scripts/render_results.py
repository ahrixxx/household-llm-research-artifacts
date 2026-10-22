"""Render published cohort aggregates; no private data or network."""
import csv
from pathlib import Path
root=Path(__file__).resolve().parents[1]
rows=list(csv.DictReader((root/'04_results/metrics.csv').open()))
lines=['# 개발 실험 결과표','', '단위: 2023년 실질 원/월 · 작을수록 오차가 작음 · 서로 다른 집단의 수치를 합산하지 않음','']
labels={'persistence':'전년 소비 유지','ridge':'Ridge','lasso':'Lasso','median_l1':'중앙값 회귀','rf_mse':'RF-MSE (보조)','rf_mae':'RF-MAE','gbm_mae':'HistGBM (보조)','catboost_mae':'CatBoost','catboost':'CatBoost','llm_v61':'LLM v6.1','llm_v6':'LLM v6','catboost_llm':'CatBoost 참고 LLM'}
cohort_labels={'1A_original':'1-A 기존 98가구','1A_education_diagnostic':'1-A 교육변화 추가 진단 53가구','1B_new':'1-B 신규 40가구','1A_catboost_assisted_matched':'CatBoost 참고 실험 공통 96가구'}
for cohort in dict.fromkeys(r['cohort'] for r in rows):
 lines += ['## '+cohort_labels[cohort],'','| 모형 | 가구 수 | 항목 평균 MAE | 총소비 MAE |','|---|---:|---:|---:|']
 for r in rows:
  if r['cohort']==cohort:lines.append(f"| {labels[r['model']]} | {r['households']} | {float(r['item_mae_krw']):,.2f} | {float(r['total_mae_krw']):,.2f} |")
 lines.append('')
(root/'04_results/TABLES.md').write_text('\n'.join(lines),encoding='utf-8')
print('Wrote 04_results/TABLES.md from public cohort-level metrics')
