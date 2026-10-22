"""Prepare independent CatBoost-assisted A98 requests; never calls an API."""
from pathlib import Path
import json,hashlib,math
from copy import deepcopy
from household_decision_v2 import synthetic,ITEMS,messages
from recover5_a152_v61_20261021 import parse
ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'outputs/phase1_evaluation_v1/v61_a152_20261020'
BASE=ROOT/'outputs/phase1_evaluation_v1/a152_baselines_20261020'
OUT=ROOT/'outputs/phase1_evaluation_v1/catboost_llm_a98_20261021'
def read(p):return [json.loads(l) for l in p.read_text().splitlines()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def lines(p,v):p.write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in v))
def main():
 assert not OUT.exists(),'Immutable bundle already exists'
 prep=json.loads((BASE/'preparation.json').read_text());assert prep['query_training_family_overlap']==0
 fit=json.loads((BASE/'catboost_mae/fit_report.json').read_text())
 predpath=BASE/'catboost_mae/predictions_private.jsonl'
 assert sha(predpath)==fit['predictions_sha256']
 inputs=[r for r in read(SOURCE/'inputs.jsonl') if r['cohort']=='original98'];ids={r['case_id'] for r in inputs};assert len(ids)==len(inputs)==98
 preds={r['case_id']:r for r in read(predpath)}
 original={r['case_id']:r['model_input'] for r in inputs}
 addition=Path(__file__).with_name('household_decision_1a_catboost_addendum_v1.txt').read_text()
 oldprompt=Path(__file__).with_name('household_decision_1a_v3_uncertainty.txt').read_text()
 jobs=[]
 for row in inputs:
  c=row['case_id'];x=deepcopy(row['model_input']);assert x['target_year']==2023
  v=preds[c]['prediction_real2023'];assert len(v)==12 and all(math.isfinite(z) and z>=0 for z in v)
  x['통계모형예측']={'모형':'CatBoost','평가연도':2023,'단위':'2023년 명목 원/월평균 (2023년 실질가격과 동일)','의미':'관측값이나 정답이 아닌 별도 학습자료 기반 예측','항목별예측액':dict(zip(ITEMS.values(),v))}
  check=deepcopy(x);del check['통계모형예측'];assert check==original[c]
  row['model_input']=x
  ms=messages(x);ms[0]['content']=oldprompt+'\n'+addition
  p=parse(json.dumps(synthetic(x),ensure_ascii=False),x)
  assert all(p['facts_match'].values()) and not p['reference_errors'] and not p['judgment_structure_errors']
  # Also test exact CatBoost final values and an explicit auxiliary pointer.
  o=synthetic(x)
  for label,z in zip(ITEMS.values(),v):
   r=o['항목별판단']['2023'][label];delta=z-r['기준소비액']
   r.update({'최종소비추정액':z,'변화금액':delta,'변화방향':'증가' if delta>0 else '감소' if delta<0 else '유지'})
   r['관측근거'].append({'경로':'/통계모형예측/항목별예측액/'+label,'값':z})
  q=parse(json.dumps(o,ensure_ascii=False),x);assert not q['reference_errors'] and not q['judgment_structure_errors']
  assert all(all(t for k,t in z.items() if k not in ['year','item']) for z in q['direction_amount_checks'])
  reserve=(len(json.dumps(ms,ensure_ascii=False).encode())+2048)*.3/1e6+32768*1.2/1e6
  jobs.append({'case_id':c,'messages':ms,'reserve':reserve})
 jobs=sorted(jobs,key=lambda j:hashlib.sha256(('cat-a98:'+j['case_id']).encode()).hexdigest())
 jobs=[dict(j,repeat=r) for r in range(3) for j in jobs];assert len(jobs)==294
 settings=json.loads((SOURCE/'settings.json').read_text());bound=sum(j['reserve'] for j in jobs);settings.update(max_calls=294,max_cost_usd=math.ceil(bound))
 targets=[r for r in read(SOURCE/'targets.jsonl') if r['case_id'] in ids];assert len(targets)==98
 OUT.mkdir();lines(OUT/'inputs.jsonl',inputs);lines(OUT/'jobs.jsonl',jobs);lines(OUT/'targets.jsonl',targets);save(OUT/'settings.json',settings)
 sources=[SOURCE/'inputs.jsonl',SOURCE/'targets.jsonl',predpath,BASE/'preparation.json',BASE/'catboost_mae/fit_report.json',Path(__file__),Path(__file__).with_name('household_decision_1a_catboost_addendum_v1.txt'),Path(__file__).with_name('household_decision_1a_v3_uncertainty.txt')]
 report={'status':'PREFLIGHT_PASSED','cases':98,'repeats':3,'planned_calls':294,'peak_reservation_usd':bound,'max_cost_usd':settings['max_cost_usd'],'family_overlap':0,'input_observations_unchanged':98,'synthetic_checks':196,'automatic_retries':0,'concurrency':4,'output_schema':'unchanged; CatBoost adjustment computed offline; reason in existing magnitude explanation','baseline_comparison':'CatBoost exact unrounded output, original v6.1 LLM historical, assisted LLM; all same98','limitations':['Development only; CatBoost selected after examining development scores','Historical LLM control not contemporaneously rerun','No guarantee of transfer to phase2'],'sources':{str(p):sha(p) for p in sources},'files':{p.name:sha(p) for p in OUT.iterdir() if p.is_file()}}
 save(OUT/'manifest.json',report);save(ROOT/'docs/evidence/catboost_llm_a98_preflight_2026_10_21.json',report)
 deps=[Path(__file__).with_name(n) for n in ['run_catboost_llm_a98_20261021.py','recover5_a152_v61_20261021.py','household_decision_v2.py','v6_format_recovery2_20261020.py','korean_contract_v2.py']]+[ROOT/'code/phase1_rag_llm/deepseek_observable_transport_v1.py']
 save(OUT/'execution_manifest.json',{'authorization':'User approved proposed original98 x3 CatBoost-assisted-only experiment, then reiterated proceed. Existing provider and household-data scope retained.','dependencies':{str(p):sha(p) for p in deps}})
 print(json.dumps({k:v for k,v in report.items() if k not in ['sources','files']},ensure_ascii=False))
if __name__=='__main__':main()
