"""Offline matched comparison; no API calls; household details remain private."""
import json,hashlib
from collections import Counter
import numpy as np
import a152_baselines_20261020 as a
b=a.b
OUT=b.ROOT/'outputs/phase1_evaluation_v1/catboost_llm_a98_20261021'
def read(p):return [json.loads(l) for l in p.read_text().splitlines()]
def draws(rows):
 d={}
 for r in rows:
  if r['status']=='PARSED':d.setdefault(r['case_id'],{})[r['repeat']]=np.array([r['parsed']['prediction']['2023'][k] for k in b.ITEMS])
 return d
def main():
 rows=read(OUT/'recovery3/results.jsonl');ds=read(OUT/'recovery3/dispatch.jsonl');jobs=read(OUT/'jobs.jsonl');key=lambda r:(r['case_id'],r['repeat']);jm={key(r):r for r in jobs}
 assert len(rows)==len(ds)==len(jm)==294 and len({key(r) for r in rows})==294
 assert {key(r) for r in rows}=={key(r) for r in ds}==set(jm)
 for r in ds:assert r['request_sha256']==hashlib.sha256(json.dumps(jm[key(r)],sort_keys=True).encode()).hexdigest()
 qs=read(OUT/'inputs.jsonl');new=draws(rows);old=draws(read(a.RUN/'recovery5/results.jsonl'))
 cat={r['case_id']:np.array(r['prediction_real2023']) for r in read(a.OUT/'catboost_mae/predictions_private.jsonl')}
 ys={r['case_id']:np.array([r['observed_output']['years'][0]['items'][k] for k in b.ITEMS]) for r in read(OUT/'targets.jsonl')}
 anchors={r['case_id']:np.array([b.anchor(r['model_input'])['items'][b.ITEMS[k]]*b.factor(2022) for k in b.ITEMS]) for r in qs}
 ids=[r['case_id'] for r in qs if len(new.get(r['case_id'],{}))==3 and len(old.get(r['case_id'],{}))==3]
 Y=np.array([ys[c] for c in ids]);A=np.array([anchors[c] for c in ids]);C=np.array([cat[c] for c in ids])[:,None,:]
 arrays={'catboost':C,'llm_v61':np.array([[old[c][j] for j in range(3)] for c in ids]),'catboost_llm':np.array([[new[c][j] for j in range(3)] for c in ids])}
 metrics={};losses={}
 for name,X in arrays.items():
  E=X-Y[:,None,:];losses[name]=abs(E).mean(axis=(1,2));D=(X-A[:,None,:]).sum(axis=2)
  metrics[name]={'item_mae':float(abs(E).mean()),'total_mae':float(abs(E.sum(axis=2)).mean()),'total_bias':float(E.sum(axis=2).mean()),'mean_total_change':float(D.mean()),'total_change_sd':float(D.std(axis=0,ddof=1).mean()),'item_maes':dict(zip(b.ITEMS.values(),abs(E).mean(axis=(0,1)).tolist()))}
  if X.shape[1]==3:metrics[name].update(repeat_item_sd=float(X.std(axis=1,ddof=1).mean()),repeat_total_sd=float(X.sum(axis=2).std(axis=1,ddof=1).mean()))
 N=arrays['catboost_llm'];diff=N-C;mask=abs(diff)>1e-6;before=np.broadcast_to(abs(C-Y[:,None,:]),N.shape);after=abs(N-Y[:,None,:]);rng=np.random.default_rng(20261021);ix=rng.integers(0,len(ids),(10000,len(ids)))
 comparison={}
 for name in ['catboost','llm_v61']:
  delta=losses['catboost_llm']-losses[name];comparison[name]={'assisted_minus_reference_item_mae':float(delta.mean()),'paired_household_ci95':np.quantile(delta[ix].mean(axis=1),[.025,.975]).tolist(),'assisted_better_households':int((delta<0).sum())}
 sensitivity={}
 for name in arrays:
  ll=[];tt=[]
  for r in qs:
   c=r['case_id'];X=cat[c][None,:] if name=='catboost' else np.array(list((new if name=='catboost_llm' else old)[c].values()));E=X-ys[c];ll.append(abs(E).mean());tt.append(abs(E.sum(axis=1)).mean())
  sensitivity[name]={'cases':len(qs),'item_mae':float(np.mean(ll)),'total_mae':float(np.mean(tt))}
 report={'planned_responses':294,'valid':sum(r['status']=='PARSED' for r in rows),'failures':dict(Counter(r['status'] for r in rows if r['status']!='PARSED')),'complete3_common_households':len(ids),'household_valid_repeat_distribution':dict(Counter(len(new.get(r['case_id'],{})) for r in qs)),'unit':'2023 KRW/month','actual_mean_total_change':float((Y-A).sum(axis=1).mean()),'actual_total_change_sd':float((Y-A).sum(axis=1).std(ddof=1)),'models':metrics,'paired_comparisons':comparison,'adjustments':{'tolerance_krw':1e-6,'changed_fraction':float(mask.mean()),'mean_absolute_revision':float(abs(diff).mean()),'changed_item_mae_before':float(before[mask].mean()),'changed_item_mae_after':float(after[mask].mean()),'improved_fraction_among_changed':float((after[mask]<before[mask]).mean()),'note':'Raw unrounded CatBoost amounts; rounding alone can count as change. Not evidence of substantive reasoning.'},'available_draw_sensitivity':sensitivity,'scope':['Development comparison; old controls reused, not contemporaneous.','MAE calculated per response before averaging, equal household/item weights.','Main analysis complete3 common households; available response sensitivity keeps all98.','Bootstrap exploratory10000 household resamples, no multiplicity adjustment.','Two transport unknown requests not retried; received-token cost excludes unknown charges.']}
 p=OUT/'analysis_completed';p.mkdir(exist_ok=True);b.save(p/'comparison.json',report);b.save(b.ROOT/'docs/evidence/catboost_llm_a98_analysis_2026_10_21.json',report)
 assert np.isclose(sensitivity['catboost_llm']['item_mae'],json.loads((OUT/'initial_scores.json').read_text())['cohorts']['original98']['item_mae'])
 print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
