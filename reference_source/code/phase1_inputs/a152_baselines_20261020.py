"""A152 latest-input local baselines. No API; train-only tuning, separate cohorts."""
import os,sys,json,subprocess,time,traceback
from pathlib import Path
from copy import deepcopy
from collections import Counter
import numpy as np
import pandas as pd
import extend_v6_baselines_20261020 as ext
b=ext.base
from korean_household_v2 import korean_view
from target_household_state_v1 import differences
from kr_consumption_validation_data import Groups,digest
ROOT=b.ROOT
OUT=ROOT/'outputs/phase1_evaluation_v1/a152_baselines_20261020'
RUN=ROOT/'outputs/phase1_evaluation_v1/v61_a152_20261020'
AUG=ROOT/'outputs/phase1_inputs/phase1a_education_development_augmented_20261020'
BASE=ROOT/'outputs/phase1a_consumption12_cases_v2_member_earnings'
MODELS=['ridge','lasso','median_l1','rf_mse','gbm_mae','catboost_mae','rf_mae']
original_feature=b.feature_row

def feature(x,year):
    z=deepcopy(x);z.pop('target_years',None)
    z['supplied_realized_household_income']=[z.pop('realized_2023_income')]
    return original_feature(z,year)
b.feature_row=feature
b.OUT=OUT;b.SOURCE=OUT/'fit_source'
b.CONFIG.update(task='1-A',reserve_fraction=None,status='development comparison; original98 and purposive education54 separate',feature_version='A_latest_state_member_earnings_relative_v1')

def prepare():
    if OUT.exists():raise FileExistsError('Preserve existing run')
    OUT.mkdir();(b.SOURCE/'1b').mkdir(parents=True)
    query=b.read(RUN/'inputs.jsonl');qids={r['case_id'] for r in query}
    tr=b.read(AUG/'training_revised_inputs.jsonl');assert len(tr)==6926
    links={r['case_id']:r for r in b.read(ROOT/'outputs/phase1a_consumption12_cases_v1/private/linkage_and_quality.jsonl')}
    groups=Groups();first={};nodes=set()
    for w in range(1,28):
        s=f'{w:02}';hp=ROOT/f'data/raw/klips_1_27차_stata/klips{s}h.dta';pp=ROOT/f'data/raw/klips_1_27차_stata/klips{s}p.dta'
        with pd.read_stata(hp,iterator=True) as rd:cols=[c for c in rd.variable_labels() if c.startswith('hhid') and c[4:].isdigit()]
        for r in pd.read_stata(hp,columns=cols,convert_categoricals=False).to_dict('records'):
            hid=r.get(f'hhid{s}')
            if pd.isna(hid) or hid<=0:continue
            n=f'{s}:{int(hid)}';groups.root(n)
            if w==27:nodes.add(n)
            for c in cols:
                if pd.notna(r[c]) and r[c]>0:groups.join(n,f'{c[4:]}:{int(r[c])}')
        for r in pd.read_stata(pp,columns=['pid',f'hhid{s}'],convert_categoricals=False).to_dict('records'):
            pid,hid=r['pid'],r[f'hhid{s}']
            if pd.notna(pid) and pid>0 and pd.notna(hid) and hid>0:
                n=f'{s}:{int(hid)}';groups.join(n,first.setdefault(int(pid),n))
        if w%5==0:print('family audit wave',w,flush=True)
    fm={digest(n):groups.root(n) for n in nodes}
    family={c:fm[r['record_id']] for c,r in links.items()}
    evalids={r['case_id'] for r in b.read(ROOT/'outputs/phase1a_consumption12_cases_v1/evaluation/inputs.jsonl')}
    protected={family[c] for c in qids|evalids}
    kept=[r for r in tr if family[r['case_id']] not in protected]
    tids={r['case_id'] for r in kept};assert not tids&qids
    assert not {family[c] for c in tids}&protected
    raw={r['case_id']:r['model_input'] for r in b.read(BASE/'training/inputs.jsonl') if r['case_id'] in tids}
    ref=json.loads((ROOT/'data/reference/klips_input_labels_20261020.json').read_text())
    template=query[0]['model_input']['household_at_2022_interview']['members'][0]['education_at_interview']
    xs=[]
    for r in kept:
        c=r['case_id'];x=deepcopy(raw[c]);source=r['model_input']
        members={m['member_index']:m for m in source['household_at_2022_interview']['members']}
        for m in x['household_at_2022_interview']['members']:
            e=deepcopy(members[m['member_index']]['education_at_interview'])
            for k in ['observation_timing','interpretation']:e[k]=template[k]
            m['education_at_interview']=e
        x=korean_view(x,ref);x['목표연도 가구상태']=deepcopy(source['목표연도 가구상태'])
        before=deepcopy(x['household_at_2022_interview'])
        for m in before['members']:m['occupation']=m.get('occupation',m.get('occupation_ksco7'))
        x['목표연도 가구상태'][0]['기준과의 관측 차이']=differences(before,x['목표연도 가구상태'][0]['관측상태'])
        xs.append(dict(case_id=c,model_input=x))
    xs+=deepcopy(query)
    qmap={r['case_id']:r['model_input'] for r in query}
    for r in xs:
        x=r['model_input'];assert x['target_year']==2023
        if r['case_id'] in qids:assert x==qmap[r['case_id']]
        f=feature(x,2023)
        assert f['requested/income/total_krw']==x['realized_2023_income']['total_krw']/10000
        assert f['requested/baseline_consumption/items/식비']==b.anchor(x)['items']['식비']*b.factor(2022)/10000
        x['target_years']=[2023] # adapter only, omitted by feature serializer
    ys=[dict(case_id=r['case_id'],observed_output={'years':[r['observed_output']]}) for r in b.read(BASE/'training/targets.jsonl') if r['case_id'] in tids]
    assert len(ys)==len(tids)
    b.lines(b.SOURCE/'1b/korean_inputs.jsonl',xs);b.lines(b.SOURCE/'1b/targets.jsonl',ys)
    roles=[dict(case_id=r['case_id'],family=family[r['case_id']],role='training' if r['case_id'] in tids else 'query',target_years=[2023]) for r in xs]
    b.lines(OUT/'split_private.jsonl',roles)
    sources=[b.SOURCE/'1b/korean_inputs.jsonl',b.SOURCE/'1b/targets.jsonl',RUN/'inputs.jsonl',RUN/'targets.jsonl',AUG/'manifest.json']
    report=dict(status='PREFLIGHT_PASSED',training_cases=len(tids),training_families=len({family[c] for c in tids}),query_cases=152,original98=98,education54=54,additional_family_overlap_exclusions=len(tr)-len(kept),query_training_family_overlap=0,evaluation_targets_read=False,query_input_exact_match=152,api_calls=0,config=b.CONFIG,sources={str(p.relative_to(ROOT)):b.sha(p) for p in sources},split_sha256=b.sha(OUT/'split_private.jsonl'))
    b.save(OUT/'preparation.json',report);b.save(ROOT/'docs/evidence/a152_baselines_preflight_2026_10_20.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ['sources','config']},ensure_ascii=False),flush=True)

def score():
    qs=b.read(RUN/'inputs.jsonl');ys={r['case_id']:np.array([r['observed_output']['years'][0]['items'][k] for k in b.ITEMS]) for r in b.read(RUN/'targets.jsonl')}
    predictions={'persistence':{r['case_id']:np.array([b.anchor(r['model_input'])['items'][b.ITEMS[k]]*b.factor(2022) for k in b.ITEMS]) for r in qs}}
    for n in MODELS:
        p=OUT/n/'predictions_private.jsonl'
        if p.exists():predictions[n]={r['case_id']:np.array(r['prediction_real2023']) for r in b.read(p)}
    results={}
    for cohort in ['original98','education54']:
        ids=[r['case_id'] for r in qs if r['cohort']==cohort];results[cohort]={}
        for n,p in predictions.items():
            err=np.array([p[c]-ys[c] for c in ids]);results[cohort][n]={'cases':len(ids),'item_mae':float(abs(err).mean()),'total_mae':float(abs(err.sum(axis=1)).mean()),'total_bias':float(err.sum(axis=1).mean()),'item_maes':dict(zip(b.ITEMS.values(),abs(err).mean(axis=0).tolist()))}
    # Match only households with all three completed numeric responses; no partial-draw ranking.
    rp=RUN/'live/results.jsonl';draws={}
    if rp.exists():
        for line in rp.read_text().splitlines(keepends=True):
            if not line.endswith('\n'):continue
            r=json.loads(line)
            if r['status']=='PARSED':draws.setdefault(r['case_id'],{})[r['repeat']]=np.array([r['parsed']['prediction']['2023'][k] for k in b.ITEMS])
    matched={}
    for cohort in results:
        ids=[r['case_id'] for r in qs if r['cohort']==cohort and len(draws.get(r['case_id'],{}))==3];matched[cohort]={'complete3_cases':len(ids),'results':{}}
        if not ids:continue
        for n,p in {**predictions,'llm_v61':None}.items():
            errors=[np.array(list(draws[c].values()))-ys[c] if p is None else np.array([p[c]-ys[c]]) for c in ids]
            matched[cohort]['results'][n]={'item_mae':float(np.mean([abs(e).mean() for e in errors])),'total_mae':float(np.mean([abs(e.sum(axis=1)).mean() for e in errors]))}
    b.save(OUT/'comparison.json',dict(unit='real2023 KRW/month',cohorts=results,complete3_matched=matched,api_calls=0,limitations=['Development only; education54 purposive, reported separately.','LLM draw errors averaged, not predictions before scoring.','Historical other-task overlap not erased; no independent final-test claim.']))

def run():
    state=dict(status='RUNNING',pid=os.getpid(),completed_models=[],failed_models=[],api_calls=0)
    def save():b.save(OUT/'status.json',state)
    save();subprocess.Popen(['caffeinate','-i','-w',str(os.getpid())])
    # Sequential model jobs limit contention with ongoing LLM processing.
    for name in MODELS:
        state['active_model']=name;save()
        with (OUT/(name+'.log')).open('w') as log:
            code=subprocess.run([sys.executable,str(Path(__file__).resolve()),'fit',name],stdout=log,stderr=subprocess.STDOUT).returncode
        state['failed_models' if code else 'completed_models'].append(name)
        score();save()
    state['status']='COMPLETE' if not state['failed_models'] else 'COMPLETE_WITH_MODEL_FAILURES';state['active_model']=None;save()

if __name__=='__main__':
    action=sys.argv[1]
    if action=='prepare':prepare()
    elif action=='fit':b.fit(sys.argv[2])
    elif action=='score':score()
    elif action=='run':run()
