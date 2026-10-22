"""Local exploratory comparison on the completed B40; no API calls.

New852 inputs only. Family split is fixed before fitting. Reserved-family
outcomes never enter fitting, preprocessing, CV, or reported scores.
"""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import date
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge, QuantileRegressor
from sklearn.model_selection import GroupKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
for p in [ROOT/'code', ROOT/'code/phase1_evaluation']:
    sys.path.insert(0, str(p))
from kr_consumption_validation_data import Groups, digest
from baselines import Fill
from household_decision_v2 import ITEMS, anchor
from v6_format_recovery2_20261020 import parse

SOURCE = ROOT/'outputs/phase1_inputs/household_decision_v6_target_state_20261020'
RUN = ROOT/'outputs/phase1_evaluation_v1/v6_b120_20261020'
OUT = ROOT/'outputs/phase1_evaluation_v1/v6_newcohort_baselines_20261020'
SEED = 20261020
CONFIG = dict(seed=SEED, folds=3, reserve_fraction=.20,
              feature_version='relative_year_requested_state_v2',
              ridge=[1., 100., 10000.], median_l1=[.01, .1, 1.],
              rf_mse=[2, 5, 10], rf_mae=[2, 5, 10], trees=200,
              gbm_mae=[7, 15, 31], gbm_iterations=150, gbm_learning_rate=.05,
              target='real2023_10000KRW', negative_predictions='clip_zero',
              weighting='equal family, equal events within family, equal years within event',
              status='exploratory development split, not frozen final study allocation')
CPI_PATH = ROOT/'data/reference/cpi_annual_national_2020base_2005_2023.json'
CPI = json.loads(CPI_PATH.read_text())['index_by_year']
CPI_EXTRA = ROOT/'data/reference/cpi_annual_national_2020base_2004_supplement_20261020.json'
CPI.update(json.loads(CPI_EXTRA.read_text())['index_by_year'])


def factor(year):
    return CPI['2023']/CPI[str(year)]


def read(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def lines(path, rows):
    path.write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False)+'\n' for r in rows))


def prepare():
    if OUT.exists():
        raise FileExistsError('Use existing immutable preparation')
    OUT.mkdir(parents=True)
    manifest = json.loads((SOURCE/'manifest.json').read_text())
    for name, h in manifest['files'].items():
        assert sha(SOURCE/name) == h
    xs = {r['case_id']:r['model_input'] for r in read(SOURCE/'1b/korean_inputs.jsonl')}
    links = {r['case_id']:r for r in read(SOURCE/'1b/private_linkage.jsonl')}
    query = {r['case_id']:r['model_input'] for r in read(RUN/'inputs.jsonl')}
    assert len(xs) == 852 and all(xs[k] == v for k,v in query.items())
    # Reconstruct the exact full-pool family linkage from all27 waves:
    # historical household links and each person's observed household moves.
    groups = Groups()
    first = {}
    raw_sources = {}
    nodes27 = set()
    for w in range(1, 28):
        s = f'{w:02}'
        hp = ROOT/f'data/raw/klips_1_27차_stata/klips{s}h.dta'
        pp = ROOT/f'data/raw/klips_1_27차_stata/klips{s}p.dta'
        with pd.read_stata(hp, iterator=True) as rd:
            cols = [c for c in rd.variable_labels() if c.startswith('hhid') and c[4:].isdigit()]
        for r in pd.read_stata(hp, columns=cols, convert_categoricals=False).to_dict('records'):
            hid = r.get(f'hhid{s}')
            if pd.isna(hid) or hid <= 0:
                continue
            node = f'{s}:{int(hid)}'
            groups.root(node)
            if w == 27:
                nodes27.add(node)
            for c in cols:
                if pd.notna(r[c]) and r[c] > 0:
                    groups.join(node, f'{c[4:]}:{int(r[c])}')
        for r in pd.read_stata(pp, columns=['pid',f'hhid{s}'], convert_categoricals=False).to_dict('records'):
            pid, hid = r['pid'], r[f'hhid{s}']
            if pd.notna(pid) and pid > 0 and pd.notna(hid) and hid > 0:
                node = f'{s}:{int(hid)}'
                groups.join(node, first.setdefault(int(pid),node))
        raw_sources[str(hp.relative_to(ROOT))] = sha(hp)
        raw_sources[str(pp.relative_to(ROOT))] = sha(pp)
        if w % 5 == 0:
            print('Family linkage wave',w,flush=True)
    for r in links.values():
        assert groups.root(first[r['pid']]) == r['family']
    qfamilies = {links[c]['family'] for c in query}
    assert len(qfamilies) == 40
    rid_family = {digest(n):groups.root(n) for n in nodes27}
    allocation = ROOT/'data/allocated/v17_snapshot_2026_10_08'
    membership = read(allocation/'current_record_membership.jsonl')
    protected = defaultdict(set)
    for r in membership:
        if r['role'] in ['phase1_a','phase2','excluded_jobloss_income']:
            assert r['record_id'] in rid_family
            protected[r['role']].add(rid_family[r['record_id']])
    # Earlier explicit income exclusions, including those absent in v17.
    earlier = ROOT/'outputs/disjoint_phase2_split_v6/private/membership.jsonl'
    for r in read(earlier):
        if r['split'] == 'excluded_income_input':
            protected['permanent_income_exclusion'].add(rid_family[r['record_id']])
    blocked = set().union(*protected.values())
    allfamilies = {r['family'] for r in links.values()}
    eligible = sorted(allfamilies - qfamilies - blocked,
                      key=lambda f:hashlib.sha256(f'v6-newcohort-split:{SEED}:{f}'.encode()).hexdigest())
    # Keep20% of non-query eligible families for future testing; no label-based stratification.
    reserve_n = math.ceil(len(eligible)*CONFIG['reserve_fraction'])
    reserve = set(eligible[:reserve_n])
    train = set(eligible[reserve_n:])
    assert not train & (qfamilies | blocked | reserve)
    roles = []
    for cid, r in links.items():
        f = r['family']
        role = ('query' if cid in query else 'query_family_other_event' if f in qfamilies
                else 'protected' if f in blocked else 'reserved' if f in reserve else 'training')
        roles.append(dict(case_id=cid, family=f, role=role,
                          target_years=xs[cid]['target_years']))
    lines(OUT/'split_private.jsonl',roles)
    report = dict(config=CONFIG, input_cases=len(xs), input_families=len(allfamilies),
                  counts={role:dict(events=sum(r['role']==role for r in roles),
                      families=len({r['family'] for r in roles if r['role']==role}),
                      target_years=sum(len(r['target_years']) for r in roles if r['role']==role))
                      for role in sorted({r['role'] for r in roles})},
                  protected_overlap={k:len(v & allfamilies) for k,v in protected.items()},
                  query_protected_overlap={k:len(v & qfamilies) for k,v in protected.items()},
                  raw_sources=raw_sources,
                  sources={str(p.relative_to(ROOT)):sha(p) for p in [
                      SOURCE/'manifest.json',SOURCE/'1b/korean_inputs.jsonl',
                      SOURCE/'1b/private_linkage.jsonl',allocation/'current_record_membership.jsonl',
                      earlier,RUN/'inputs.jsonl',CPI_PATH]},
                  split_sha256=sha(OUT/'split_private.jsonl'),
                  limits=['Prior outcome exposure is not erased by this new split.',
                          'Query pilot retained even if an existing other-phase boundary overlaps; conflicts reported.',
                          'Reserved families are prospective reserved cases, not certified untouched final tests.'])
    save(OUT/'preparation.json', report)
    print(json.dumps({k:report[k] for k in ['counts','protected_overlap','query_protected_overlap']},ensure_ascii=False),flush=True)


# Metadata and explanatory prose have no statistical category. Observed label,
# status, amounts, dates, uncertainty flags and structured history are retained.
SKIP = {'task','output_unit','unit','definition','interpretation','주의','정보 성격',
        'source','source_variable','classification','observation_timing',
        '계산식','근거','profile_note','가구조사 정확한 날짜','기간 미확정 이유',
        '변화시점','연속 미취업 확인','기준과의 관측 차이','입력 완비 상태'}


def feature_row(x, target_year):
    """Same input serializer for train and query; no outcomes or IDs."""
    base_year = anchor(x)['reference_year']
    out = {'requested_horizon':float(target_year-base_year),
           'requested_year':float(target_year)}
    def align(v):
        if isinstance(v,dict):
            return {k:align(z) for k,z in v.items()}
        if isinstance(v,list) and v and all(isinstance(z,dict) for z in v):
            yearkey=next((k for k in ['reference_year','평가연도','observation_year']
                          if all(k in z and isinstance(z[k],int) for z in v)),None)
            if yearkey:
                assert len({z[yearkey] for z in v})==len(v)
                return {f'relative_year_{z[yearkey]-base_year}':align(z) for z in v}
            if all('member_index' in z for z in v):
                assert len({z['member_index'] for z in v})==len(v)
                return {f'member_{z["member_index"]}':align(z) for z in v}
        if isinstance(v,list):
            return [align(z) for z in v]
        return v
    def visit(v, path, year, money=False):
        if isinstance(v,dict):
            year = v.get('reference_year',v.get('interview_year',
                    v.get('관측연도',v.get('평가연도',v.get('기준연도',year)))))
            for k,z in v.items():
                if k in SKIP:
                    continue
                visit(z,path+'/'+k,year,money or '_krw' in k or k=='items' or '(원)' in k)
        elif isinstance(v,list):
            out[path+'/length'] = float(len(v))
            for i,z in enumerate(v):
                visit(z,path+'/'+str(i),year,money)
        elif v is None:
            out[path] = None
        elif isinstance(v,bool):
            out[path] = str(v)
        elif isinstance(v,(int,float)):
            assert math.isfinite(v)
            out[path] = (str(v) if path.endswith('/member_index') or path.endswith('/person_member_index')
                         else float(v)*factor(year)/10000 if money else float(v))
        elif isinstance(v,str):
            try:
                dt = date.fromisoformat(v)
            except ValueError:
                out[path] = v
            else:
                out[path+'/days_from_anchor'] = float((dt-date(base_year,1,1)).days)
        else:
            raise TypeError(type(v))
    visit(align(x),'input',base_year)
    requested_income=next(v for v in x['supplied_realized_household_income']
                          if v['reference_year']==target_year)
    visit(align(requested_income),'requested/income',target_year)
    visit(anchor(x),'requested/baseline_consumption',base_year)
    if '목표연도 가구상태' in x:
        requested_state=next(v for v in x['목표연도 가구상태'] if v['평가연도']==target_year)
        visit(align(requested_state),'requested/state',target_year)
        changes=requested_state['기준과의 관측 차이']
        out['requested/added_members']=float(len(changes['목표명부에만 있는 가구원번호']))
        out['requested/removed_members']=float(len(changes['기준명부에만 있는 가구원번호']))
    if '평가연도 내 기간 요약' in x:
        period=next(v for v in x['평가연도 내 기간 요약'] if v['평가연도']==target_year)
        visit(align(period),'requested/period',target_year)
    assert not any(t in k for k in out for t in ['case_id','/pid','/family','observed_output'])
    return out


def dense(x):
    return x.toarray()


def pipeline(name, param):
    steps = [('fill',Fill()),('vectorize',DictVectorizer(sparse=True))]
    if name in ['ridge','median_l1']:
        steps.append(('scale',StandardScaler(with_mean=False)))
    if name == 'ridge':
        reg = Ridge(alpha=param, solver='lsqr')
    elif name == 'median_l1':
        reg = MultiOutputRegressor(QuantileRegressor(quantile=.5,alpha=param,solver='highs'),n_jobs=1)
    elif name == 'gbm_mae':
        steps.append(('dense',FunctionTransformer(dense)))
        reg = MultiOutputRegressor(HistGradientBoostingRegressor(
            loss='absolute_error',max_leaf_nodes=param,min_samples_leaf=10,
            max_iter=CONFIG['gbm_iterations'],learning_rate=CONFIG['gbm_learning_rate'],
            l2_regularization=1.,early_stopping=False,random_state=SEED),n_jobs=1)
    else:
        reg = RandomForestRegressor(n_estimators=CONFIG['trees'],min_samples_leaf=param,
                    criterion='absolute_error' if name=='rf_mae' else 'squared_error',
                    max_features=1.,random_state=SEED,n_jobs=8 if name=='rf_mae' else 2)
    return Pipeline(steps+[('model',reg)])


def weights(rows):
    events = defaultdict(set)
    counts = Counter(r['case_id'] for r in rows)
    for r in rows:
        events[r['family']].add(r['case_id'])
    return np.array([1/len(events[r['family']])/counts[r['case_id']] for r in rows])


def fit(name, resume=False):
    prep = json.loads((OUT/'preparation.json').read_text())
    assert sha(OUT/'split_private.jsonl') == prep['split_sha256']
    for n,h in prep['sources'].items():
        assert sha(ROOT/n) == h
    target = OUT/name
    if target.exists() and not resume:
        raise FileExistsError('Preserve model run')
    if resume:
        assert name=='catboost_mae' and (target/'cv_progress.json').exists() and not (target/'fit_report.json').exists()
    target.mkdir(exist_ok=resume)
    roles = read(OUT/'split_private.jsonl')
    trids = {r['case_id'] for r in roles if r['role']=='training'}
    qids = {r['case_id'] for r in roles if r['role']=='query'}
    xs = {r['case_id']:r['model_input'] for r in read(SOURCE/'1b/korean_inputs.jsonl')}
    # Stream and keep training targets only. Reserved outcomes are not fitted/scored.
    ys = {r['case_id']:r['observed_output'] for r in read(SOURCE/'1b/targets.jsonl') if r['case_id'] in trids}
    trainrows = [dict(case_id=r['case_id'],family=r['family'],year=y)
                 for r in roles if r['case_id'] in trids for y in xs[r['case_id']]['target_years']]
    qrows = [dict(case_id=r['case_id'],family=r['family'],year=y)
             for r in roles if r['case_id'] in qids for y in xs[r['case_id']]['target_years']]
    X = [feature_row(xs[r['case_id']],r['year']) for r in trainrows]
    Q = [feature_row(xs[r['case_id']],r['year']) for r in qrows]
    Y = np.array([[next(y for y in ys[r['case_id']]['years'] if y['reference_year']==r['year'])['items'][k]
                   *factor(r['year'])/10000 for k in ITEMS] for r in trainrows])
    groups = np.array([r['family'] for r in trainrows])
    folds = list(GroupKFold(n_splits=CONFIG['folds']).split(X,Y,groups))
    w = weights(trainrows)
    candidates = json.loads((target/'cv_progress.json').read_text()) if resume else []
    if resume:
        assert {v['parameter'] for v in candidates}==set(CONFIG[name])
    for param in CONFIG[name]:
        if any(v['parameter']==param for v in candidates):continue
        held = np.zeros_like(Y)
        for i,(tr,te) in enumerate(folds):
            assert not set(groups[tr]) & set(groups[te])
            model = pipeline(name,param)
            # Balance event-year observations by family also in model fitting.
            fitweights = weights([trainrows[j] for j in tr])
            fitweights *= len(tr)/fitweights.sum()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                with threadpool_limits(limits=2):
                    model.fit([X[j] for j in tr],Y[tr],model__sample_weight=fitweights)
            if any('ConvergenceWarning' in str(v.category) for v in caught):
                raise RuntimeError('Unconverged candidate')
            with threadpool_limits(limits=2):
                held[te] = np.maximum(model.predict([X[j] for j in te]),0)
            print(name,param,'fold',i+1,'complete',flush=True)
        loss = float(np.average(abs(held-Y).mean(axis=1),weights=w)*10000)
        candidates.append(dict(parameter=param,cv_family_mae=loss))
        save(target/'cv_progress.json',candidates)
    selected = min(candidates,key=lambda v:v['cv_family_mae'])['parameter']
    model = pipeline(name,selected)
    with threadpool_limits(limits=2):
        model.fit(X,Y,model__sample_weight=w*len(w)/w.sum())
        joblib.dump(model,target/'model_before_prediction_private.joblib')
        raw = model.predict(Q)*10000
    pred = np.maximum(raw,0)
    lines(target/'predictions_private.jsonl',[dict(**r,prediction_real2023=v.tolist()) for r,v in zip(qrows,pred)])
    joblib.dump(model,target/'model_private.joblib')
    report = dict(model=name,selected_parameter=selected,candidates=candidates,
                  feature_version=CONFIG['feature_version'],
                  training_events=len(trids),training_families=len(set(groups)),training_rows=len(X),
                  query_events=len(qids),query_rows=len(Q),
                  encoded_features=len(model.named_steps['vectorize'].feature_names_),
                  clipped_negative_cells=int((raw<0).sum()),prediction_cells=raw.size,
                  source_targets_sha256=sha(SOURCE/'1b/targets.jsonl'),
                  cpi_supplement_sha256=sha(CPI_EXTRA),
                  predictions_sha256=sha(target/'predictions_private.jsonl'))
    save(target/'fit_report.json',report)
    print(json.dumps(report,ensure_ascii=False),flush=True)


def score(names=None):
    xs = {r['case_id']:r['model_input'] for r in read(RUN/'inputs.jsonl')}
    ys = {r['case_id']:r['observed_output'] for r in read(RUN/'targets.jsonl')}
    ids = sorted(xs)
    actual = {}
    bases = {}
    for cid,x in xs.items():
        a = anchor(x)
        for y in x['target_years']:
            actual[cid,y] = np.array([next(z for z in ys[cid]['years'] if z['reference_year']==y)['items'][k]
                                      *factor(y) for k in ITEMS])
            bases[cid,y] = np.array([a['items'][label]*factor(a['reference_year']) for label in ITEMS.values()])
    predictions = {'persistence':{key:v[None,:] for key,v in bases.items()}}
    llm = defaultdict(dict)
    for r in sum((read(RUN/d/'results.jsonl') for d in ['live','recovery1','recovery2']),[]):
        p = parse(r['provider']['response']['text'],xs[r['case_id']])['prediction']
        for y,v in p.items():
            llm[r['case_id'],int(y)][r['repeat']] = [v[k]*factor(int(y)) for k in ITEMS]
    assert all(set(v)=={0,1,2} for v in llm.values())
    predictions['llm_v6'] = {k:np.array([v[i] for i in range(3)]) for k,v in llm.items()}
    fits = {}
    for name in (names or ['ridge','median_l1','rf_mse','rf_mae','gbm_mae']):
        fits[name] = json.loads((OUT/name/'fit_report.json').read_text())
        assert fits[name]['feature_version']==CONFIG['feature_version']
        predictions[name] = {(r['case_id'],r['year']):np.array(r['prediction_real2023'])[None,:]
                             for r in read(OUT/name/'predictions_private.jsonl')}
    results = {}
    family_losses = {}
    rng = np.random.default_rng(SEED)
    boot = rng.integers(0,len(ids),size=(10000,len(ids)))
    for name,rows in predictions.items():
        assert set(rows) == set(actual)
        losses=[];totals=[];bias=[];changes=[];items=[]
        for cid in ids:
            keys=[(cid,y) for y in xs[cid]['target_years']]
            errors=[rows[k]-actual[k] for k in keys]
            losses.append(float(np.mean([abs(v).mean() for v in errors])))
            totals.append(float(np.mean([abs(v.sum(axis=1)).mean() for v in errors])))
            bias.append(float(np.mean([v.sum(axis=1).mean() for v in errors])))
            changes.append(np.mean([rows[k].sum(axis=1)-bases[k].sum() for k in keys],axis=0))
            items.append(np.mean([abs(v).mean(axis=0) for v in errors],axis=0))
        family_losses[name]=np.array(losses)
        results[name]=dict(item_mae=float(np.mean(losses)),total_mae=float(np.mean(totals)),
                          total_bias=float(np.mean(bias)),total_mean_change=float(np.mean(changes)),
                          total_change_sd=float(np.array(changes).std(axis=0,ddof=1).mean()),
                          item_mae_by_item={label:float(v) for label,v in zip(ITEMS.values(),np.mean(items,axis=0))})
        for grouping in ['target_year','horizon']:
            strata=defaultdict(list)
            for cid,y in rows:
                group=y if grouping=='target_year' else y-anchor(xs[cid])['reference_year']
                e=rows[cid,y]-actual[cid,y]
                strata[group].append((float(abs(e).mean()),
                    float(abs(e.sum(axis=1)).mean()),float(e.sum(axis=1).mean())))
            results[name]['by_'+grouping]={str(k):dict(records=len(v),
                item_mae=float(np.mean([a[0] for a in v])),
                total_mae=float(np.mean([a[1] for a in v])),
                total_bias=float(np.mean([a[2] for a in v]))) for k,v in sorted(strata.items())}
    for name in results:
        d = family_losses[name]-family_losses['llm_v6']
        results[name]['mae_minus_llm_ci95'] = np.quantile(d[boot].mean(axis=1),[.025,.975]).tolist()
        results[name]['families_better_than_llm'] = int((d<0).sum())
    original=json.loads((RUN/'analysis/summary.json').read_text())
    assert abs(results['llm_v6']['item_mae']-original['overall']['llm_mae'])<1e-6
    assert abs(results['persistence']['item_mae']-original['overall']['persistence_mae'])<1e-6
    assert abs(results['llm_v6']['total_change_sd']-original['changes']['pred_total_change_sd'])<1e-6
    prep=json.loads((OUT/'preparation.json').read_text())
    report=dict(status='COMPLETED_EXPLORATORY_MATCHED_B40_COMPARISON',config=CONFIG,
                split={k:prep[k] for k in ['counts','protected_overlap','query_protected_overlap']},
                model_fits=fits,results=results,actual_changes=original['changes'],
                limitations=[
                    'New852 cohort, not old276 training. No target-based split or tuning.',
                    'Same40 query families/52years; LLM120 draws already obtained; statistical models deterministic.',
                    'All input amounts and targets deflated by annual national CPI to2023; not item-specific CPI.',
                    'Observed structured values and categories encoded; prompt prose is not a statistical feature.',
                    'Full input snapshots retained; derived narrative difference list omitted as redundant.',
                    'Random20% prospective reserve among eligible non-query families; not certified untouched test.',
                    'Three-fold family CV and fixed inherited grids; no claim globally optimal model.',
                    'No new LLM calls; historical other LLM versions are not matched on these40 families.',
                    'Paired family bootstrap exploratory, no multiplicity correction.'],
                source_manifest_sha256=sha(SOURCE/'manifest.json'),
                preparation_sha256=sha(OUT/'preparation.json'),script_sha256=sha(Path(__file__)))
    save(OUT/'comparison.json',report)
    save(ROOT/'docs/evidence/v6_newcohort_baselines_2026_10_20.json',report)
    save(OUT/'family_losses_private.json',{name:dict(zip(ids,v.tolist())) for name,v in family_losses.items()})
    print(json.dumps(results,ensure_ascii=False,indent=2))


def run_all():
    """Finish authorized local fits and scoring even while the user reviews literature."""
    names=['rf_mae','ridge','median_l1','rf_mse','gbm_mae']
    status={'status':'RUNNING','feature_version':CONFIG['feature_version'],
            'completed_models':[],'failed_models':[],'pid':os.getpid(),'api_calls':0}
    save(OUT/'status.json',status)
    def work(name):
        env=dict(os.environ,LOKY_MAX_CPU_COUNT='10',OMP_NUM_THREADS='2',
                 OPENBLAS_NUM_THREADS='2',VECLIB_MAXIMUM_THREADS='2')
        with (OUT/(name+'_aligned.log')).open('w') as log:
            return subprocess.run([sys.executable,str(Path(__file__).resolve()),
                'fit','--model',name],stdout=log,stderr=subprocess.STDOUT,env=env).returncode
    with ThreadPoolExecutor(max_workers=2) as ex:
        futures={ex.submit(work,name):name for name in names}
        for future in as_completed(futures):
            name=futures[future]
            code=future.result()
            status['completed_models' if code==0 else 'failed_models'].append(name)
            save(OUT/'status.json',status)
            print(name,'completed' if code==0 else 'failed',flush=True)
    if status['failed_models']:
        status['status']='STOPPED_REVIEW_REQUIRED'
    else:
        score()
        status['status']='COMPLETE'
    save(OUT/'status.json',status)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['prepare','fit','score','run_all'])
    p.add_argument('--model',choices=['ridge','median_l1','rf_mse','rf_mae','gbm_mae'])
    args=p.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='fit':fit(args.model)
    elif args.action=='score':score()
    else:run_all()
