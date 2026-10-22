"""Local five-year jobloss conditional-reconstruction cases from v17.

Inputs: two pre-event consumption/income years, dated prior household state,
actual jobloss/reemployment/UI and three realized household-income years.
Targets: 3 x 12 consumption values, held separately except in training RAG.
"""
from collections import Counter,defaultdict
from datetime import date
from itertools import combinations
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import pandas as pd
from coupon_inputs import integer,conditional_amount,amount
from audit_jobloss_inputs_v1 import historical_income
from audit_five_year_jobloss_v1 import bounds
from build_phase1a_cases_v1 import ITEMS,category,read,write,jsave
from draft_feature_balanced_split_v4 import sha

ROOT=Path(__file__).resolve().parents[1]
VERSION='phase1b_consumption12_cases_v1'
PARTS={'training':('training_five_year_representatives.jsonl','training',250),
       'development':('development_five_year_representatives.jsonl','development',41),
       'evaluation':('phase1_b.jsonl','phase1_b',877)}
JOB_FIELDS=['jobtype','j001','j002','j003','j004','j005','j006','j601','j602','j611','j612','j613','j614']


def consumption(row,wave):
    values={key:amount(row.get(f'h{wave:02}{s}'))['value_krw'] for key,(s,_) in ITEMS.items()}
    if any(v is None for v in values.values()):raise ValueError('Incomplete historical12-item consumption')
    return dict(reference_year=wave+1996,unit='nominal_KRW_per_month_annual_average',
                items=values,total_12_items_krw=sum(values.values()))


def income(value):
    if value['total_krw'] is None:raise ValueError('Incomplete selected household income')
    return dict(reference_year=value['reference_year'],unit='nominal_KRW_per_year',
        definition='KLIPS_six_source_income_including_irregular_receipts',total_krw=value['total_krw'],
        components_krw={k:v['value_krw'] for k,v in value['components'].items()})


def date_payload(interval):
    return None if interval is None else dict(earliest=interval[0].isoformat(),latest=interval[1].isoformat(),
                                             precision='day' if interval[0]==interval[1] else 'bounded_dates')


def nonnegative(v):
    return float(v) if v is not None and pd.notna(v) and v>=0 else None


def ui_payload(job):
    if job is None:return dict(status='job_record_unavailable',daily_amount_krw=None,duration_days=None,
                               not_added_again_to_household_income=True)
    v,cf=job['values'],set(job['conflicts']);application=integer(v.get('j611'),1);receipt=integer(v.get('j612'),1)
    daily=nonnegative(v.get('j613'));days=nonnegative(v.get('j614'))
    status='received' if receipt==1 else 'not_received_or_not_applied' if receipt==2 or application==2 else 'unknown'
    if cf&{'j611','j612'} or (status=='not_received_or_not_applied' and ((daily or 0)>0 or (days or 0)>0)):
        status='conflicting_reports'
    return dict(status=status,
        daily_amount_krw=round(daily*1000) if status=='received' and daily is not None and 'j613' not in cf else None,
        duration_days=days if status=='received' and days is not None and 'j614' not in cf else None,
        detail_status='conflicting_reports' if cf&{'j613','j614'} else 'reported_or_missing',
        payment_calendar='not_allocated_to_months',not_added_again_to_household_income=True)


def employment_path(event_job,jobs,y):
    """Only supplied calendar events through Y+2; no post-window job dates."""
    stop=date(y+2,12,31)
    if event_job is None or event_job['end'] is None:
        return dict(status='event_date_unresolved',window_end=stop.isoformat(),spells=[],first_reemployment=None)
    end=event_job['end'];spells=[];ambiguous=0
    for j in jobs:
        if j['jobseq']==event_job['jobseq']:continue
        start,finish=j['start'],j['end']
        if finish and finish[1]<=end[0]:continue
        if start and start[0]>stop:continue
        if start is None or start[0]<=end[1] or start[1]>stop:
            ambiguous+=1;continue
        stop_payload=date_payload(finish) if finish and finish[1]<=stop else None
        spells.append(dict(start=date_payload(start),end_within_window=stop_payload,
            end_status='observed_within_window' if stop_payload else 'not_established_within_window'))
    spells.sort(key=lambda r:(r['start']['earliest'],r['start']['latest']))
    first=None
    if spells:
        earliest=min(date.fromisoformat(s['start']['earliest']) for s in spells)
        latest=min(date.fromisoformat(s['start']['latest']) for s in spells)
        first=dict(start=date_payload((earliest,latest)),elapsed_days_min=(earliest-end[1]).days,
                   elapsed_days_max=(latest-end[0]).days)
    return dict(status='observed_starts_with_date_uncertainty' if spells and ambiguous else
                       'observed_reemployment' if spells else 'unresolved_job_dates_or_overlap' if ambiguous else
                       'no_later_start_observed_within_window',
        window_end=stop.isoformat(),spells=spells,first_reemployment=first,
        unresolved_episode_count=ambiguous,
        interpretation='Observed job dates, not job-search effort or proof of continuous unemployment.')


def household_state(row,w,people,pid,maps,occupations):
    s=f'{w:02}';audit=Counter();members=[];links=[];target_index=None
    for slot in range(1,16):
        p=integer(row.get(f'h{s}{220+slot:04d}'),1)
        if p is None:continue
        residence=integer(row.get(f'h{s}{380+slot:04d}'),1)
        membership=integer(row.get(f'h{s}{420+slot:04d}'),1)
        if residence in (3,4) or membership in (3,4):continue
        person=people.get(p)
        if person is not None and integer(person.get(f'hhid{s}'),1)!=integer(row.get(f'hhid{s}'),1):
            audit['person_household_link_mismatch']+=1;person=None
        employed=integer(person.get(f'p{s}0201'),1) if person is not None else None
        employed=employed if employed in (1,2) else None
        member=dict(member_index=len(members)+1,age=integer(row.get(f'h{s}{360+slot:04d}'),0),
            relationship_to_head=category(row.get(f'h{s}{260+slot:04d}'),'relationship',maps,audit),
            residence=category(residence,'h__0381',maps,audit),
            employment_status=category(employed,'p__0201',maps,audit),employment_type=None,
            employment_position=None,occupation=None)
        if employed==1:
            member['employment_type']=category(person.get(f'p{s}0211'),'p__0211',maps,audit)
            member['employment_position']=category(person.get(f'p{s}0314'),'p__0314',maps,audit)
            for suffix,revision in [(352,7),(351,6),(350,5)]:
                code=integer(person.get(f'p{s}{suffix:04d}'),0)
                if code is not None:
                    label=occupations.get(str(code).zfill(3)) if revision==7 else None
                    member['occupation']=dict(classification='KSCO',revision=revision,code=str(code).zfill(3),label=label)
                    if label is None:audit['occupation_label_unavailable']+=1
                    break
            if member['occupation'] is None:audit['employed_occupation_missing']+=1
        if employed is None:audit['member_person_response_unknown']+=1
        members.append(member);links.append(dict(member_index=len(members),pid=p))
        if p==pid:target_index=len(members)
    assets={}
    for name,f,a in [('bank_deposit',2561,2562),('stocks_bonds_trusts',2563,2564)]:
        val=conditional_amount(row.get(f'h{s}{f}'),row.get(f'h{s}{a}'))
        assets[name]=dict(amount_krw=val['value_krw'],status=val['basis'])
        if val['value_krw'] is None:audit['missing_asset:'+name]+=1
    size=integer(row.get(f'h{s}0150'),1)
    if size is None:audit['reported_household_size_missing']+=1
    if target_index is None:audit['event_person_not_in_pre_event_roster']+=1
    result=dict(interview_year=w+1997,reported_household_size=size,members=members,
        housing_tenure=category(row.get(f'h{s}1406'),'h__1406',maps,audit),
        housing_type=category(row.get(f'h{s}1407'),'h__1407',maps,audit),
        assets=dict(unit='nominal_KRW_stock_at_interview',**assets))
    return result,target_index,dict(issues=dict(audit),member_links=links)


def make_input(y,state,past,incomes,shock):
    assert [r['reference_year'] for r in past]==[y-2,y-1]
    assert [r['reference_year'] for r in incomes]==[y,y+1,y+2]
    return dict(task='conditional_jobloss_consumption12_path_reconstruction',event_year=y,
        target_years=[y,y+1,y+2],output_unit='nominal_KRW_per_month_annual_average',
        household_before_event=state,past_observations=past,
        supplied_realized_household_income=incomes,supplied_realized_jobloss_path=shock)


def assert_safe_input(x):
    assert set(x)=={'task','event_year','target_years','output_unit','household_before_event',
                   'past_observations','supplied_realized_household_income','supplied_realized_jobloss_path'}
    y=x['event_year'];assert x['target_years']==[y,y+1,y+2]
    assert x['household_before_event']['interview_year']==y-1
    assert [o['reference_year'] for o in x['past_observations']]==[y-2,y-1]
    for o in x['past_observations']:assert o['consumption']['reference_year']<y
    assert [o['reference_year'] for o in x['supplied_realized_household_income']]==[y,y+1,y+2]
    forbidden={'pid','jobseq','family','group_id','record_id','case_id','observed_output','targets','survey_weight'}
    def walk(o):
        if isinstance(o,dict):
            assert not forbidden&set(o)
            assert not any(k.startswith('hhid') for k in o)
            for v in o.values():walk(v)
        elif isinstance(o,list):
            for v in o:walk(v)
    walk(x)


def main():
    raw=ROOT/'data/raw/klips_1_27차_stata';allocation=ROOT/'data/allocated/v17_snapshot_2026_10_08'
    out=ROOT/'outputs'/VERSION
    if out.exists():raise ValueError('Immutable version exists; use a new version')
    assert subprocess.run(['git','check-ignore','-q',str(out/'evaluation/inputs.jsonl')],cwd=ROOT).returncode==0
    manifest=json.loads((allocation/'manifest.json').read_text());sources={}
    for a in manifest['files']:
        p=allocation/a['file'];assert sha(p)==a['sha256'];sources[str(p.relative_to(ROOT))]=sha(p)
    sources[str((allocation/'manifest.json').relative_to(ROOT))]=sha(allocation/'manifest.json')
    roles={r['family']:r['role'] for r in read(allocation/'family_roles.jsonl')}
    parts={p:read(allocation/f) for p,(f,_,_) in PARTS.items()};all_events=sum(parts.values(),[])
    sets={}
    for part,rows in parts.items():
        _,role,n=PARTS[part];assert len(rows)==n==len({r['family'] for r in rows})
        assert all(roles[r['family']]==role for r in rows);sets[part]={r['family'] for r in rows}
    assert all(not a&b for a,b in combinations(sets.values(),2))
    maps_path=ROOT/'data/reference/klips_profile_codes_2026_09_27.json';occ_path=ROOT/'data/reference/KSCO7차_직업코드표.csv'
    maps=json.loads(maps_path.read_text())['maps']
    with occ_path.open(encoding='utf-8-sig',newline='') as f:
        occupations={r['코드'].zfill(3):r['항목명'] for r in csv.DictReader(f) if r['분류수준']=='소분류'}
    for p in (maps_path,occ_path):sources[str(p.relative_to(ROOT))]=sha(p)
    hh,cons,people,labels,columns={},{},{},{},{}
    state_waves={e['event_year']-1998 for e in all_events}
    waves=sorted({int(n.split(':')[0]) for e in all_events for n in e['household_nodes']})
    for w in waves:
        s=f'{w:02}';p=raw/f'klips{s}h.dta'
        with pd.read_stata(p,iterator=True) as r:lab=r.variable_labels()
        labels[w]=lab
        base={f'hhid{s}',f'hwave{s}'}
        base|={f'h{s}{v:04d}' for v in list(range(2101,2192))+[4001,4002,4003,4004,150,1406,1407,2561,2562,2563,2564]}
        base|={f'h{s}{b+i:04d}' for b in (220,260,360,380,420) for i in range(1,16)}
        cols=sorted(base&lab.keys());assert not any(c.startswith(f'h{s}23') for c in cols)
        df=pd.read_stata(p,columns=cols,convert_categoricals=False);df=df.loc[df[f'hwave{s}'].eq(1)]
        assert df[f'hhid{s}'].is_unique
        hh[w]={int(r[f'hhid{s}']):r for r in df.to_dict('records')}
        cc=[f'hhid{s}',f'hwave{s}']+[f'h{s}{suffix}' for suffix,_ in ITEMS.values()]
        cf=pd.read_stata(p,columns=cc,convert_categoricals=False);cf=cf.loc[cf[f'hwave{s}'].eq(1)]
        cons[w]={int(r[f'hhid{s}']):r for r in cf.to_dict('records')}
        columns[f'klips{s}h']=dict(context_income=cols,consumption=cc)
        sources[str(p.relative_to(ROOT))]=sha(p)
        if w in state_waves:
            pp=raw/f'klips{s}p.dta'
            with pd.read_stata(pp,iterator=True) as r:plab=r.variable_labels()
            pc=['pid',f'hhid{s}']+[f'p{s}{suffix:04d}' for suffix in (201,211,314,350,351,352)]
            pc=[c for c in pc if c in plab]
            pf=pd.read_stata(pp,columns=pc,convert_categoricals=False);assert pf.pid.is_unique
            people[w]={int(r['pid']):r for r in pf.to_dict('records')}
            columns[f'klips{s}p']=pc;sources[str(pp.relative_to(ROOT))]=sha(pp)
    jp=raw/'klips27w.dta'
    jobs=pd.read_stata(jp,columns=['pid','jobseq','jobwave']+JOB_FIELDS,convert_categoricals=False)
    jobs=jobs.loc[jobs.pid.isin({r['pid'] for r in all_events})].sort_values('jobwave')
    for c in JOB_FIELDS:jobs[c]=jobs[c].where(jobs[c]>=0 if c in ('j613','j614') else jobs[c]>0)
    histories={}
    for y in sorted({e['event_year'] for e in all_events}):
        cutoff=y-1994 # survey Y+3 contains annual observations for Y+2
        g=jobs.loc[jobs.jobwave<=cutoff].groupby(['pid','jobseq'],sort=False)
        last,conf=g[JOB_FIELDS].last(),g[JOB_FIELDS].nunique().gt(1);history=defaultdict(list)
        for (pid,seq),r in last.iterrows():
            v=r.to_dict();cf=[c for c in JOB_FIELDS if conf.loc[(pid,seq),c]]
            history[int(pid)].append(dict(jobseq=int(seq),values=v,conflicts=cf,start=bounds(v,'start',cf),end=bounds(v,'end',cf)))
        histories[y]=history
    sources[str(jp.relative_to(ROOT))]=sha(jp)
    packets={};audits=[];totals={};cache={}
    for part,events in parts.items():
        xs=[];ys=[];quality=Counter()
        for e in events:
            y=e['event_year'];nodes=[tuple(map(int,n.split(':'))) for n in e['household_nodes']]
            assert [w+1996 for w,h in nodes]==list(range(y-2,y+3))
            observations=[];income_audit=[]
            for w,h in nodes:
                if (w,h) not in cache:cache[w,h]=historical_income(hh[w][h],w,labels[w])
                inc=cache[w,h];income_audit.append(inc)
                observations.append(dict(reference_year=w+1996,income=income(inc),consumption=consumption(cons[w][h],w)))
            w,h=nodes[0]
            state,target_member,state_audit=household_state(hh[w][h],w,people[w],e['pid'],maps,occupations)
            history=histories[y].get(e['pid'],[]);job=next((j for j in history if j['jobseq']==e['jobseq']),None)
            path=employment_path(job,history,y);benefit=ui_payload(job)
            event_date=date_payload(job['end']) if job else None
            reason=integer(job['values'].get('j602'),1) if job and 'j602' not in job['conflicts'] else None
            shock=dict(event_year=y,termination_date=event_date,involuntary_jobloss=True,
                termination_reason_code=reason,termination_reason_code_system='KLIPS_j602',
                person_member_index=target_member,employment_path=path,unemployment_insurance=benefit,
                job_history_observation_cutoff_survey_year=y+3)
            x=make_input(y,state,observations[:2],[o['income'] for o in observations[2:]],shock)
            assert_safe_input(x)
            target=dict(unit='nominal_KRW_per_month_annual_average',years=[o['consumption'] for o in observations[2:]])
            cid=hashlib.sha256(f"{VERSION}:{e['family']}:{e['pid']}:{e['jobseq']}:{y}".encode()).hexdigest()[:24]
            xs.append(dict(case_id=cid,model_input=x));ys.append(dict(case_id=cid,observed_output=target))
            audits.append(dict(case_id=cid,family=e['family'],split=part,pid=e['pid'],jobseq=e['jobseq'],event_year=y,
                household_nodes=e['household_nodes'],income_source_audit=income_audit,state_audit=state_audit,
                job_record_conflicts=job['conflicts'] if job else [],job_record_available=job is not None,
                full_history_eligibility_date=e.get('exit_date_bounds'),eligibility_gap=e['gap']))
            quality.update(state_audit['issues']);quality['UI:'+benefit['status']]+=1;quality['reemployment:'+path['status']]+=1
            if event_date is None:quality['termination_date_missing_at_cutoff']+=1
            if benefit['status']=='received' and (benefit['daily_amount_krw'] is None or benefit['duration_days'] is None):quality['UI_received_missing_details']+=1
        packets[part]=(xs,ys);totals[part]=dict(cases=len(xs),families=len(sets[part]),quality=dict(quality))
    assert len(audits)==len({r['case_id'] for r in audits})==1168
    assert all(sha(ROOT/p)==h for p,h in sources.items())
    out.mkdir(parents=True)
    for part,(xs,ys) in packets.items():write(out/part/'inputs.jsonl',xs);write(out/part/'targets.jsonl',ys)
    rag=[dict(case_id=x['case_id'],model_input=x['model_input'],observed_output=y['observed_output']) for x,y in zip(*packets['training'])]
    write(out/'rag/jobloss_cases.jsonl',rag);write(out/'private/linkage_and_quality.jsonl',audits)
    schema=dict(version=VERSION,items={k:dict(source_suffix=s,label_ko=l) for k,(s,l) in ITEMS.items()},
        input_relative_consumption_years=[-2,-1],supplied_income_relative_years=[-2,-1,0,1,2],
        target_relative_years=[0,1,2],columns=columns,
        model_output_json_schema={'type':'object','properties':{f'year_{i}':{'type':'object','properties':{k:{'type':'number','minimum':0} for k in ITEMS},'required':list(ITEMS),'additionalProperties':False} for i in range(3)},'required':['year_0','year_1','year_2'],'additionalProperties':False},
        notes=['Three annual outputs in one conditional path task; no observed post-event consumption fed into later predictions.',
               'Job records observed through surveyY+3; supplied calendar job events stop atDec31(Y+2).',
               'No observed reemployment record is not confirmed continuous unemployment or job-search duration.',
               'UI daily wage-job amount:1000KRW; reported duration:days. No monthly180-day assumption or income double addition.',
               'Historical occupation codes keep the observed KSCO revision. Missing old names are not mapped to KSCO7 by guesswork.',
               'Nominal amounts only; CPI adjustment and evaluation metrics remain separate.'])
    jsave(out/'schema.json',schema)
    summary=dict(version=VERSION,status='LOCAL_1B_CASES_BUILT_NO_MODEL_CALLS',allocation_still_unfrozen=True,by_split=totals,
        rag_jobloss_cases=len(rag),event_year_range=[min(e['event_year'] for e in all_events),max(e['event_year'] for e in all_events)],
        family_overlap={f'{a}__{b}':len(sets[a]&sets[b]) for a,b in combinations(sets,2)},
        sources_sha256=sources,code_sha256=sha(Path(__file__)),
        dependencies_sha256={p:sha(ROOT/'code'/p) for p in ['audit_jobloss_inputs_v1.py','audit_five_year_jobloss_v1.py','build_phase1a_cases_v1.py','coupon_inputs.py','kr_consumption_validation_data.py','draft_feature_balanced_split_v4.py']},
        files_sha256={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()},external_calls=0,
        limitations=['Input/target construction, not retrieval/index/model evaluation.',
                     '1 representative event per family; additional99 eligible training events not serialized as extra examples.',
                     'Missing/conflicting job/UI/context details preserved, not imputed or excluded.',
                     'Full-history allocation eligibility and horizon-limited input job histories have different observation boundaries.',
                     'No CPI transformation, no power/sample-sufficiency claim, no actual policy path.'])
    jsave(out/'manifest.json',summary);jsave(ROOT/'docs/evidence'/f'{VERSION}.json',summary)
    print(json.dumps({k:summary[k] for k in ['by_split','rag_jobloss_cases','event_year_range','family_overlap']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
