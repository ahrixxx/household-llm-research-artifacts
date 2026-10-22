"""Build 12-item 2022->2023 conditional reconstruction cases from v17.

Local only. Strict input column allowlists; target consumption is read in a
separate table. No target-dependent sampling, model fitting, retrieval or calls.
"""
from collections import Counter
from itertools import combinations
import csv
import hashlib
import json
from pathlib import Path
import subprocess

import pandas as pd
from coupon_inputs import amount, conditional_amount, integer
from kr_consumption_validation_data import annual_income, income_specs, digest
from draft_feature_balanced_split_v4 import sha

ROOT=Path(__file__).resolve().parents[1]
VERSION='phase1a_consumption12_cases_v1'
ALLOCATION='data/allocated/v17_snapshot_2026_10_08'
ITEMS={
    'food':(2311,'식비'), 'dining_out':(2312,'외식비'),
    'public_education':(2313,'공교육비'), 'private_education':(2314,'사교육비'),
    'housing':(2316,'주거비'), 'healthcare':(2318,'보건의료비'),
    'recreation':(2319,'교양오락비'), 'durables':(2320,'내구재 구입비'),
    'communication':(2321,'통신비'), 'clothing':(2326,'피복비'),
    'public_transport':(2329,'대중교통비'), 'daily_necessities':(2330,'생필품 구입비'),
}
PARTS={'training':('training_current.jsonl','training',7020),
       'development':('development_current_cases.jsonl','development',98),
       'evaluation':('phase1_a.jsonl','phase1_a',1000)}


def read(path):return [json.loads(line) for line in path.read_text().splitlines()]
def write(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in rows))
def jsave(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def input_columns():
    cols={25:{'hhid25','hwave25','h250150','h251406','h251407','h252561','h252562','h252563','h252564'},
          26:{'hhid26','hhid25','hwave26'},27:{'hhid27','hhid26','hhid25','hwave27'}}
    cols[25]|={f'h25{b+i:04d}' for b in (220,260,360,380,420) for i in range(1,16)}
    cols[26]|={f'h26{s}' for s,_ in ITEMS.values()}
    for w in (26,27):
        for flag,fields in income_specs(w).values():cols[w].update([flag,*fields])
        cols[w]|={f'h{w}{s}' for s in (2132,4001,4003)}
    assert not any(c.startswith('h2723') for c in cols[27])
    return {w:sorted(c) for w,c in cols.items()}


def consumption(row,wave):
    values={key:amount(row.get(f'h{wave}{s}'))['value_krw'] for key,(s,_) in ITEMS.items()}
    if any(v is None for v in values.values()):raise ValueError('incomplete 12-item consumption')
    return dict(reference_year=wave+1996,unit='nominal_KRW_per_month_annual_average',
                items=values,total_12_items_krw=sum(values.values()))


def category(value,key,maps,audit):
    code=integer(value,0)
    if code is None:
        audit['missing_category:'+key]+=1
        return dict(code=None,label=None,status='unknown')
    label=maps[key]['values'].get(str(code))
    if label is None:audit['unmapped_category:'+key]+=1
    return dict(code=code,label=label,status='observed' if label else 'unmapped_code')


def income_payload(value):
    if value['total_krw'] is None:raise ValueError('incomplete annual household income')
    return dict(reference_year=value['reference_year'],unit=value['unit'],
        definition=value['definition'],total_krw=value['total_krw'],
        components_krw={name:r['value_krw'] for name,r in value['components'].items()})


def make_input(h25,h26,income_only_h27,people,maps,occupations):
    """Never accepts a target table or copies arbitrary source-row fields."""
    audit=Counter();members=[];member_links=[]
    for slot in range(1,16):
        pid=integer(h25.get(f'h25{220+slot:04d}'),1)
        if pid is None:continue
        residence=integer(h25.get(f'h25{380+slot:04d}'),1)
        membership=integer(h25.get(f'h25{420+slot:04d}'),1)
        if residence in (3,4) or membership in (3,4):
            audit['departed_or_deceased_slots_excluded']+=1;continue
        age=integer(h25.get(f'h25{360+slot:04d}'),0)
        person=people.get(pid)
        if person is not None and integer(person.get('hhid25'),1)!=integer(h25.get('hhid25'),1):
            audit['person_household_mismatch']+=1;person=None
        state=integer(person.get('p250201'),1) if person is not None else None
        state=state if state in (1,2) else None
        member=dict(member_index=len(members)+1,age=age,
            relationship_to_head=category(h25.get(f'h25{260+slot:04d}'),'relationship',maps,audit),
            residence=category(residence,'h__0381',maps,audit),
            employment_status=category(state,'p__0201',maps,audit),
            employment_type=None,employment_position=None,occupation_ksco7=None)
        if state==1:
            member['employment_type']=category(person.get('p250211'),'p__0211',maps,audit)
            member['employment_position']=category(person.get('p250314'),'p__0314',maps,audit)
            code=integer(person.get('p250352'),0)
            label=occupations.get(str(code).zfill(3)) if code is not None else None
            member['occupation_ksco7']=dict(code=str(code).zfill(3) if code is not None else None,
                                           label=label,status='observed' if label else 'unknown')
            audit['employed_members']+=1
            if label is None:audit['employed_members_without_occupation_label']+=1
        elif state is None:
            audit['members_without_confirmed_person_response']+=1
            if age is not None and age>=15:audit['members_age15plus_without_confirmed_person_response']+=1
        if age is None:audit['members_without_age']+=1
        if residence==2:audit['noncohabiting_members']+=1
        if residence not in (1,2):audit['members_without_confirmed_residence']+=1
        members.append(member);member_links.append(dict(member_index=len(members),pid=pid,roster_slot=slot))
    assert len({r['pid'] for r in member_links})==len(member_links)
    size=integer(h25.get('h250150'),1)
    if size is None:raise ValueError('missing previously required household size')
    cohab=sum(m['residence']['code']==1 for m in members)
    audit['cohabiting_member_count_differs_reported_size']+=int(cohab!=size)
    audit['head_count_not_one']+=int(sum(m['relationship_to_head']['code']==10 for m in members)!=1)
    assets={}
    for name,flag,val in [('bank_deposit',2561,2562),('stocks_bonds_trusts',2563,2564)]:
        obj=conditional_amount(h25.get(f'h25{flag}'),h25.get(f'h25{val}'))
        assets[name]=dict(amount_krw=obj['value_krw'],status=obj['basis'],missing_reason=obj['reason'])
        audit['missing_asset:'+name]+=int(obj['value_krw'] is None)
    inc22,inc23=annual_income(h26,26),annual_income(income_only_h27,27)
    model_input=dict(task='conditional_consumption12_reconstruction',target_year=2023,
        output_unit='nominal_KRW_per_month_annual_average',
        household_at_2022_interview=dict(interview_year=2022,reported_household_size=size,
            members=members,housing_tenure=category(h25.get('h251406'),'h__1406',maps,audit),
            housing_type=category(h25.get('h251407'),'h__1407',maps,audit),
            assets=dict(unit='nominal_KRW_stock_at_2022_interview',**assets)),
        past_2022=dict(consumption=consumption(h26,26),income=income_payload(inc22)),
        realized_2023_income=income_payload(inc23))
    return model_input,dict(issues=dict(audit),member_links=member_links,
                           income_source_audit={'2022':inc22,'2023':inc23})


def assert_safe_input(payload):
    assert set(payload)=={'task','target_year','output_unit','household_at_2022_interview','past_2022','realized_2023_income'}
    assert set(payload['past_2022'])=={'consumption','income'}
    assert payload['past_2022']['consumption']['reference_year']==2022
    assert payload['realized_2023_income']['reference_year']==2023
    forbidden={'pid','family','group_id','record_id','case_id','observed_output','target','targets','weight','survey_weight'}
    def walk(value):
        if isinstance(value,dict):
            for k,v in value.items():
                assert k not in forbidden and not k.startswith('hhid') and not k.startswith('h2723')
                walk(v)
        elif isinstance(value,list):
            for v in value:walk(v)
    walk(payload)


def main():
    allocation=ROOT/ALLOCATION;raw=ROOT/'data/raw/klips_1_27차_stata'
    out=ROOT/'outputs'/VERSION
    if out.exists():raise ValueError('Immutable output version already exists; choose a new version.')
    assert subprocess.run(['git','check-ignore','-q',str(out/'evaluation/inputs.jsonl')],cwd=ROOT).returncode==0
    allocation_manifest=json.loads((allocation/'manifest.json').read_text())
    sources={str((allocation/'manifest.json').relative_to(ROOT)):sha(allocation/'manifest.json')}
    for item in allocation_manifest['files']:
        p=allocation/item['file'];assert sha(p)==item['sha256'];sources[str(p.relative_to(ROOT))]=sha(p)
    roles={r['family']:r['role'] for r in read(allocation/'family_roles.jsonl')}
    parts={part:read(allocation/name) for part,(name,_,_) in PARTS.items()}
    sets={}
    for part,rows in parts.items():
        _,role,n=PARTS[part];assert len(rows)==n
        assert len({r['record_id'] for r in rows})==n
        sets[part]={r['group_id'] for r in rows}
        assert all(roles[r['group_id']]==role for r in rows)
    assert all(not a&b for a,b in combinations(sets.values(),2))
    maps_path=ROOT/'data/reference/klips_profile_codes_2026_09_27.json'
    occ_path=ROOT/'data/reference/KSCO7차_직업코드표.csv'
    maps=json.loads(maps_path.read_text())['maps']
    with occ_path.open(encoding='utf-8-sig',newline='') as f:
        occupations={r['코드'].zfill(3):r['항목명'] for r in csv.DictReader(f) if r['분류수준']=='소분류'}
    for p in (maps_path,occ_path):sources[str(p.relative_to(ROOT))]=sha(p)
    tables={};column_labels={};columns=input_columns()
    for w,cols in columns.items():
        path=raw/f'klips{w}h.dta'
        with pd.read_stata(path,iterator=True) as reader:labels=reader.variable_labels()
        column_labels[f'input_klips{w}h']={c:labels[c] for c in cols}
        df=pd.read_stata(path,columns=cols,convert_categoricals=False)
        df=df.loc[df[f'hwave{w}'].eq(1)&df[f'hhid{w}'].gt(0)]
        assert df[f'hhid{w}'].is_unique
        tables[w]={int(r[f'hhid{w}']):r for r in df.to_dict('records')}
        sources[str(path.relative_to(ROOT))]=sha(path)
    pcols=['pid','hhid25','p250201','p250211','p250314','p250352']
    path=raw/'klips25p.dta'
    with pd.read_stata(path,iterator=True) as reader:labels=reader.variable_labels()
    column_labels['input_klips25p']={c:labels[c] for c in pcols}
    df=pd.read_stata(path,columns=pcols,convert_categoricals=False);assert df.pid.is_unique
    people={int(r['pid']):r for r in df.to_dict('records')};sources[str(path.relative_to(ROOT))]=sha(path)
    # Independent target-only read. Never feed this table to make_input.
    target_cols=['hhid27','hwave27']+[f'h27{s}' for s,_ in ITEMS.values()]
    path=raw/'klips27h.dta'
    with pd.read_stata(path,iterator=True) as reader:labels=reader.variable_labels()
    column_labels['target_klips27h']={c:labels[c] for c in target_cols}
    df=pd.read_stata(path,columns=target_cols,convert_categoricals=False)
    df=df.loc[df.hwave27.eq(1)&df.hhid27.gt(0)];assert df.hhid27.is_unique
    target_rows={int(r['hhid27']):r for r in df.to_dict('records')}
    lookup={digest(f'27:{hid}'):hid for hid in tables[27]}
    records={};audit_rows=[];split_audits={};counts={};family_counts={}
    # Construct and validate everything before creating the output directory.
    for part,rows in parts.items():
        inputs=[];targets=[];issues=Counter()
        for entry in rows:
            rid=entry['record_id'];hid27=lookup[rid];r27=tables[27][hid27]
            hid26=integer(r27['hhid26'],1);hid25=integer(r27['hhid25'],1)
            r25,r26=tables[25][hid25],tables[26][hid26]
            assert integer(r26['hhid25'],1)==hid25
            payload,audit=make_input(r25,r26,r27,people,maps,occupations)
            assert_safe_input(payload)
            y=consumption(target_rows[hid27],27)
            cid=hashlib.sha256(f'{VERSION}:{rid}'.encode()).hexdigest()[:24]
            inputs.append(dict(case_id=cid,model_input=payload))
            targets.append(dict(case_id=cid,observed_output=y))
            audit_rows.append(dict(case_id=cid,record_id=rid,family=entry['group_id'],split=part,
                source_households={'25':hid25,'26':hid26,'27':hid27},**audit))
            issues.update(audit['issues'])
        assert [r['case_id'] for r in inputs]==[r['case_id'] for r in targets]
        records[part]=(inputs,targets);split_audits[part]=dict(issues)
        counts[part]=len(inputs);family_counts[part]=len(sets[part])
    assert len({r['case_id'] for r in audit_rows})==sum(counts.values())
    source_unchanged=all(sha(ROOT/p)==h for p,h in sources.items());assert source_unchanged
    out.mkdir(parents=True)
    for part,(inputs,targets) in records.items():
        write(out/part/'inputs.jsonl',inputs);write(out/part/'targets.jsonl',targets)
    train_x,train_y=records['training']
    rag=[dict(case_id=x['case_id'],model_input=x['model_input'],observed_output=y['observed_output']) for x,y in zip(train_x,train_y)]
    write(out/'rag/general_cases.jsonl',rag)
    write(out/'private/linkage_and_quality.jsonl',audit_rows)
    schema=dict(version=VERSION,task='conditional_consumption12_reconstruction',
        items=[dict(key=k,label_ko=label,source_suffix=s) for k,(s,label) in ITEMS.items()],
        years={'household_state_interview':2022,'past_consumption_income':2022,'supplied_realized_income':2023,'target_consumption':2023},
        model_output_json_schema={'type':'object','properties':{k:{'type':'number','minimum':0} for k in ITEMS},
                                  'required':list(ITEMS),'additionalProperties':False},
        units={'consumption':'nominal_KRW_per_month_annual_average','income':'nominal_KRW_per_year','assets':'nominal_KRW_stock_at_2022_interview'},
        input_columns=columns,person_columns=pcols,target_columns=target_cols,
        notes=['Sum only the 12 items in code; do not normalize to reported total living expenses.',
               'Input/target record envelopes have opaque case IDs; model_input contains no global identifiers or role.',
               'Same input payload structure across training, development and evaluation.',
               'Missing optional context is null/unknown, never silently zero or future-filled.',
               '2023 realized income is an explicit condition; this is not an ex-ante 2022 forecast.',
               'Annual household income retains irregular receipts; phase2 one-off separation is not applied.',
               'General RAG cases only, not yet historical 1-B cases or a fitted retrieval index.'])
    jsave(out/'schema.json',schema)
    summary=dict(version=VERSION,status='LOCAL_1A_CASES_BUILT_NO_MODEL_CALLS',allocation_version='joint_allocation_v17_five_year_draft',
        allocation_still_unfrozen=True,counts=counts,family_counts=family_counts,rag_general_cases=len(rag),
        target_items=len(ITEMS),split_family_overlap={f'{a}__{b}':len(sets[a]&sets[b]) for a,b in combinations(sets,2)},
        family_overlap_with_B_or_policy={part:len(fs&{f for f,role in roles.items() if role in ('phase1_b','phase2')}) for part,fs in sets.items()},
        quality_counts=split_audits,source_variable_labels=column_labels,source_sha256=sources,
        source_files_unchanged=source_unchanged,code_sha256=sha(Path(__file__)),
        dependency_sha256={name:sha(ROOT/'code'/name) for name in ['coupon_inputs.py','kr_consumption_validation_data.py','draft_feature_balanced_split_v4.py']},
        file_sha256={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()},
        output_directory=str(out.relative_to(ROOT)),external_calls=0,
        limitations=['Not a frozen experiment or fitted retrieval system.','Development uses98 current-input-complete families, not all100 reserved development families.',
                     'Remaining optional member/job/asset/housing missingness is disclosed, not exclusion or imputation.',
                     'All7020 training records retained;5562 training families means multiple records per family. Retrieval needs an explicit family repetition limit.',
                     'Nominal2022/2023 amounts only; no CPI correction, outlier clipping or metric fitting.',
                     'New case microdata remains local; prior snapshot push was a scoped exception, not automatic publication authorization.'])
    jsave(out/'manifest.json',summary)
    jsave(ROOT/'docs/evidence'/f'{VERSION}.json',summary)
    print(json.dumps({k:summary[k] for k in ['version','counts','family_counts','rag_general_cases','quality_counts','split_family_overlap','family_overlap_with_B_or_policy']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
