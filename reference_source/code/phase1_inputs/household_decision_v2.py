"""One-call structured decision summaries, not private reasoning traces.

Single point estimate per year/item. Diagnostic disagreement never repairs amounts.
"""
from pathlib import Path
import json,math,statistics
from korean_household_v2 import ITEMS
from korean_contract_v2 import FACTS as B_FACTS,facts as b_facts,pointers,resolve,equal,score_prediction
def is_b(x):return 'event_year' in x
def years(x):return x['target_years'] if is_b(x) else [x['target_year']]
def base_facts(x):
 if is_b(x):return b_facts(x)
 s=x['household_at_2022_interview'];c=x['past_2022']['consumption'];inc=x['realized_2023_income']
 vals=[s['interview_year'],years(x),x['output_unit'],[c['reference_year']],[c['unit']],[inc['reference_year']],[inc['unit']],[inc['total_krw']],s['assets']['unit'],s['assets']['bank_deposit'],[{'가구원번호':m['member_index'],'상태':m['employment_status']} for m in s['members']],[{'가구원번호':m['member_index'],'교육':m['education_at_interview']} for m in s['members']]]
 return dict(zip(B_FACTS[:12],vals))
def facts(x):
 result=base_facts(x)
 result['목표연도 가구상태 확인']=[{'관측연도':v['관측상태']['interview_year'],'가구원수':v['관측상태']['reported_household_size'],'가구원번호':[m['member_index'] for m in v['관측상태']['members']],'입주형태':v['관측상태']['housing_tenure'],'주택종류':v['관측상태']['housing_type'],'변화시점':'미확인'} for v in x['목표연도 가구상태']]
 return result
FIELDS={'관측근거','불확실성','기준연도','기준소비액','변화방향','방향판단근거','변화금액','크기판단근거','최종소비추정액'}
def anchor(x):
 if not is_b(x):return x['past_2022']['consumption']
 rows=[r for r in x['past_observations'] if r['reference_year']==x['event_year']-1]
 if len(rows)!=1:raise ValueError('Require one Y-1 anchor')
 return rows[0]['consumption']
def messages(x):
 system=Path(__file__).with_name('household_decision_1b_v2_target_state.txt' if is_b(x) else 'household_decision_1a_v2_target_state.txt').read_text()
 payload={'가구입력':x,'사실확인필드':list(facts(x)),'사실확인방법':'가구입력의 해당 값이나 객체를 그대로 보고한다. 가구원취업상태는 {가구원번호,상태}, 가구원교육상태는 {가구원번호,교육}. 예금은 bank_deposit 객체. 퇴직 관련 사실은 1-B에만 해당한다.','소비항목':list(ITEMS.values()),'관측근거경로':pointers(x)}
 return [{'role':'system','content':system},{'role':'user','content':json.dumps(payload,ensure_ascii=False,allow_nan=False)}]
def num(x):return type(x) in (int,float) and math.isfinite(x)
def sign(x):return '감소' if x<0 else '증가' if x>0 else '유지'
def close(a,b):return num(a) and math.isclose(a,b,rel_tol=0,abs_tol=1e-6)
def parse(text,x):
 def pairs(rows):
  d={}
  for k,v in rows:
   if k in d:raise ValueError('Duplicate JSON key')
   d[k]=v
  return d
 def invalid(_):raise ValueError('Nonfinite JSON')
 o=json.loads(text,object_pairs_hook=pairs,parse_constant=invalid)
 if not isinstance(o,dict) or set(o)!={'사실확인','항목별판단'}:raise ValueError('Output fields')
 if not isinstance(o['사실확인'],dict) or set(o['사실확인'])!=set(facts(x)):raise ValueError('Fact schema')
 if not isinstance(o['항목별판단'],dict) or set(o['항목별판단'])!={str(y) for y in years(x)}:raise ValueError('Target years')
 a=anchor(x);pred={};checks=[];refs=[];structure=[]
 for year,items in o['항목별판단'].items():
  if not isinstance(items,dict) or set(items)!=set(ITEMS.values()):raise ValueError('12 Korean items required')
  pred[year]={}
  for key,label in ITEMS.items():
   row=items[label]
   if not isinstance(row,dict):raise ValueError('Item object required')
   amount=row.get('최종소비추정액')
   if not num(amount) or amount<0:raise ValueError('Final amount finite nonnegative')
   pred[year][key]=amount
   tag={'year':year,'item':label}
   if set(row)!=FIELDS:structure.append({**tag,'error':'judgment_fields'})
   for field in ['불확실성','방향판단근거','크기판단근거']:
    if not isinstance(row.get(field),str) or not row[field].strip():structure.append({**tag,'error':field})
   ev=row.get('관측근거')
   if not isinstance(ev,list) or not ev:structure.append({**tag,'error':'관측근거'})
   else:
    for r in ev:
     try:ok=isinstance(r,dict) and set(r)=={'경로','값'} and equal(resolve(x,r['경로']),r['값'])
     except (ValueError,KeyError,IndexError,TypeError):ok=False
     if not ok:refs.append(tag)
   base=a['items'][label];delta=amount-base
   checks.append({**tag,'anchor_year_correct':type(row.get('기준연도')) is int and row['기준연도']==a['reference_year'],'anchor_amount_correct':close(row.get('기준소비액'),base),'direction_matches_amount':row.get('변화방향')==sign(delta),'change_matches_amount':close(row.get('변화금액'),delta)})
 return {'prediction':pred,'reported_judgments':o['항목별판단'],'facts_match':{k:equal(v,o['사실확인'][k]) for k,v in facts(x).items()},'reference_errors':refs,'judgment_structure_errors':structure,'direction_amount_checks':checks,'diagnostic_errors_require_retry':False}
def synthetic(x):
 a=anchor(x);o={'사실확인':facts(x),'항목별판단':{}}
 idx=next(i for i,v in enumerate(x['past_observations']) if v['reference_year']==a['reference_year']) if is_b(x) else None
 for year in years(x):
  block={}
  for j,label in enumerate(ITEMS.values()):
   base=a['items'][label];delta=1000 if j%3==0 else -min(base,1000) if j%3==1 else 0
   block[label]={'관측근거':[{'경로':f'/past_observations/{idx}/consumption/items/{label}' if is_b(x) else f'/past_2022/consumption/items/{label}','값':base}],'불확실성':'합성 검사: 실제 판단 아님','기준연도':a['reference_year'],'기준소비액':base,'변화방향':sign(delta),'방향판단근거':'구조 검사를 위한 합성 방향','변화금액':delta,'크기판단근거':'산술 검사를 위한 합성 금액','최종소비추정액':base+delta}
  o['항목별판단'][str(year)]=block
 return o

def describe_repeats(parsed_draws):
 """Descriptive outputs only: does not choose a primary scoring estimator."""
 if not parsed_draws:raise ValueError('No valid draws')
 first=parsed_draws[0]['prediction']
 if any(set(r['prediction'])!=set(first) for r in parsed_draws):raise ValueError('Different years')
 out={}
 for y in first:
  out[y]={}
  for item in ITEMS:
   vals=[r['prediction'][y][item] for r in parsed_draws]
   out[y][item]={'valid_draws':len(vals),'repeat_mean':statistics.mean(vals),'repeat_median':statistics.median(vals),'repeat_sample_sd':statistics.stdev(vals) if len(vals)>1 else None}
 return out
