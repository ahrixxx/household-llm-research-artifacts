"""Render public aggregate CSV only; requires matplotlib. No model/data calls."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
ROOT=Path(__file__).resolve().parents[1]
font=Path('/System/Library/Fonts/AppleSDGothicNeo.ttc')
if font.exists():
 font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
plt.rcParams.update({'font.size':11,'axes.unicode_minus':False,'svg.fonttype':'none','pdf.fonttype':42})
rows=list(csv.DictReader((ROOT/'03_results/metrics.csv').open()))
out=ROOT/'03_results/figures';out.mkdir(exist_ok=True)
order=['llm_v61','llm_v6','catboost_mae','persistence','ridge','lasso','median_l1','rf_mae','rf_mse','gbm_mae']
labels={'llm_v61':'LLM','llm_v6':'LLM','catboost_mae':'CatBoost','persistence':'전년 소비 유지','ridge':'Ridge','lasso':'Lasso','median_l1':'중앙값 회귀','rf_mae':'RF-MAE','rf_mse':'RF-MSE','gbm_mae':'HistGBM','catboost':'CatBoost','catboost_llm':'CatBoost 보조 LLM'}
def panel(ax,data,key,limit,title):
 y=list(range(len(data)));vals=[float(r[key])/10000 for r in data]
 colors=['#3659D9' if r['model'].startswith('llm') else '#15857B' if r['model']=='catboost_llm' else '#738293' for r in data]
 ax.barh(y,vals,height=.55,color=colors,zorder=3)
 ax.set_yticks(y,[labels[r['model']] for r in data]);ax.invert_yaxis();ax.set_xlim(0,limit)
 ax.set_title(title,loc='left',fontsize=13,fontweight='bold',pad=16,color='#202C3B')
 ax.xaxis.grid(True,color='#E8ECF1',zorder=0);ax.set_axisbelow(True)
 for sp in ax.spines.values():sp.set_visible(False)
 ax.tick_params(axis='both',length=0,labelcolor='#4C5868');ax.set_xlabel('MAE · 만 원/월',color='#667285',labelpad=9)
 for i,v in enumerate(vals):ax.text(v+limit*.015,i,f'{v:.2f}',va='center',fontsize=10,color='#27354A')
def draw(cohorts,name,title,sub):
 fig,axs=plt.subplots(len(cohorts),2,figsize=(14,8 if len(cohorts)==1 else 11),squeeze=False)
 fig.patch.set_facecolor('white')
 for i,(cohort,caption) in enumerate(cohorts):
  data=sorted([r for r in rows if r['cohort']==cohort],key=lambda r:order.index(r['model']) if r['model'] in order else 99)
  panel(axs[i,0],data,'item_mae_krw',12 if 'education' in cohort else 6 if 'assisted' in cohort else 9.5,caption+'  |  항목 평균 오차')
  panel(axs[i,1],data,'total_mae_krw',100 if 'education' in cohort else 35 if 'assisted' in cohort else 75,caption+'  |  총소비 오차')
 fig.suptitle(title,x=.035,y=.97,ha='left',fontsize=22,fontweight='bold',color='#182638')
 fig.text(.035,.89 if len(cohorts)==1 else .925,sub,fontsize=11,color='#667285')
 fig.subplots_adjust(left=.19 if len(cohorts)==1 else .14,right=.94,top=.76 if len(cohorts)==1 else .85,bottom=.23 if len(cohorts)==1 else .16,hspace=.55,wspace=.65 if len(cohorts)==1 else .4)
 footnote = '출처: 개발 실험 집계 metrics.csv · 2023년 실질금액 · 막대가 짧을수록 오차가 작음\n개발 결과의 점추정치 비교이며 최종 평가 또는 통계적 우위 판정을 뜻하지 않음'
 if any(cohort == '1B_new' for cohort, _ in cohorts):
  footnote = '1-B: 가구별 평가 대상 연도 자료 총 52건\n' + footnote
 fig.text(.035,.025,footnote,fontsize=10,color='#667285',linespacing=1.6)
 for ext in ['png','svg','pdf']:fig.savefig(out/f'{name}.{ext}',dpi=190,facecolor='white')
 plt.close(fig)
draw([('1A_original','1-A · 98가구'),('1B_new','1-B · 40가구')],'baseline_comparison','1차 개발 실험 | 전체 기준선 비교','같은 패널의 모든 모형은 같은 평가 집단에서 비교 · 두 과제는 별도 해석')
draw([('1A_education_diagnostic','교육변화 진단 · 53가구')],'education_diagnostic','교육변화 가구 | 추가 진단','목적표집한 별도 집단 · 기존 98가구와 합산하지 않음')
order[:]=['llm_v61','catboost','catboost_llm']
draw([('1A_catboost_assisted_matched','공통 96가구')],'catboost_assisted','CatBoost 보조 입력 | 동일 가구 비교','기존 LLM · CatBoost 단독 · CatBoost 보조 LLM의 공통 표본 비교')
print(out)
