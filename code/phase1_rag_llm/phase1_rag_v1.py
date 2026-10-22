"""Phase-1 KLIPS case retrieval (1-A general / 1-B jobloss). Offline, numpy only.

Corpus = training-family RAG cases only. Retrieval uses allowed input features,
never a query's target. Preprocessing (median/scale) is fitted on the corpus.
Priority weights, k=5 and one-case-per-family are the UNFROZEN draft in
research/retrieval.md; every value is an explicit config field and is logged.
Family IDs come from the private linkage file and never enter model messages.
"""
from dataclasses import dataclass, field, asdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
VERSION = 'phase1_rag_v1'
ITEMS = ['food', 'dining_out', 'public_education', 'private_education', 'housing', 'healthcare',
         'recreation', 'durables', 'communication', 'clothing', 'public_transport', 'daily_necessities']
TASKS = {
    '1a': dict(cases='phase1a_consumption12_cases_v2_member_earnings', corpus='rag/general_cases.jsonl',
               input_task='conditional_consumption12_reconstruction'),
    '1b': dict(cases='phase1b_consumption12_cases_v3_training276', corpus='rag/jobloss_cases.jsonl',
               input_task='conditional_jobloss_consumption12_path_reconstruction'),
}
PURCHASING_POWER_BASE_YEAR = 2023  # user-confirmed base; CPI series itself is supplied separately
CHILD_MAX_AGE, ELDERLY_MIN_AGE = 17, 65  # draft feature definitions
EMPLOYED_CODE = 1  # p__0201: 1 employed, 2 not employed
HEAD_CODE = 10


@dataclass
class RetrievalConfig:
    """Draft defaults from research/retrieval.md; not a frozen setting."""
    k: int = 5
    max_per_family: int = 1
    # Priority group -> weight. Groups 1..5 follow the retrieval.md table order.
    group_weights: dict = field(default_factory=lambda: {'1': 5.0, '2': 4.0, '3': 3.0, '4': 2.0, '5': 1.0})
    status: str = 'DRAFT_UNFROZEN'

    def digest(self):
        return sha_json(asdict(self))


def sha_json(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()).hexdigest()


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


class Deflator:
    """amount_t * CPI(2023)/CPI(t). Index series must be supplied, never invented here."""

    def __init__(self, index_by_year=None, source=None):
        self.index = {int(y): float(v) for y, v in (index_by_year or {}).items()}
        self.source = source
        if self.index:
            if PURCHASING_POWER_BASE_YEAR not in self.index:
                raise ValueError('CPI series must contain the 2023 base year')
            if any(not (v > 0 and math.isfinite(v)) for v in self.index.values()):
                raise ValueError('CPI index values must be positive and finite')

    @property
    def nominal(self):
        return not self.index

    @classmethod
    def from_file(cls, path):
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        return cls(data['index_by_year'], source=dict(path=str(path), sha256=sha_file(path),
                                                       description=data.get('source')))

    def __call__(self, amount, year):
        if amount is None:
            return None
        if self.nominal:
            return float(amount)
        if int(year) not in self.index:
            raise ValueError(f'CPI index missing for year {year}')
        return float(amount) * self.index[PURCHASING_POWER_BASE_YEAR] / self.index[int(year)]

    def describe(self):
        return dict(mode='nominal_no_cpi' if self.nominal else 'real_2023_purchasing_power',
                    base_year=PURCHASING_POWER_BASE_YEAR, source=self.source)


# ---------------------------------------------------------------- features

def log_amount(x):
    """Signed log in 10k-KRW units; tolerates zero and negative income."""
    return None if x is None else math.copysign(math.log1p(abs(x) / 1e4), x)


def log_ratio(new, old):
    a, b = log_amount(new), log_amount(old)
    return None if a is None or b is None else a - b


def shares(items):
    total = sum(items.values())
    return {f'share_{k}': (items[k] / total if total > 0 else None) for k in ITEMS}


def code_of(obj):
    return obj.get('code') if isinstance(obj, dict) else None


def household_counts(members):
    ages = [m.get('age') for m in members]
    known = [a for a in ages if a is not None]
    statuses = [code_of(m.get('employment_status')) for m in members]
    complete_ages = bool(members) and len(known) == len(members)
    complete_employment = bool(members) and all(v in (1, 2) for v in statuses)
    return dict(
        children=sum(a <= CHILD_MAX_AGE for a in known) if complete_ages else None,
        elderly=sum(a >= ELDERLY_MIN_AGE for a in known) if complete_ages else None,
        employed=sum(v == EMPLOYED_CODE for v in statuses) if complete_employment else None)


def occupation_major(member, key):
    occ = member.get(key) if member else None
    code = occ.get('code') if isinstance(occ, dict) else None
    if not code:
        return None
    revision = occ.get('revision', 7)  # 1-A records KSCO7 only
    return f'ksco{revision}:{str(code)[0]}'  # never mapped across revisions


def features_1a(x, deflate):
    """Groups follow the retrieval.md 1-A priority table."""
    hh = x['household_at_2022_interview']
    past = x['past_2022']
    cons, inc22, inc23 = past['consumption'], past['income'], x['realized_2023_income']
    y_c, y_i, y_r = cons['reference_year'], inc22['reference_year'], inc23['reference_year']
    members = hh['members']
    counts = household_counts(members)
    head = next((m for m in members if code_of(m.get('relationship_to_head')) == HEAD_CODE), None)
    assets = hh['assets']
    real22, real23 = deflate(inc22['total_krw'], y_i), deflate(inc23['total_krw'], y_r)
    num = {
        '1': dict(log_total_consumption=log_amount(deflate(cons['total_12_items_krw'], y_c)), **shares(cons['items'])),
        '2': dict(log_income_past=log_amount(real22), income_change=log_ratio(real23, real22)),
        '3': dict(household_size=hh.get('reported_household_size'), **counts),
        '4': dict(log_deposit=log_amount(deflate(assets['bank_deposit']['amount_krw'], hh['interview_year'])),
                  log_stocks=log_amount(deflate(assets['stocks_bonds_trusts']['amount_krw'], hh['interview_year']))),
        '5': dict(head_age=head.get('age') if head else None),
    }
    cat = {
        '4': dict(housing_tenure=code_of(hh.get('housing_tenure'))),
        '5': dict(head_occupation=occupation_major(head, 'occupation_ksco7')),
    }
    return num, cat


def features_1b(x, deflate):
    """Groups follow the retrieval.md 1-B priority table."""
    hh = x['household_before_event']
    past = x['past_observations']  # [Y-2, Y-1]
    incomes = x['supplied_realized_household_income']  # [Y, Y+1, Y+2]
    shock = x['supplied_realized_jobloss_path']
    real = lambda d: deflate(d['total_krw'], d['reference_year'])
    base = real(past[1]['income'])
    c1, c0 = past[1]['consumption'], past[0]['consumption']
    path = shock.get('employment_path') or {}
    first = path.get('first_reemployment')
    if first is not None:
        reemp_days = first.get('elapsed_days_min')
    else:
        reemp_days = None
    ui = shock.get('unemployment_insurance') or {}
    ui_received = {'received': 1.0, 'not_received_or_not_applied': 0.0}.get(ui.get('status'))
    members = hh['members']
    idx = shock.get('person_member_index')
    person = members[idx - 1] if idx else None
    counts = household_counts(members)
    person_status = code_of(person.get('employment_status')) if person else None
    other_employed = (counts['employed'] - int(person_status == EMPLOYED_CODE)
                      if counts['employed'] is not None and person_status in (1, 2) else None)
    assets, iy = hh['assets'], hh['interview_year']
    num = {
        '1': dict(income_pre_change=log_ratio(base, real(past[0]['income'])),
                  **{f'income_change_y{i}': log_ratio(real(incomes[i]), base) for i in range(3)}),
        '2': dict(log_total_consumption=log_amount(deflate(c1['total_12_items_krw'], c1['reference_year'])),
                  consumption_pre_change=log_ratio(deflate(c1['total_12_items_krw'], c1['reference_year']),
                                                   deflate(c0['total_12_items_krw'], c0['reference_year'])),
                  **shares(c1['items'])),
        '3': dict(reemployment_days=reemp_days, ui_received=ui_received,
                  ui_duration_days=ui.get('duration_days') if ui_received else (0.0 if ui_received == 0 else None)),
        '4': dict(household_size=hh.get('reported_household_size'), children=counts['children'],
                  elderly=counts['elderly'], other_employed=other_employed,
                  log_deposit=log_amount(deflate(assets['bank_deposit']['amount_krw'], iy)),
                  log_stocks=log_amount(deflate(assets['stocks_bonds_trusts']['amount_krw'], iy))),
        '5': dict(person_age=person.get('age') if person else None),
    }
    cat = {
        '3': dict(reemployment_status=path.get('status')),
        '4': dict(housing_tenure=code_of(hh.get('housing_tenure'))),
        '5': dict(person_occupation=occupation_major(person, 'occupation')),
    }
    return num, cat


FEATURES = {'1a': features_1a, '1b': features_1b}


# ---------------------------------------------------------------- index

class CaseIndex:
    def __init__(self, task, cases, families, config=None, deflator=None, allow_nominal=False):
        if task not in TASKS:
            raise ValueError('task must be 1a or 1b')
        self.task, self.config = task, config or RetrievalConfig()
        self.deflator = deflator or Deflator()
        if task == '1b' and self.deflator.nominal and not allow_nominal:
            # 1-B corpus spans 2005-2023; nominal level distances mix price levels.
            raise ValueError('1-B retrieval needs a CPI series (or explicit allow_nominal for a non-final check)')
        if self.config.k < 1 or self.config.max_per_family < 1:
            raise ValueError('k and max_per_family must be positive')
        self.cases = list(cases)
        ids = [c['case_id'] for c in self.cases]
        if len(set(ids)) != len(ids):
            raise ValueError('duplicate case_id in corpus')
        for c in self.cases:
            if c['model_input'].get('task') != TASKS[task]['input_task']:
                raise ValueError('corpus case has a different task schema')
            if 'observed_output' not in c:
                raise ValueError('RAG corpus case lacks observed_output')
        missing = [i for i in ids if i not in families]
        if missing:
            raise ValueError(f'{len(missing)} corpus cases lack a family link')
        self.case_family = [families[i] for i in ids]
        self.families = set(self.case_family)
        rows = [self._raw(c['model_input']) for c in self.cases]
        self.num_names = sorted({(g, n) for num, _ in rows for g, d in num.items() for n in d})
        self.cat_names = sorted({(g, n) for _, cat in rows for g, d in cat.items() for n in d})
        X = np.array([[self._get(num, g, n) for g, n in self.num_names] for num, _ in rows], dtype=float)
        # Robust scaling fitted on the corpus only; an all-missing column stays missing.
        self.center, self.scale = np.zeros(X.shape[1]), np.ones(X.shape[1])
        for j in range(X.shape[1]):
            col = X[~np.isnan(X[:, j]), j]
            if col.size:
                q75, q25 = np.percentile(col, [75, 25])
                spread = (q75 - q25) / 1.349
                sd = col.std()
                self.center[j] = np.median(col)
                self.scale[j] = spread if spread > 0 else sd if sd > 0 else 1.0
        self.Z = (X - self.center) / self.scale
        self.C = [np.array([cat.get(g, {}).get(n) for _, cat in rows], dtype=object) for g, n in self.cat_names]
        self.C_present = [np.array([v is not None for v in col]) for col in self.C]
        self.num_w = self._weights(self.num_names, self.cat_names, numeric=True)
        self.cat_w = self._weights(self.num_names, self.cat_names, numeric=False)

    def _raw(self, model_input):
        return FEATURES[self.task](model_input, self.deflator)

    @staticmethod
    def _get(d, g, n):
        v = d.get(g, {}).get(n)
        return np.nan if v is None else float(v)

    def _weights(self, num_names, cat_names, numeric):
        """Group weight is split equally over that group's features."""
        sizes = {}
        for g, _ in num_names + cat_names:
            sizes[g] = sizes.get(g, 0) + 1
        names = num_names if numeric else cat_names
        return np.array([float(self.config.group_weights[g]) / sizes[g] for g, _ in names])

    def describe(self):
        return dict(version=VERSION, task=self.task, corpus_cases=len(self.cases), corpus_families=len(self.families),
                    config=asdict(self.config), config_sha256=self.config.digest(), price=self.deflator.describe(),
                    numeric_features=[f'{g}:{n}' for g, n in self.num_names],
                    categorical_features=[f'{g}:{n}' for g, n in self.cat_names],
                    distance='sqrt(weighted mean of squared robust-z differences and 0/1 category mismatches over features observed in both)',
                    missing='feature skipped for that pair; no imputation',
                    tie_break='distance, then sha256(case_id)')

    def query(self, model_input, query_family=None, exclude_families=()):
        """Return top-k corpus hits with at most max_per_family per family."""
        if query_family is not None and query_family in self.families and query_family not in set(exclude_families):
            raise ValueError('query family appears in the retrieval corpus (leakage)')
        num, cat = self._raw(model_input)
        q = (np.array([self._get(num, g, n) for g, n in self.num_names]) - self.center) / self.scale
        both = ~np.isnan(self.Z) & ~np.isnan(q)
        diff = np.where(both, self.Z - q, 0.0)
        num_sum = (diff ** 2 * self.num_w).sum(axis=1)
        num_wt = (both * self.num_w).sum(axis=1)
        qc = [cat.get(g, {}).get(n) for g, n in self.cat_names]
        cat_sum = np.zeros(len(self.cases))
        cat_wt = np.zeros(len(self.cases))
        for j, v in enumerate(qc):
            if v is None:
                continue
            present = self.C_present[j]
            cat_wt += self.cat_w[j] * present
            cat_sum += self.cat_w[j] * (present & (self.C[j] != v))
        total_wt = num_wt + cat_wt
        dist = np.where(total_wt > 0, np.sqrt((num_sum + cat_sum) / np.where(total_wt > 0, total_wt, 1)), np.inf)
        order = sorted(range(len(self.cases)),
                       key=lambda i: (dist[i], hashlib.sha256(self.cases[i]['case_id'].encode()).hexdigest()))
        excluded = set(exclude_families)
        used, hits = {}, []
        for i in order:
            fam = self.case_family[i]
            if fam in excluded or used.get(fam, 0) >= self.config.max_per_family or not math.isfinite(dist[i]):
                continue
            used[fam] = used.get(fam, 0) + 1
            hits.append(dict(case_id=self.cases[i]['case_id'], distance=float(dist[i]),
                             compared_weight=float(total_wt[i]), case=self.cases[i]))
            if len(hits) == self.config.k:
                break
        if len(hits) < self.config.k:
            raise ValueError('not enough eligible corpus cases for k')
        return hits


def load_task(task, cases_dir=None):
    """Read one task's corpus and family links from the local (git-ignored) case build."""
    base = Path(cases_dir) if cases_dir else ROOT / 'outputs' / TASKS[task]['cases']
    corpus_path = base / TASKS[task]['corpus']
    linkage_path = base / 'private' / 'linkage_and_quality.jsonl'
    corpus = read_jsonl(corpus_path)
    linkage = read_jsonl(linkage_path)
    families = {r['case_id']: r['family'] for r in linkage}
    splits = {r['case_id']: r['split'] for r in linkage}
    if any(splits.get(c['case_id']) != 'training' for c in corpus):
        raise ValueError('RAG corpus contains a non-training case')
    sources = {str(p): sha_file(p) for p in (corpus_path, linkage_path)}
    return dict(base=base, corpus=corpus, families=families, splits=splits, sources=sources)
