"""Economic memory views; original research records remain immutable.

An episode is one completed month, linking its environment, resources, realized
transactions, closing state and committed numeric plan. Retrieval is local and
deterministic, with no model calls or generated summaries.
"""
from .store import clone, digest


def economic_view(record):
    """Never pass correction prose/links through plan or execution records."""
    if record is None:
        return None
    kind, data = record['kind'], record['data']
    if kind == 'plan':
        data = dict(schedule=clone(data['schedule']))
    elif kind == 'execution':
        if not data['valid']:
            return None
        fields = ('action', 'calculation', 'closing_state', 'cbdc_redeemed',
                  'cbdc_expired', 'etf_value_krw')
        data = {k: clone(data[k]) for k in fields}
    elif kind == 'event':
        if data.get('type') != 'economic_environment':
            return None
    elif kind not in ('profile', 'state', 'interpretation'):
        return None
    view = clone(record)
    view.update(data=clone(data), data_hash=digest(data), view='economic_memory_v2',
                source_data_hash=record['data_hash'])
    return view


def month_episode(store, month):
    """Also usable after execution but before quarter reflection completes."""
    execution = store.get('execution', f'm{month:03d}')
    if execution is None:
        return []
    plan_kind, plan_key = execution['data']['plan_id'].split('/')
    records = [store.get('event', f'm{month:03d}_environment'),
               store.get('state', f'm{month:03d}_open'), execution,
               store.get('state', f'm{month:03d}_close'), store.get(plan_kind, plan_key)]
    return [view for record in records if (view := economic_view(record)) is not None]


def previous_plan(store, month):
    execution = store.get('execution', f'm{month-1:03d}') if month > 1 else None
    if execution is None:
        return None
    kind, key = execution['data']['plan_id'].split('/')
    return economic_view(store.get(kind, key))


def recall(store, month, observed, limit):
    """Always last month/latest reflection; up to limit older relevant months.

Positive event-tag overlap is required. A source month already supplied as
mandatory material or reflection provenance is not selected again as an extra.
"""
    sources = {}
    def add(record):
        if record and record['month'] < month and record['available_month'] <= month:
            sources[record['id']] = record

    if month > 1:
        for record in month_episode(store, month - 1):
            add(record)
    reflections = [r for r in store.list('interpretation')
                   if r['month'] < month and r['available_month'] <= month]
    if reflections:
        latest = max(reflections, key=lambda r: (r['month'], r['id']))
        add(economic_view(latest))
        for claim in latest['data']['claims']:
            parts = claim['source_id'].split('/')
            if len(parts) != 2 or parts[0] not in ('profile', 'state', 'event', 'execution'):
                continue
            try:
                record = store.get(*parts)
            except ValueError:
                continue
            add(economic_view(record))
    mandatory_ids = list(sources)
    represented_months = {r['month'] for r in sources.values()}
    tags = {tag for event in observed['events'] for tag in event['tags']}
    candidates = []
    for record in store.list('event'):
        if record['data'].get('type') != 'economic_environment':
            continue
        m = record['month']
        if m >= month or record['available_month'] > month or m in represented_months:
            continue
        if store.get('event', f'm{m:03d}_complete') is None:
            continue
        overlap = len(tags & set(record['data']['tags']))
        if overlap:
            candidates.append((overlap, m))
    selected = sorted(candidates, reverse=True)[:limit]
    for _, m in selected:
        for record in month_episode(store, m):
            add(record)
    audit = dict(algorithm='completed_month_episodes_tag_overlap_then_recency_v2',
                 history_scope='all_available_past', additional_episode_limit=limit,
                 mandatory_source_ids=mandatory_ids,
                 additional_episodes=[dict(month=m, tag_overlap=score) for score, m in selected],
                 source_ids=list(sources), correction_history_included=False,
                 model_calls=0)
    return list(sources.values()), audit
