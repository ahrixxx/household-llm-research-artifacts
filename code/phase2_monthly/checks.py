"""Observable consistency checks; no LLM-as-judge or latent reasoning claims."""
from .store import canonical

FACT_KINDS = {'profile', 'state', 'event', 'execution'}


def pointer(value, path):
    if not isinstance(path, str) or not path.startswith('/'):
        raise ValueError('JSON pointer required')
    for part in path[1:].split('/'):
        part = part.replace('~1', '/').replace('~0', '~')
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


def equal(left, right):
    return canonical(left) == canonical(right)


def check_claims(claims, sources):
    source_map = {record['id']: record for record in sources}
    rows = []
    for claim in claims:
        record = source_map.get(claim['source_id'])
        status = 'MATCH'
        observed = None
        if record is None:
            status = 'SOURCE_NOT_AVAILABLE'
        elif record['kind'] not in FACT_KINDS:
            status = 'INTERPRETATION_OR_PLAN_IS_NOT_FACT'
        else:
            try:
                observed = pointer(record['data'], claim['path'])
                if not equal(claim['value'], observed):
                    status = 'FACT_MISMATCH'
            except (KeyError, ValueError, IndexError, TypeError):
                status = 'PATH_NOT_FOUND'
        rows.append(dict(claim=claim, status=status, observed=observed))
    return dict(claim_count=len(rows), mismatch_count=sum(r['status'] != 'MATCH' for r in rows),
                coverage='CHECKED_CLAIMS_ONLY' if rows else 'NO_FACT_CLAIMS', details=rows,
                prose_status='REQUIRES_SEMANTIC_REVIEW')


def leaf_claims(value, prefix=''):
    rows = []
    for key, item in value.items():
        path = prefix + '/' + key
        if isinstance(item, dict):
            rows.extend(leaf_claims(item, path))
        else:
            rows.append({'path': path, 'value': item})
    return rows


def compare_plan(plan, previous, month):
    expected = previous['id'] if previous else None
    old_target = previous['data']['schedule'].get(str(month)) if previous else None
    changed = old_target is not None and not equal(old_target, plan['schedule'][str(month)])
    return dict(previous_plan_link_matches=plan['supersedes'] == expected,
                comparable_previous_target=old_target is not None,
                current_target_changed=changed, reason_recorded=bool(plan['reason'].strip()),
                supporting_claim_count=len(plan['claims']),
                reason_adequacy='REQUIRES_SEMANTIC_REVIEW')


def check_decision(decision, plan, sources, month, executed=None):
    target = plan['data']['schedule'][str(month)]
    deviates = not equal(target, decision['action'])
    action_rows = []
    expected_paths = {row['path'] for row in leaf_claims(decision['action'])}
    seen = set()
    for claim in decision['action_claims']:
        row = dict(claim=claim, matches_choice=False, matches_execution=None)
        try:
            row['matches_choice'] = equal(pointer(decision['action'], claim['path']), claim['value'])
            if executed is not None:
                row['matches_execution'] = equal(pointer(executed, claim['path']), claim['value'])
        except (KeyError, ValueError, IndexError, TypeError):
            pass
        row['duplicate'] = claim['path'] in seen
        seen.add(claim['path'])
        action_rows.append(row)
    return dict(factual_memory=check_claims(decision['explanation']['claims'], sources),
                plan_id_matches=decision['plan_id'] == plan['id'], deviates_from_plan=deviates,
                deviation_documented=not deviates or bool(decision['deviation_reason']),
                deviation_reason=decision['deviation_reason'],
                explanation_action=dict(details=action_rows, missing_paths=sorted(expected_paths - seen),
                                        invalid_or_duplicate_count=sum(not r['matches_choice'] or r['duplicate'] for r in action_rows),
                                        prose_status='REQUIRES_SEMANTIC_REVIEW'),
                choice_equals_execution=None if executed is None else equal(decision['action'], executed))


def retrieve(records, month, tags, window, limit):
    """Deterministic tag overlap, then recency. Only the caller's own namespace."""
    candidates = [r for r in records if r['month'] < month and r['available_month'] <= month
                  and month - r['month'] <= window]
    def rank(record):
        overlap = len(set(record['data'].get('tags', [])) & set(tags))
        return overlap, record['month'], record['id']
    return sorted(candidates, key=rank, reverse=True)[:limit]


def environment_truth(observed):
    """Truth from information already provided, not an unpublished future path."""
    expected = dict(income_krw=observed['income_krw'], cash_transfer_krw=observed['cash_transfer_krw'],
                    cbdc_grant_krw=sum(lot['amount_krw'] for lot in observed['cbdc_grants']),
                    etf_price_krw=observed['etf_price_krw'], last_ui_paid_month=None,
                    first_ui_unpaid_month=None, current_ui_krw=None)
    for event in observed['events']:
        for source, target in (('last_paid_month', 'last_ui_paid_month'),
                               ('first_unpaid_month', 'first_ui_unpaid_month'),
                               ('current_ui_krw', 'current_ui_krw')):
            if source in event['facts']:
                value = event['facts'][source]
                if expected[target] is not None and not equal(expected[target], value):
                    raise ValueError('conflicting announced UI facts')
                expected[target] = value
    return expected


def check_understanding(answer, observed):
    expected = environment_truth(observed)
    rows = {key: dict(answer=answer[key], expected=value, matches=equal(answer[key], value))
            for key, value in expected.items()}
    return dict(details=rows, assessed_fields=len(rows),
                mismatches=sum(not r['matches'] for r in rows.values()),
                behavioral_response='SEPARATE_PATH_COMPARISON_REQUIRED')
