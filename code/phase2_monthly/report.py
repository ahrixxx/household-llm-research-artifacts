"""Transparent denominators; schema/transport failures never count as feasible."""


def summarize(store):
    first_requests = [r for r in store.list('request') if r['id'].endswith('_a000')]
    first_audits = [r for r in store.list('audit') if r['id'].endswith('_a000')]
    first_bad = [r for r in first_audits if not r['data']['feasibility']['valid']]
    completed = [r for r in store.list('event') if r['id'].endswith('_complete')]
    decisions = [r for r in store.list('audit') if 'consistency' in r['data']]
    factual_claims = mismatches = deviations = undocumented = explanation_mismatches = 0
    unclaimed_action_fields = 0
    revisions = {}
    for attempt in (1, 2):
        rows = [r for r in decisions if r['id'].endswith(f'_a{attempt:03d}')]
        success = sum(r['data']['feasibility']['valid'] for r in rows)
        revisions[str(attempt)] = dict(assessed=len(rows), feasible=success,
                                      success_rate=success / len(rows) if rows else None)
    for audit in decisions:
        check = audit['data']['consistency']
        factual_claims += check['factual_memory']['claim_count']
        mismatches += check['factual_memory']['mismatch_count']
        deviations += check['deviates_from_plan']
        undocumented += not check['deviation_documented']
        explanation_mismatches += check['explanation_action']['invalid_or_duplicate_count']
        unclaimed_action_fields += len(check['explanation_action']['missing_paths'])
    return dict(months_started=len([r for r in store.list('state') if r['id'].endswith('_open')]),
                months_completed=len(completed),
                first_choice_requests=len(first_requests), first_choices_schema_valid=len(first_audits),
                first_choices_unassessed=len(first_requests) - len(first_audits),
                first_choices_infeasible=len(first_bad),
                first_violation_rate_among_assessed=len(first_bad) / len(first_audits) if first_audits else None,
                months_corrected=sum(not r['data']['first_choice_feasible'] for r in completed),
                corrections_by_attempt=revisions,
                paths_exhausted=len([r for r in store.list('failure') if r['id'].endswith('_exhausted')]),
                decision_attempts_assessed=len(decisions), factual_claim_occurrences=factual_claims,
                first_environment_fields=sum(r['data']['environment_understanding']['assessed_fields'] for r in first_audits),
                first_environment_mismatches=sum(r['data']['environment_understanding']['mismatches'] for r in first_audits),
                factual_mismatches=mismatches, plan_deviations=deviations,
                undocumented_plan_deviations=undocumented, explanation_action_mismatches=explanation_mismatches,
                missing_action_claim_fields=unclaimed_action_fields,
                semantic_review='NOT_AUTOMATICALLY_SCORED', empirical_plausibility='NOT_TESTED_BY_THIS_REPORT')
