"""Unfrozen, input-only 1-B calendar-time retrieval candidate; original index intact."""
from phase1_rag_v1 import CaseIndex, RetrievalConfig
from event_timing_v1 import retrieval_features, VERSION as TIMING_VERSION

VERSION = 'phase1_rag_timing_v2_20261015'


class TimingCaseIndex(CaseIndex):
    def __init__(self, task, cases, families, config=None, deflator=None):
        if task != '1b':
            raise ValueError('Timing retrieval applies only to 1-B')
        if config is None:
            config = RetrievalConfig(group_weights={'1': 5., '2': 4., '3': 3., '4': 2., '5': 1., '6': 3.},
                                     status='DEVELOPMENT_CANDIDATE_NOT_OUTCOME_TUNED')
        if set(config.group_weights) != set('123456') or config.group_weights['6'] <= 0:
            raise ValueError('Require six groups including positive calendar timing weight')
        super().__init__(task, cases, families, config=config, deflator=deflator)

    def _raw(self, model_input):
        num, cat = super()._raw(model_input)
        num['6'] = retrieval_features(model_input)
        return num, cat

    def describe(self):
        return dict(super().describe(), version=VERSION, timing_version=TIMING_VERSION,
                    timing_weight_rationale='3, equal to existing employment group; not selected on development outcomes',
                    timing_missing='Unknown gap stays missing; never imputed as zero or continuous unemployment')
