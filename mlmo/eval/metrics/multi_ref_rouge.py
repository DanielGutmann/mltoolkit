from mlmo.eval.metrics import BaseMetric
from mlutils.helpers.general import flatten
from collections import OrderedDict
from rouge import Rouge as _Rouge

# TODO: make it work based on pure strings instead of list of strings.
# TODO: use an internal sentence splitter

class MultiRefRouge(BaseMetric):
    """
    Accumulates Rouge-1, Rouge-2, and Rouge-L for each (hypothesis, references)
    pair. Upon aggregation computes micros over each metrics sub-score(R, P, F1). 
    """

    def __init__(self):
        super(MultiRefRouge, self).__init__()
        self._rouge = _RougeMod()
        self.summed_metr_scores = {}  # summed scores over instances
        self.inst_total_nr = 0  # the total number of instances

    def accum(self, hypothesis, references):
        """
        Inputs one pair (hypothesis, references) at a time, where references
        are multiple true summaries, and hypothesis is a summary generated
        by an algorithm.

        :param hypothesis: list of strings, where each string is a sentence.
        :param references: list of sub-lists, where each sub-list is a reference 
                           summary that contains string sentences.
        :return: dict of average over hypotheses scores.
        """
        references = flatten(references)
        curr_scores = self._rouge._get_scores(hypothesis, references)

        # storing scores to the internal collector
        for sname, svals in curr_scores.items():
            if sname not in self.summed_metr_scores:
                self.summed_metr_scores[sname] = {k: 0. for k in svals}
            for k, v in svals.items():
                self.summed_metr_scores[sname][k] += v

        self.inst_total_nr += 1

        return curr_scores

    def aggr(self):
        """Computes macro-scores based on accumulated micro-scores."""
        res = OrderedDict()
        for metr_name, metr_scores in self.summed_metr_scores.items():
            if metr_name not in res:
                res[metr_name] = {}
            for k, val in metr_scores.items():
                res[metr_name][k] = val/float(self.inst_total_nr)
        return res

class _RougeMod(_Rouge):
    """
    Modification version of the module's class to avoid internal sentences
    splitting, as it's inaccurate and leads to errors and problems.
    """
    
    def _get_scores(self, hyps, refs):
        """
        :param hyps: list of lists of strings.
        :param refs: list of lists of strings.
        """
        score = {}
        for m in self.metrics:
            fn = _Rouge.AVAILABLE_METRICS[m]
            sc = fn(hyps, refs)
            score[m] = {s: sc[s] for s in self.stats}
        return score
