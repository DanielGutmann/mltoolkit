from mlmo.eval.metrics import BaseMetric
from collections import OrderedDict
from rouge import Rouge as _Rouge
import numpy as np


class RougeMulti(BaseMetric):
    """
    Rouge wrapper expecting sets of references and hypotheses to be provided.
    And for each hypothesis arg-max scores (based on recall, precision, or f1)
    over each reference is computed. In other words, it tries to find the best
    match for each hypothesis and save its score.

    Computes Rouge-1, Rouge-2, and Rouge-L, for each hypothesis.
    """

    def __init__(self, max_key='f'):
        """
        :param max_key: a key with respect to which a maximum hypothesis score
                        is computed. Allowed values: 'r', 'p', 'f'.
        """
        super(RougeMulti, self).__init__()
        self._rouge = _RougeMod()
        self.summed_metr_scores = {}  # summed scores over instances
        self.inst_total_nr = 0  # the total number of instances
        self.max_key = max_key

    def accum(self, hypotheses, references):
        """
        Computes different Rouge scores, where each is an average over
        hypotheses scores. Each hypothesis score is computed as maximum rouge
        score over each reference.

        Inputs one instance at a time, which can be composed of multiple
        hypotheses and references.

        :param hypotheses: hypotheses (actual) summaries list where each one is
                           a string (single sentence) or a list of strings
                           (multi sentence). If a string is passed a single
                           sentence summary is assumed.
        :type hypotheses: list of lists of strings or a string (one sentence).
        :param references: reference (true) summaries list where each one is a
                           string (single sentence) or a list of strings
                           (multi sentence). If a string is passed a single
                           sentence summary is assumed.
        :type references: list of lists of strings or a string (one sentence).
        :return: dict of average over hypotheses scores.
        """
        hypotheses = _format_summs(hypotheses)
        references = _format_summs(references)

        avg_scores = OrderedDict()
        for h in hypotheses:
            hyp_max_scores = {}
            for r in references:
                curr_scores = self._rouge._get_scores(h, r)
                for sname, svals in curr_scores.items():
                    if sname not in hyp_max_scores:
                        hyp_max_scores[sname] = svals
                    else:
                        if hyp_max_scores[sname][self.max_key] < svals[self.max_key]:
                            hyp_max_scores[sname] = svals

            # averaging each metric score over the number of hypotheses
            for sname, svals in hyp_max_scores.items():
                if sname not in avg_scores:
                    avg_scores[sname] = OrderedDict()
                for k, v in svals.items():
                    if k not in avg_scores[sname]:
                        avg_scores[sname][k] = 0.
                    avg_scores[sname][k] += v/float(len(hypotheses))

        # storing scores to the internal collector
        for sname, svals in avg_scores.items():
            if sname not in self.summed_metr_scores:
                self.summed_metr_scores[sname] = {k: 0. for k in svals}
            for k, v in svals.items():
                self.summed_metr_scores[sname][k] += v

        self.inst_total_nr += 1

        return avg_scores

    def aggr(self):
        """Computes the micro-scores avg. over the number of instances."""
        res = OrderedDict()
        for metr_name, metr_scores in self.summed_metr_scores.items():
            if metr_name not in res:
                res[metr_name] = {}
            for k, val in metr_scores.items():
                res[metr_name][k] = val/float(self.inst_total_nr)
        return res


def _format_summs(summs):
    # 1. single sentence summary
    if isinstance(summs, str):
        return [[summs]]
    # 2. multiple summaries
    if isinstance(summs, (list, np.ndarray, tuple)):
        formatted = []
        for summ in summs:
            # 2.1. single sentence summary
            if isinstance(summ, str):
                formatted.append([summ])
            # 2.2 multi sentence summary
            else:
                formatted.append(summ)
        return formatted
            

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
