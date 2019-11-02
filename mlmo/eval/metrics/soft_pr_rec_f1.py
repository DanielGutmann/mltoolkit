from mlmo.eval.metrics import BaseMetric, RougeMulti
from mlmo.utils.helpers.metrics import comp_f1
from collections import OrderedDict


class SoftPrRecF1(BaseMetric):
    """
    Computes a soft version of precision, recall, and f1 for sequences.
    Currently, it's uses RougeMulti to produce soft-matches/scores.
    TODO: provide a better documentation when the metric will be approved.
    """

    def __init__(self, key_soft_score='f'):
        """
        :param key_soft_score: what soft-score to use. Allowed: 'r', 'p', 'f'.
        """
        super(SoftPrRecF1, self).__init__()
        self.summed_metr_scores = {}  # summed scores over instances
        self.inst_total_nr = 0
        self._rouge_multi = RougeMulti(max_key=key_soft_score)
        self.key_soft_score = key_soft_score

    def accum(self, hypotheses, references):
        """
        :param hypotheses: hypotheses (actual) summaries where each one is a
                           string (single sentence) or a list of strings
                           (multi sentence). If a string is passed a single
                           sentence summary is assumed.
        :type hypotheses: list of lists of strings or a string (one sentence).
        :param references: reference (true) summaries where each one is a
                           string (single sentence) or a list of strings
                           (multi sentence). If a string is passed a single
                           sentence summary is assumed.
        :type references: list of lists of strings or a string (one sentence).
        """

        soft_prs = self._rouge_multi.accum(hypotheses=hypotheses,
                                           references=references)
        soft_recs = self._rouge_multi.accum(hypotheses=references,
                                            references=hypotheses)
        soft_prs = self._select_key_scores(soft_prs,
                                           key_score_name=self.key_soft_score)
        soft_recs = self._select_key_scores(soft_recs,
                                            key_score_name=self.key_soft_score)

        metr_names = soft_prs.keys()
        metr_soft_scores = {}
        for metr_name in metr_names:
            soft_pr = soft_prs[metr_name]
            soft_rec = soft_recs[metr_name]
            soft_f1 = comp_f1(soft_pr, soft_rec)

            metr_soft_scores[metr_name] = OrderedDict(soft_p=soft_pr,
                                                      soft_r=soft_rec,
                                                      soft_f1=soft_f1)

            if metr_name not in self.summed_metr_scores:
                self.summed_metr_scores[metr_name] = OrderedDict(soft_p=0.,
                                                                 soft_r=0.,
                                                                 soft_f1=0.)
            self.summed_metr_scores[metr_name]['soft_p'] += soft_pr
            self.summed_metr_scores[metr_name]['soft_r'] += soft_rec
            self.summed_metr_scores[metr_name]['soft_f1'] += soft_f1

        self.inst_total_nr += 1

        return metr_soft_scores

    def aggr(self):
        """Computes the soft metrics avg. over the number of instances."""
        res = OrderedDict()
        for metr_name, metr_scores in self.summed_metr_scores.items():
            if metr_name not in res:
                res[metr_name] = {}
            for k, val in metr_scores.items():
                res[metr_name][k] = val / float(self.inst_total_nr)
        return res

    def _select_key_scores(self, metrs, key_score_name):
        res = OrderedDict()
        for metr_name, scores in metrs.items():
            res[metr_name] = scores[key_score_name]
        return res
