from mlmo.eval.metrics import BaseMetric
from mlmo.utils.helpers.metrics import comp_recall_precision_f1
from collections import OrderedDict
import numpy as np


class PrRecF1(BaseMetric):
    """Iterative computation of multi-class precision, recall, and F1."""

    def __init__(self, excluded_labels=None,):
        super(PrRecF1, self).__init__()
        self.excluded_labels = excluded_labels if excluded_labels else []
        self._pred_to_true_hits = {}

    def accum(self, predicted_labels, true_labels):
        """
        :type predicted_labels: list or array.
        :type true_labels: list or array.
        """
        assert len(predicted_labels) == len(true_labels)
        for p_lbl, t_lbl in zip(predicted_labels, true_labels):
            self._accum(p_lbl, t_lbl)

    def aggr(self, macro=False):
        """
        computes precision, recall, comp_f1 for each class based on collection
        information in confusion matrix.
        
        :param macro: whether to compute average over metrics or return the full
                      version results.
        """
        res = OrderedDict()
        real_classes_number = 0
        total_recall = 0.
        total_precision = 0.
        total_f1 = 0.

        conf_matrix, lbl_to_indx = self._comp_conf_matrix()

        excluded_indxs = set([lbl_to_indx[lbl] for lbl in self.excluded_labels])

        for lbl, indx in lbl_to_indx.items():
            if indx in excluded_indxs:
                continue

            real_classes_number += 1
            tp = conf_matrix[indx, indx]
            fp = np.sum(conf_matrix[indx, :]) - tp
            fn = np.sum(conf_matrix[:, indx]) - tp

            recall, precision, f1 = comp_recall_precision_f1(nr_correct=tp,
                                                             nr_predicted=tp+fp,
                                                             nr_total=tp+fn)
            # store to the collector local statistics
            if macro:
                total_recall += recall
                total_precision += precision
                total_f1 += f1
            else:
                res[lbl] = OrderedDict()
                for nm, value in zip(['p', 'r', 'f1'], [precision, recall, f1]):
                    res[lbl][nm] = float(value)

        if macro:
            macro_rec = total_recall / real_classes_number
            macro_pr = total_precision / real_classes_number
            macro_f1 = total_f1 / real_classes_number
            res["macro_p"] = float(macro_pr)
            res["macro_r"] = float(macro_rec)
            res["macro_f1"] = float(macro_f1)

        return res

    def _accum(self, pred_lbl, true_lbl):
        if pred_lbl not in self._pred_to_true_hits:
            self._pred_to_true_hits[pred_lbl] = {}
        if true_lbl not in self._pred_to_true_hits[pred_lbl]:
            self._pred_to_true_hits[pred_lbl][true_lbl] = 0
        self._pred_to_true_hits[pred_lbl][true_lbl] += 1

    def _comp_conf_matrix(self):
        """
        Computes the confusion matrix that has float hits, and mapping from
        labels to indxs.
        """
        # compute the number of unique labels
        lbl_to_indx = {}
        for p_lbl in self._pred_to_true_hits.keys():
            if p_lbl not in lbl_to_indx:
                lbl_to_indx[p_lbl] = len(lbl_to_indx)
            for t_lbl in self._pred_to_true_hits[p_lbl].keys():
                if t_lbl not in lbl_to_indx:
                    lbl_to_indx[t_lbl] = len(lbl_to_indx)

        # initialize and fill the confusion matrix
        conf_matrix = np.zeros((len(lbl_to_indx), len(lbl_to_indx)),
                               dtype="float32")
        for p_lbl in self._pred_to_true_hits.keys():
            for t_lbl, hits in self._pred_to_true_hits[p_lbl].items():
                conf_matrix[lbl_to_indx[p_lbl], lbl_to_indx[t_lbl]] = hits

        return conf_matrix, lbl_to_indx
