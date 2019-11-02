from mlmo.eval.metrics import BaseMetric
from mlutils.helpers.general import sort_hash
from collections import OrderedDict


class Accuracy(BaseMetric):
    def __init__(self, excluded_labels=None):
        super(Accuracy, self).__init__()
        self.excluded_labels = excluded_labels if excluded_labels else []
        self.nr_correct = 0
        self._true_label_counts = {}
        self._pred_label_counts = {}
        self.nr_total = 0

    def accum(self, predicted_labels, true_labels):
        """
        Evaluates a chunk of predicted and true labels, and updates the
        necessary statistics.

        :param predicted_labels: list or array of predictions.
        :param true_labels: list or array of true labels.
        :return boolean list of matching labels. If a true labels was excluded,
                it will be marked as None.

        """
        assert len(predicted_labels) == len(true_labels)
        matches = []
        for pr_l, tr_l in zip(predicted_labels, true_labels):
            if tr_l in self.excluded_labels:
                matches.append(None)
                continue
            match = pr_l == tr_l
            self.nr_correct += match
            self.nr_total += 1
            matches.append(match)
            # store counts
            if tr_l not in self._true_label_counts:
                self._true_label_counts[tr_l] = 0
            if pr_l not in self._pred_label_counts:
                self._pred_label_counts[pr_l] = 0
            self._true_label_counts[tr_l] += 1
            self._pred_label_counts[pr_l] += 1
        return matches

    def aggr(self):
        res = OrderedDict()
        accuracy = float(self.nr_correct)/float(self.nr_total)
        res["accuracy"] = accuracy
        return res

    def calculate_label_freq(self, print_friendly=False):
        """Computes frequency for true and predicted labels."""
        res = {}
        for freq_name, counts_hash in zip(["true label freq.",
                                           "pred. label freq."],
                                          [self._true_label_counts,
                                           self._pred_label_counts]):
            if counts_hash:
                counts_hash = sort_hash(counts_hash, by_key=False)
                norm_counts_hash = {k: float(v) / self.nr_total for k, v
                                    in counts_hash.items()}
                if print_friendly:
                    form = ["%s: %f" % (k, v) for k, v in norm_counts_hash.items()]
                    res[freq_name] = " ".join(form)
                else:
                    res[freq_name] = norm_counts_hash
        return res
