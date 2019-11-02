from mlmo.eval.metrics import BaseMetric
from mlmo.utils.helpers.metrics import comp_recall_precision_f1
from collections import OrderedDict


# TODO: think if it makes sense to pass begin_tag and inside_tag to accum
# TODO: the method.
class ExactMatches(BaseMetric):
    """
    Computes the number of exact token matches between true and predicted
    label/tag sequences. Assumes IOB sequences.
    """

    def __init__(self, begin_tag='b', inside_tag='i', **kwargs):
        """
        It assumes that all sequences will have static IOB tags, based on
        begin_tag and inside_tag provided.
        """
        super(ExactMatches, self).__init__(**kwargs)
        self.nr_correct = 0.
        self.nr_total = 0.
        self.nr_predicted = 0.
        self.begin_tag = begin_tag
        self.inside_tag = inside_tag

    def aggr(self):
        r_exact, p_exact, f1_exact = comp_recall_precision_f1(self.nr_correct,
                                                              self.nr_predicted,
                                                              self.nr_total)
        return OrderedDict([(self._combine_with_prefix("r_exact"), r_exact),
                            (self._combine_with_prefix("p_exact"), p_exact),
                            (self._combine_with_prefix("f1_exact"), f1_exact)])

    def accum(self, true_seq, pred_seq):
        assert len(true_seq) == len(pred_seq)
        nr_correct = 0
        nr_total = 0
        nr_pred = 0

        correct_entity = False

        for true_tag, pred_tag in zip(true_seq, pred_seq):
            # if true_tag not in allowed_tags:
            #     continue
            if true_tag == self.begin_tag:

                # the reason for the below condition is to check whether the
                # predicted entity is also ended
                if pred_tag != self.inside_tag:
                    # stores the number of correct only when scanning of the
                    # current entity is finished
                    nr_correct += correct_entity
                nr_total += 1
                correct_entity = True

            if pred_tag != true_tag:
                # extra condition that says that if the current true entity is
                # not yet known to end, and the current predicted entity has
                # already ended - do not immediately say that's not match.
                if not (true_tag not in [self.begin_tag, self.inside_tag]
                        and pred_tag == self.begin_tag):
                    correct_entity = False

            if pred_tag == self.begin_tag:
                nr_pred += 1

        # end of the sequence
        nr_correct += correct_entity

        # update statistics
        self.nr_correct += nr_correct
        self.nr_total += nr_total
        self.nr_predicted += nr_pred

        return nr_correct, nr_pred, nr_total
