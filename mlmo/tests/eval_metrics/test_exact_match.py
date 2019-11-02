from mlmo.eval.metrics import ExactMatches
import unittest


class TestExactMatch(unittest.TestCase):

    def test_case1(self):
        exact_matches = ExactMatches(begin_tag="b", inside_tag="i")
        true_seq = ["o", "o", "b", "i", "o", "b", "i", "b"]
        pred_seq = ["o", "o", "b", "o", "o", "b", "i", "o"]
        nr_correct, nr_pred, nr_total = exact_matches.accum(true_seq,
                                                            pred_seq)
        self.assertEqual([nr_correct, nr_pred, nr_total], [1, 2, 3])

    def test_case2(self):
        exact_matches = ExactMatches(begin_tag="b", inside_tag="i")
        true_seq = ["o", "b", "o", "b", "b"]
        pred_seq = ["b", "b", "b", "b", "i"]
        nr_correct, nr_pred, nr_total = exact_matches.accum(true_seq,
                                                            pred_seq, )
        self.assertEqual([nr_correct, nr_pred, nr_total], [1, 4, 3])

    def test_case3(self):
        exact_matches = ExactMatches(begin_tag="b", inside_tag="i")
        true_seq = ["b", "b"]
        pred_seq = ["b", "i"]
        nr_correct, nr_pred, nr_total = exact_matches.accum(true_seq,
                                                            pred_seq)
        self.assertEqual([nr_correct, nr_pred, nr_total], [0, 1, 2])

    def test_case4(self):
        exact_matches = ExactMatches(begin_tag="b", inside_tag="i")
        true_seq = ["b", "o"]
        pred_seq = ["b", "b"]
        nr_correct, nr_pred, nr_total = exact_matches.accum(true_seq,
                                                            pred_seq)
        self.assertEqual([nr_correct, nr_pred, nr_total], [1, 2, 1])

    def test_case5(self):
        exact_matches = ExactMatches(begin_tag="b", inside_tag="i")
        true_seq = ["b", "o"]
        pred_seq = ["b", "i"]
        nr_correct, nr_pred, nr_total = exact_matches.accum(true_seq,
                                                            pred_seq)
        self.assertEqual([nr_correct, nr_pred, nr_total], [0, 1, 1])

    def test_case6(self):
        exact_matches = ExactMatches(begin_tag="b", inside_tag="i")
        true_seq = ["b", "A", "B", "C", "i", "b"]
        pred_seq = ["b", "i", "o", "o", "o", "b"]
        nr_correct, nr_pred, nr_total = exact_matches.accum(true_seq,
                                                            pred_seq)
        self.assertEqual([nr_correct, nr_pred, nr_total], [1, 2, 2])


if __name__ == '__main__':
    unittest.main()
