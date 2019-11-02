import unittest
from mlmo.eval.metrics import PrRecF1
from collections import OrderedDict
from numpy import isclose


class TestPrRecF1(unittest.TestCase):

    def test_full_results(self):

        pred_labels = [1, 1, 2, 2, 100, 105, 3]
        true_labels = [1, 1, 1, 2, 3, 100, 100]

        exp_res = {1: {'p': 1., 'r': 2./3, 'f1': 0.8},
                   2: {'p': 0.5, 'r': 1., 'f1': 2./3},
                   3: {'p': 0., 'r': 0., 'f1': 0.},
                   100: {'p': 0., 'r': 0., 'f1': 0.},
                   105: {'p': 0., 'r': 0., 'f1': 0.}}

        metr = PrRecF1()
        metr.accum(pred_labels, true_labels)
        res = metr.aggr()

        self.assertTrue(equal_dicts(res, exp_res))

    def test_macro_results(self):
        pred_labels = [1, 1, 2, 2, 100, 105, 3]
        true_labels = [1, 1, 1, 2, 3, 100, 100]

        exp_res = {"macro_p": 0.3, "macro_r": (1. + 2./3)/5.,
                   'macro_f1': (0.8 + 2./3.)/5.}

        metr = PrRecF1()
        metr.accum(pred_labels, true_labels)
        res = metr.aggr(macro=True)

        self.assertTrue(equal_dicts(res, exp_res))


def equal_dicts(dict1, dict2):
    if len(dict1.keys()) != len(dict2.keys()):
        return False
    for k in dict1:
        if k not in dict2:
            return False
        v1 = dict1[k]
        v2 = dict2[k]
        if isinstance(v1, (OrderedDict, dict)) and \
            isinstance(v2, (OrderedDict, dict)):
            return equal_dicts(v1, v2)
        if type(v1) == type(v2):
            return isclose(v1, v2)
        else:
            return False
    return True


if __name__ == '__main__':
    unittest.main()
