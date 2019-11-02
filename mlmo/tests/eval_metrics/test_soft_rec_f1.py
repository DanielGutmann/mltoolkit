import unittest
from mlmo.eval.metrics import SoftPrRecF1
from numpy import isclose


class TestSoftRecF1(unittest.TestCase):
    
    def test_accum_output(self):
        hyps = ["two three 2 3", "four five six", "seven ten", 'dummy']
        refs = ["one two three", "four five six", "seven eight ten"]

        exp_res = {'rouge-1': {'soft_r': 0.79047619, 'soft_p': 0.5928571425},
                   'rouge-2': {'soft_r': 0.46666666, 'soft_p': 0.349999995}}

        eval_metr = SoftPrRecF1()
        act_res = eval_metr.accum(hyps, refs)

        for metr_name in exp_res.keys():
            exp_metr_scores = exp_res[metr_name]
            act_metr_scores = act_res[metr_name]

            for score_name in exp_metr_scores:
                exp_score_val = exp_metr_scores[score_name]
                act_score_val = act_metr_scores[score_name]
                self.assertTrue(isclose(act_score_val, exp_score_val))


if __name__ == '__main__':
    unittest.main()
