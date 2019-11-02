import itertools
import unittest

from numpy.random import permutation

from mlmo.eval.metrics import RougeMulti
from mlmo.utils.helpers.metrics import comp_f1
from .helpers import get_fixture_refs_and_hyps, generate_sents


class TestRougeMulti(unittest.TestCase):

    def test_same_ref_as_hyp_single_instance(self):
        """Scores should be maximum when references and hypothesis are same."""
        sents_max_count = 6
        exp_scores = {"r": 1., "p": 1., "f": 1.}

        for sents_num in range(1, sents_max_count):
            refs = generate_sents(sents_num)
            for hyps in itertools.permutations(refs):
                if sents_num == 1:
                    hyps = hyps[0]
                    refs = refs[0]

                rouge_multi = RougeMulti()
                try:
                    act_metr_scores = rouge_multi.accum(hypotheses=hyps,
                                                        references=refs)
                except:
                    pass

                for act_scores in act_metr_scores.values():
                    for k in act_scores:
                        self.assertTrue(k in exp_scores)
                        self.assertAlmostEqual(act_scores[k], exp_scores[k],
                                               places=7)

    def test_same_ref_as_hyp_multiple_instance(self):
        """test_same_ref_as_hyp_single_instance but with multiple instances."""
        inst_max_count = 10
        sents_max_count = 6
        exp_scores = {"r": 1., "p": 1., "f": 1.}

        for inst_num in range(1, inst_max_count):
            for sents_num in range(1, sents_max_count):
                all_refs = [generate_sents(sents_num) for _ in range(inst_num)]
                all_hyps = [list(permutation(refs)) for refs in all_refs]

                rouge_multi = RougeMulti()
                for hyps, refs in zip(all_hyps, all_refs):
                    rouge_multi.accum(hypotheses=hyps, references=refs)

                act_metr_scores = rouge_multi.aggr()
                for act_scores in act_metr_scores.values():
                    for k in act_scores:
                        self.assertTrue(k in exp_scores)
                        act_score = act_scores[k]
                        exp_score = exp_scores[k]
                        self.assertAlmostEqual(act_score, exp_score, places=7)

    def test_one_hyp_vs_all_refs(self):
        hyps, refs = get_fixture_refs_and_hyps()

        hyps_exp_metr_scores = [{"rouge-1":
                                     {'r': 2. / 3, 'p': 2. / 4,
                                      'f': comp_f1(2. / 4, 2. / 3)},
                                 "rouge-2":
                                     {'r': 1. / 2, 'p': 1. / 3,
                                      'f': comp_f1(1. / 2, 1. / 3)}
                                 },
                                {'rouge-1': {"r": 1., "p": 1., "f": 1.},
                                 'rouge-2': {"r": 1., "p": 1., "f": 1.}
                                 },
                                {'rouge-1': {'r': 2. / 3, 'p': 1.,
                                             'f': comp_f1(1., 2. / 3)},
                                 'rouge-2': {'r': 0., 'p': 0., 'f': 0.}}
                                ]

        rouge_multi = RougeMulti()

        for hyp, exp_metr_scores in zip(hyps, hyps_exp_metr_scores):
            act_metr_scores = rouge_multi.accum(hypotheses=hyp,
                                                references=refs)

            for metr_name in exp_metr_scores:
                if metr_name in act_metr_scores:
                    for score_name in act_metr_scores[metr_name]:
                        act_score = act_metr_scores[metr_name][score_name]
                        exp_score = exp_metr_scores[metr_name][score_name]
                        self.assertAlmostEqual(act_score, exp_score, places=7)

    def test_all_hyp_vs_all_refs(self):
        hyps, refs = get_fixture_refs_and_hyps()

        r1_r = (2. / 3 + 1. + 2./3)/3
        r1_p = (2. / 4 + 1. + 1.)/3
        r1_f = (comp_f1(2. / 4, 2. / 3) + 1. + comp_f1(1., 2. / 3))/3

        r2_r = (1. / 2 + 1. + 0.)/3
        r2_p = (1. / 3 + 1. + 0.)/3
        r2_f = (comp_f1(1. / 2, 1. / 3) + 1. + 0.)/3
        exp_metr_scores = {"rouge-1": {'r': r1_r, 'p': r1_p, 'f': r1_f},
                           "rouge-2": {'r': r2_r, 'p': r2_p, 'f': r2_f},
                           }

        rouge_multi = RougeMulti()

        act_metr_scores = rouge_multi.accum(hypotheses=hyps,
                                            references=refs)

        for metr_name in exp_metr_scores:
            if metr_name in act_metr_scores:
                for score_name in act_metr_scores[metr_name]:
                    act_score = act_metr_scores[metr_name][score_name]
                    exp_score = exp_metr_scores[metr_name][score_name]
                    self.assertAlmostEqual(act_score, exp_score, places=7)


if __name__ == '__main__':
    unittest.main()
