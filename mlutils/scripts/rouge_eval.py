import argparse
from pandas import read_csv
from shared.eval.metrics import GoogleRouge
import os
from csv import QUOTE_NONE


def rouge_eval(gen_summ_fp, gen_summ_key_fn, gen_summ_val_fn, true_summ_fp,
               true_summ_key_fn, true_summ_val_fns):
    """
    Performs evaluation of summaries by reading generated and true ones
    from csv files.
    """
    gen_summs_ds = read_csv(gen_summ_fp, sep='\t', encoding='utf-8',
                            quoting=QUOTE_NONE) \
        .sort_values(gen_summ_key_fn)
    true_summs_ds = read_csv(true_summ_fp, sep='\t', encoding='utf-8',
                             quoting=QUOTE_NONE)\
        .sort_values(true_summ_key_fn)

    gen_summs_id = list(gen_summs_ds[gen_summ_key_fn])
    gen_summs = list(gen_summs_ds[gen_summ_val_fn])

    true_summs_id = list(true_summs_ds[true_summ_key_fn])

    true_summs = []
    for _, du in true_summs_ds.iterrows():
        true_summs.append([du[true_summ_fn] for true_summ_fn in true_summ_val_fns])

    assert gen_summs_id == true_summs_id

    rouge = GoogleRouge()
    rouge.accum(gen_summs, true_summs)
    
    print(rouge.aggr(avg=True))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-gen_summ_fp", type=str)
    parser.add_argument("-gen_summ_key_fn", type=str)
    parser.add_argument("-gen_summ_val_fn", type=str)
    parser.add_argument("-true_summ_fp", type=str)
    parser.add_argument("-true_summ_key_fn", type=str)
    parser.add_argument("-true_summ_val_fns", type=str, nargs="+")
    args = parser.parse_args()
    
    rouge_eval(**vars(args))
