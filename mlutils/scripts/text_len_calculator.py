import argparse
from mldp.steps.readers import CsvReader, JsonReader
from sacremoses import MosesTokenizer
import numpy as np
import os

moses_tokenizer = MosesTokenizer()


def calculate_text_length(input_fp, column_names, use_moses=True, sep='\t'):
    """
    Calculates length mean and std of text in a csv file. Uses moses tokenizer.
    """
    use_moses = bool(use_moses)
    tokenize = moses_tokenizer.tokenize if use_moses else lambda x: x.split()
    _, ext = os.path.splitext(input_fp)
    assert ext in ['.csv', '.json']
    reader = CsvReader(sep=sep, engine='python') if ext == '.csv' else JsonReader()
    lens = []
    for chunk in reader.iter(data_path=input_fp):
        for cname in column_names:
            for text in chunk[cname]:

                # TODO: change it, this solution seems too hacky!
                if isinstance(text, list):
                    text = " ".join(text)

                tokens = tokenize(text)
                lens.append(len(tokens))

    print "min: %f" % np.min(lens)
    print("mean: %f" % np.mean(lens))
    print("std: %f" % np.std(lens))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-input_fp", type=str)
    parser.add_argument("-column_names", type=str, nargs='+')
    parser.add_argument("-use_moses", type=int, default=1)
    parser.add_argument("-sep", type=str, required=False)
    args = parser.parse_args()
    calculate_text_length(**vars(args))
