from argparse import ArgumentParser
from pandas import read_csv, concat
from csv import QUOTE_NONE
from mlutils.helpers.paths_and_files import safe_mkfdir


def merge_csv_files(input_fps, output_fp, sep="\t"):
    """
    Merges the csv files that have the same header into one file.
    Does not perform shuffling to avoid problems with the same group entries.
    """
    dfs = []
    for fp in input_fps:
        dfs.append(read_csv(fp, sep=sep, quoting=QUOTE_NONE, encoding='utf-8'))
    df = concat(dfs, axis=0, ignore_index=True, copy=True)
    safe_mkfdir(output_fp)
    df.to_csv(output_fp, sep=sep, index=False, encoding='utf-8',
              quoting=QUOTE_NONE)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--input_fps", type=str, nargs="+")
    parser.add_argument("--output_fp", type=str)
    parser.add_argument("--sep", type=str, default="\t")
    merge_csv_files(**vars(parser.parse_args()))
