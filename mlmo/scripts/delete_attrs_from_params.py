import argparse
import torch as T
from mlutils.helpers.paths_and_files import safe_mkfdir
from mlmo.utils.constants.checkpoint import MODEL_PARAMS


def delete_attr_from_params(input_fp, output_fp, attr_names, device='cpu'):
    """Removes a particular attrs from the dictionary of params, saves back."""
    model_params = T.load(input_fp, device)[MODEL_PARAMS]

    for attr_name in attr_names:
        if attr_name in model_params:
            del model_params[attr_name]

    # dumping to the disk
    safe_mkfdir(output_fp)
    T.save({MODEL_PARAMS: model_params}, f=output_fp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_fp", type=str)
    parser.add_argument("--output_fp", type=str)
    parser.add_argument("--attr_names", nargs='+')

    args = parser.parse_args()
    delete_attr_from_params(**vars(args))
