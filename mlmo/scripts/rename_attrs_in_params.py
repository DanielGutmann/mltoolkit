import argparse
import torch as T
from mlutils.helpers.paths_and_files import safe_mkfdir
from mlmo.utils.constants.checkpoint import MODEL_PARAMS


def rename_attrs_in_params(input_fp, output_fp, old_attr_names, new_attr_names,
                           device='cpu'):
    """Renames a model's parameters, saves them to an output file."""
    assert len(old_attr_names) == len(new_attr_names)
    model_params = T.load(input_fp, device)[MODEL_PARAMS]
    for old_name, new_name in zip(old_attr_names, new_attr_names):
        model_params[new_name] = model_params[old_name]
        del model_params[old_name]
    # dumping to the disk
    safe_mkfdir(output_fp)
    T.save({MODEL_PARAMS: model_params}, f=output_fp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-input_fp", type=str)
    parser.add_argument("-output_fp", type=str)
    parser.add_argument("-old_attr_names", nargs='+')
    parser.add_argument("-new_attr_names", nargs='+')

    args = parser.parse_args()
    rename_attrs_in_params(**vars(args))
