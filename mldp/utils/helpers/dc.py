import numpy as np
from mldp.utils.tools import DataChunk


def concat_chunks(*dcs):
    """Combines data-chunks horizontally and returns them as one chunk."""
    new_dc = DataChunk()
    is_arr = None
    for dc in dcs:
        for k, v in dc.items():
            if k not in new_dc:
                new_dc[k] = []
            if isinstance(v, np.ndarray):
                if is_arr is False:
                    raise TypeError("All values must either 'arrays' or "
                                    "'lists'.")
                is_arr = True
                new_dc[k].append(v)
            elif isinstance(v, list):
                if is_arr is True:
                    raise TypeError("All values must either 'arrays' or "
                                    "'lists'.")
                is_arr = False
                new_dc[k] += v
            else:
                raise TypeError("Can't concat values other than 'lists' or "
                                "'arrays'.")

    if is_arr:
        for k in new_dc:
            new_dc[k] = np.concatenate(tuple(new_dc[k]))
    return new_dc


def merge_chunks(dc_one, dc_two, merge_key):
    """Merges (vertically) chunks together by a 'merge_key'."""
    if not (dc_one[merge_key] == dc_two[merge_key]).all():
        raise ValueError("Can't merge chunks that have different fvalues.")

    new_dc = DataChunk()
    new_dc[merge_key] = dc_two[merge_key]

    for dc in [dc_one, dc_two]:
        for k in dc:
            if k != merge_key:
                new_dc[k] = dc[k]

    return new_dc
