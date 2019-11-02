from mldp.steps.transformers.base_transformer import BaseTransformer
from numpy.random import permutation
from mlutils.helpers.multi_processing import get_os_seed
import numpy as np


class Shuffler(BaseTransformer):
    """
    Shuffles data chunk values. Can be used to break sequential order dependency
    of data-units.

    A straightforward application of the shuffler is to shuffle large data-chunk
    produced by a reader and afterwards batch data units into smaller data-chunks

    For example, the architecture could look like this:
        Reader -> Shuffler -> ValueTransformer -> ChunkAccumulator -> Formatter
    """

    def __init__(self, reseed=False, seed=None, **kwargs):
        """
        :param seed: initial seed.
        :param reseed: whether to reseed the shuffler based on OS random numbers
                       every time transform is called. It's useful for running
                       on multiple processes.
        """
        super(Shuffler, self).__init__(**kwargs)
        assert isinstance(reseed, bool)
        self.reseed = reseed
        np.random.seed(seed)

    def _transform(self, data_chunk):
        """
        :param data_chunk: self.explanatory.
        :return: data-chunk with shuffled field values.
        """
        if self.reseed:
            np.random.seed(get_os_seed())
        shuffled_order = permutation(range(len(data_chunk)))
        for key in data_chunk.keys():
            val = data_chunk[key]
            if isinstance(val, np.ndarray):
                shuff_vals = val[shuffled_order]
            elif isinstance(val, list):
                shuff_vals = [val[indx] for indx in shuffled_order]
            else:
                raise TypeError("Can't shuffle the value type.")
            data_chunk[key] = shuff_vals
        return data_chunk
