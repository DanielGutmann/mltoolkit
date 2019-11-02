from mlutils.helpers.paths_and_files import safe_mkfdir
from mlutils.tools import OrderedAttrs
from collections import OrderedDict
import os
from logging import getLogger
import json
import codecs

OVERRIDABLE_ATTRS = (list, int, float, bool, str, dict, OrderedAttrs)
FLOAT_FORMAT = "%.3f"
MAX_LINE_LEN = 70
logger_name = os.path.basename(__file__)
logger = getLogger(logger_name)


class BaseHp(OrderedAttrs):
    """
    Children objects of this class can be used to store default hyper-parameters
    override then when necessary through loading, and then save them to a json
    file. 

    Not all hyper-param fields can be saved and loaded, only of simple types,
    such as string, integer, boolean, float, list, etc.
    """

    def save(self, file_path, encoding='utf-8'):
        """Saves hyper-params object as a json file."""
        safe_mkfdir(file_path)
        hparams_to_save = self._get_simple_attrs()
        f = codecs.open(file_path, encoding=encoding, mode='w')
        json.dump(hparams_to_save, f, indent=2)
        logger.debug("Extracted the following hparams: '%s'."
                     "" % " ".join(hparams_to_save.keys()))
        logger.info("Saved hyper-parameters to '%s'." % file_path)    

    def load(self, file_path, encoding='utf-8'):
        """Loads hyper-params from a json file."""
        f = codecs.open(file_path, encoding=encoding)
        hparams = json.load(f, encoding=encoding)
        for attr_name, attr_val in hparams.items():
            if not hasattr(self, attr_name):
                logger.warn("Could not override '%s' because the field does not"
                            " exist." % attr_name)
            setattr(self, attr_name, attr_val)
        logger.info("Loaded hyper-parameters from '%s'." % file_path)

    def _get_simple_attrs(self):
        """Returns an ordered dict with simple attribute types."""
        hparams_to_save = OrderedDict()
        for attr_name in self.__odict__:
            attr_val = getattr(self, attr_name)
            if isinstance(attr_val, OVERRIDABLE_ATTRS):
                hparams_to_save[attr_name] = attr_val
        return hparams_to_save

    def __str__(self):
        """
        Creates a string from the object for logging and printing.
        Lines are splat by the new line based on the maximum length
        chars (constant).
        """
        hparams_to_save = self._get_simple_attrs()
        lines = []
        curr_line_len = 0
        curr_attrs = []
        for name, val in hparams_to_save.items():
            if curr_line_len >= MAX_LINE_LEN:
                lines.append(", ".join(curr_attrs))
                curr_line_len = 0
                curr_attrs = []
            if isinstance(val, float):
                template = "%s: "+FLOAT_FORMAT
            else:
                template = "%s: %s"
            name_val_str = template % (name, val)
            curr_line_len += len(name_val_str)
            curr_attrs.append(name_val_str)
        if curr_line_len > 0:
            lines.append(", ".join(curr_attrs))
        lines_str = "\n".join(lines)
        return lines_str
