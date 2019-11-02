from mldp.steps.transformers import BaseTransformer


class Postfixer(BaseTransformer):
    """Adds a special postfix to each entry of a field.

    The step records the unique identifiers in chunks, and adds a count postfix
    to each data unit's `id_fname`.
    """

    def __init__(self, id_fname, **kwargs):
        """
        :param id_fname: unique identifier for an entity, e.g., `group_id` or
            `product_id`.
        """
        super(Postfixer, self).__init__(**kwargs)
        self.id_fname = id_fname
        self._id_to_count = {}

    def _transform(self, data_chunk):
        for du in data_chunk.iter():
            id = du[self.id_fname]
            if id not in self._id_to_count:
                self._id_to_count[id] = len(self._id_to_count) + 1
            count = self._id_to_count[id]
            du[self.id_fname] = "%s_%d" % (du[self.id_fname], count)
        return data_chunk
