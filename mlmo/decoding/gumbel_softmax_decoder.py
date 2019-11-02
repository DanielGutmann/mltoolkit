from torch.nn import Module
from torch.nn.functional import gumbel_softmax
import torch as T
from mlmo.utils.tools import DecState


# TODO: refactor this decoder to work with the modifications of the codebase
class GumbelSoftmaxDecoder(object):
    """
    PyTorch implementation of Gumbel Softmax decoder. Can be used for
    training and development purposes.

    Notice that it's not a Pytorch module but just a custom class that wraps
    word embeddings and a decoder's function, both of which should be PyTorch
    modules.

    This version returns a soft mask that can be used for encoding and attenting
    over produced hidden states.
    """

    def __init__(self, decoding_func, embds_module, start_id, end_id):
        """
        :param decoding_func: function that inputs:
                              1. prev token embds [batch_size, emb_dim]
                              2. mask [batch_size, 1]
                              3. initial hidden state [batch_size, hidden_dim]
                              and outputs:
                              1. next words scores/probs [batch_size, 1, vocab_size]
                              2. recurrent values that need to be fed back to 
                                 the model
                              3. artifacts that need to be collected and 
                                 returned
        :param embds_module: module that contains embeddings of size
                             [vocab_size, emb_dim]
        :param start_id: id of the symbol that starts sequences.
        :param end_id: id of the symbol that ends sequences.
        """
        super(GumbelSoftmaxDecoder, self).__init__()
        self.decoding_func = decoding_func
        self.embds = embds_module
        self.vocab_size = embds_module.weight.data.size(0)
        self.emb_dim = embds_module.weight.data.size(1)
        self.start_id = start_id
        self.end_id = end_id

    def __call__(self, init_hidden, max_len, tau=1., **dec_kwargs):
        """
        Decoding of sequences where tokens have to be sampled from Gumbel
        softmax. Output ids are not padded after 'end_id', but instead if a
        sequence ends with 'end_id' it will be reflected in 'decoded_lens'.

        It collects additional artifacts that are produced by the decoding
        function returns them together with the output.

        :param init_hidden: [batch_size, hidden_dim]
        :param max_len: maximum length of the produced summaries.
        :param tau: annealing temperature of Gumbel softmax (used only for
                    training).
        :param dec_kwargs: additional decoder's parameters.
        :return: decoded_word_hots: [batch_size, seq_len, vocab_size]
                 word_embds: [batch_size, seq_len, emb_dim]
                 artifacts_coll: tuple of [batch_size, seq_len, *]
        """
        assert max_len > 1
        bs = init_hidden.size(0)
        device = init_hidden.device

        decoded_word_hots = []
        decoded_mask = []
        word_embds = []

        start_one_hot = T.zeros((bs, self.vocab_size), dtype=T.float32,
                                device=device)
        start_one_hot[:, self.start_id] = 1.

        # the start symbol is not decoded so just appending corresponding
        # dummy mask and embedding
        decoded_word_hots.append(start_one_hot)
        word_embds.append(self._embed_one_hot(start_one_hot))
        decoded_mask.append(T.ones(bs, device=device))

        arts_coll = {}  # additional artifacts collector

        # previous recurring items from the decoder
        # TODO: can I get rid off the hard-coded name?
        # TODO: E.g. use constants instead
        prev_recc = {"init_hidden": init_hidden}

        for t in range(1, max_len):

            # computing the soft mask
            mask = self._create_mask(prev_word_hots=decoded_word_hots[-1],
                                     prev_mask=decoded_mask[-1])
            decoded_mask.append(mask)

            # merging static and recurrent values together
            new_kwargs = merge_dicts(dec_kwargs, prev_recc)

            prev_words_embds = word_embds[-1].unsqueeze(1)
            mask = mask.unsqueeze(1)
            out = self.decoding_func(prev_words_embds, mask, **new_kwargs)
            assert isinstance(out, DecState)

            scores = out.word_scores

            # updating the recurring values so they would be fed in the next
            # loop
            prev_recc = out.rec_vals

            # collecting artifacts if produced by the decoder
            if out.coll_vals:
                collect_arts(arts_coll, out.coll_vals)

            scores = scores.squeeze(dim=1)
            curr_word_hots = gumbel_softmax(scores, hard=True, tau=tau)

            decoded_word_hots.append(curr_word_hots)
            word_embds.append(self._embed_one_hot(curr_word_hots))

        # converting to tensors
        decoded_word_hots = T.stack(decoded_word_hots, dim=1)
        word_embds = T.stack(word_embds, dim=1)
        decoded_mask = T.stack(decoded_mask, dim=1)

        if arts_coll:
            # concatenating artifacts produced by the decoder over sequences
            for k, ar in arts_coll.items():
                arts_coll[k] = T.cat(ar, dim=1)

            return decoded_word_hots, word_embds, decoded_mask, arts_coll

        return decoded_word_hots, word_embds, decoded_mask

    def _create_mask(self, prev_word_hots, prev_mask):
        """
        Creates gradient propagatable mask as m_t = (1 - p_{t-1}) * m_{t-1}.
        Where p is a gumbel softmax sample (one-hot during forward and soft
        during backwards pass)
        """
        mask = (1. - prev_word_hots[:, self.end_id]) * prev_mask
        return mask

    def _embed_one_hot(self, one_hot):
        return T.mm(one_hot, self.embds.weight)


def merge_dicts(dct1, dct2):
    new_dict = {}
    for dct in [dct1, dct2]:
        for k,v in dct.items():
            assert k not in new_dict
            new_dict[k] = dct[k]
    return new_dict


def collect_arts(coll, new_arts):
    """Collects recently produced artifacts."""
    for k, v in new_arts.items():
        if k not in coll:
            coll[k] = []
        coll[k].append(v)
