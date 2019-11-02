from torch.nn import Module, GRU
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch as T


class BiGruEncoder(Module):
    """
    Bidirectional GRU encoder inputs sequences and outputs their representations,
    which are concatenated hidden states. Optionally returns other hidden states.
    """

    def __init__(self, input_dim, hidden_dim, **kwargs):
        super(BiGruEncoder, self).__init__()
        self.gru = GRU(input_dim, hidden_dim, bidirectional=True,
                       batch_first=True, **kwargs)

    def forward(self, seqs, seq_lens):
        """
        :param seqs: [batch_size, seq_len, input_dim]
        :param seq_lens: [batch_size]
        :return [batch_size, hidden_dim], [batch_size, seq_len, hidden_dim]
        """
        packed_inp = pack_padded_sequence(seqs, seq_lens, batch_first=True)
        out, hidden = self.gru(packed_inp)
        h = T.cat([hidden[0], hidden[1]], dim=1)
        out, _ = pad_packed_sequence(out, batch_first=True)
        return h, out
