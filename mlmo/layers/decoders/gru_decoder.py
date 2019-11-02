from torch.nn import Module, GRU
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class GruDecoder(Module):
    """
    Basic implementation of a GRU based decoder that outputs all hidden
    states and the last hidden layer.

    The decoder can be used both for training and test time. In the latter phase
    , one could feed sequences with single tokens (i.e. seq_len=1), and previous
    states.
    """

    def __init__(self, input_dim, hidden_dim, **kwargs):
        super(GruDecoder, self).__init__()
        self.gru = GRU(input_dim, hidden_dim, batch_first=True, **kwargs)

    def forward(self, x, lens, init_hidden):
        """
        :param x: [batch_size, seq_len, input_dim]
                  embedded input sequences.
        :param lens: [batch_size]
        :param init_hidden: [batch_size, hidden_dim]
        :return: un_normalized scores [batch_size, seq_len, vocab_size],
                 last hidden [batch_size, hidden_dim]
        """
        assert len(init_hidden.shape) == 2
        init_hidden = init_hidden.unsqueeze(0)
        packed_inp = pack_padded_sequence(x, lens, batch_first=True)
        output, hidden = self.gru(packed_inp, init_hidden)
        output, _ = pad_packed_sequence(output, batch_first=True)

        hidden = hidden.squeeze(0)

        return output, hidden
