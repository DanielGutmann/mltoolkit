from torch.nn import Module, GRU
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from mlmo.utils.helpers.pytorch.data import reverse_padded_sequence


class RevGruEncoder(Module):
    """Reverse GRU based encoder that encodes sequences backwards."""

    def __init__(self, input_dim, hidden_dim):
        super(RevGruEncoder, self).__init__()
        self.gru = GRU(input_dim, hidden_dim, batch_first=True)

    def forward(self, inp, lens, only_last=False):
        """
        :param inp: [batch_size, seq_len, input_dim]
        :param lens: [batch_size]
        :return [batch_size, hidden_dim]
        """
        inp = reverse_padded_sequence(inp, lens, batch_first=True)
        packed_inp = pack_padded_sequence(inp, lens, batch_first=True)
        out, hidden = self.gru(packed_inp)
        if only_last:
            return hidden[0]
        else:
            out, _ = pad_packed_sequence(out, batch_first=True)
            out = reverse_padded_sequence(out, lens, batch_first=True)
            return hidden[0], out
