from torch.nn import Module, GRUCell
import torch as T


class GruMaskedDecoder(Module):
    """
    This decoder operates based on mask and has a drop-out component in it,
    which is applied to every time-step hidden state.

    This decoder allows to pass a dropout mask that is applied to each hidden
    state.
    """

    def __init__(self, input_dim, hidden_dim, **gru_kwargs):
        super(GruMaskedDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.gru_cell = GRUCell(input_dim, hidden_dim, **gru_kwargs)

    def forward(self, x, mask, init_hidden=None, dropout_mask=None):
        """
        :param x: [batch_size, seq_len, input_dim]
                  embedded input sequences.
        :param mask: [batch_size, seq_len]
        :param init_hidden: [batch_size, hidden_dim]
        :param dropout_mask: [batch_size, hidden_dim]
                             values dropout mask that is applied to each
                             hidden state.
        :return: hiddens: [batch_size, seq_len, hidden_dim]
                          collected hidden states over time-steps
                 last_hidden: [batch_size, hidden_dim]
        """
        assert init_hidden is None or len(init_hidden.shape) == 2
        bs = x.size(0)
        seq_len = x.size(1)
        device = x.device

        hiddens = T.empty((bs, seq_len, self.hidden_dim), dtype=T.float32,
                          device=device)

        prev_hidden = init_hidden if init_hidden is not None else \
            T.zeros((bs, self.hidden_dim), device=device)

        mask = mask.unsqueeze(-1)

        for t in range(seq_len):
            _x = x[:, t]
            _m = mask[:, t]

            if dropout_mask is not None:
                prev_hidden = prev_hidden * (dropout_mask * _m + (1. - _m))
            curr_hidden = self.gru_cell(_x, prev_hidden)

            prev_hidden = _m * curr_hidden + (1. - _m) * prev_hidden

            hiddens[:, t] = prev_hidden

        last_hidden = prev_hidden

        return hiddens, last_hidden
