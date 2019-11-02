from torch.nn import Module, GRUCell, Dropout, LayerNorm
import torch as T


class GruMaskedEncoder(Module):
    """
    GRU encoder inputs embedded sequences and outputs their hidden states.

    Uses layer normalization and dropout to make the encoder converge faster
    and robust.
    """

    def __init__(self, input_dim, hidden_dim, dropout_prob=0., hidden_norm=False,
                 **kwargs):
        super(GruMaskedEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout_prob = dropout_prob
        self.gru_cell = GRUCell(input_dim, hidden_dim, **kwargs)
        self.dropout_layer = Dropout(dropout_prob)
        self.hidden_norm = LayerNorm(hidden_dim) if hidden_norm else None

    def forward(self, x, mask):
        """
        :param x: [batch_size, seq_len, input_dim]
                  embedded input sequences.
        :param mask: [batch_size] or [batch_size, seq_len]
        :return  last hidden: [batch_size, hidden_dim]
                 all hiddens: [batch_size, seq_len, hidden_dim]
        """
        batch_size = x.size(0)
        max_seq_len = x.size(1)
        device = x.device
        out = T.empty((batch_size, max_seq_len, self.hidden_dim), device=device)
        prev_h = T.zeros((batch_size, self.hidden_dim), device=device)
        mask = mask.unsqueeze(-1)
        
        for t in range(max_seq_len):
            _x = x[:, t]
            _m = mask[:, t]

            new_h = self.gru_cell(_x, prev_h)

            if self.hidden_norm is not None:
                new_h = self.hidden_norm(new_h)

            if self.dropout_prob > 0.:
                new_h = self.dropout_layer(new_h)

            # copying previous hidden if it's masked
            prev_h = _m * new_h + (1. - _m) * prev_h

            out[:, t] = prev_h
            
        last_hidden = prev_h
        
        return last_hidden, out
