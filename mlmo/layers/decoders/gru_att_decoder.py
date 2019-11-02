from torch.nn import Module, Linear, GRUCell, LayerNorm
import torch as T


class GruAttDecoder(Module):
    """
    Attention based GRU decoder that attends a custom sequence. Another
    difference from the classical gru decoder is that here I manually loop over
    steps instead of relying on the internals provided by PyTorch.

    Concatenates the previous context vectors to the word embeddings, then
    passes through an GRU cell. Outputs are concatenations of current context
    vectors and current hidden states, which can be used to obtain word
    distributions by a separate FFNN.

    Optionally computes the coverage vector and uses it as an additional input
    to the attention mechanism.
    """

    def __init__(self, input_dim, contxt_dim, hidden_dim, att_module,
                 input_norm=False, hidden_norm=False, cat_contx_to_inp=True,
                 **kwargs):
        """
        :param input_dim: usually embeddings dim.
        :param hidden_dim: dimensionality of the hidden layer.
        :param att_module: module to attend over some external values.
        :param input_norm: if set to True will use layer normalization to
                           normalize inputs to the decoder.
        :param hidden_norm: if set to True will use layer normalization over
                            produced hidden states.
        :param cat_contx_to_inp: whether to concatenated context vector to the
                                 input word embeddings.
        :param kwargs:
        """
        super(GruAttDecoder, self).__init__()
        self.input_dim = input_dim
        self.contxt_dim = contxt_dim
        self.hidden_dim = hidden_dim
        self.att = att_module
        self.input_norm = input_norm
        self.hidden_norm = hidden_norm
        self.cat_conxt_to_inp = cat_contx_to_inp

        cell_inp_dim = input_dim + contxt_dim if cat_contx_to_inp else input_dim

        self.gru_cell = GRUCell(cell_inp_dim, hidden_dim, **kwargs)

        if self.input_norm:
            self.inp_norm_layer = LayerNorm(input_dim + contxt_dim)
        if self.hidden_norm:
            self.hidden_norm_layer = LayerNorm(hidden_dim)

    def forward(self, x, mask, init_hidden, att_keys, att_values,
                att_mask=None, init_cont=None, **att_kwargs):
        """
        Performs forward pass given input and optional values to attend. If
        attention values are not provided, the corresponding output vector's
        part will be filled with zeroes.

        :param x: [batch_size, out_seq_len, input_dim]
                  sequences of input embeddings of the 'input_dim' size.
        :param mask: [batch_size, out_seq_len]
        :param init_hidden: [batch_size, hidden_dim]
        :param att_keys: [batch_size, inp_seq_len, att_hidden_dim]
        :param att_values: [batch_size, inp_seq_len, value_dim]
        :param att_mask: [batch_size, inp_seq_len]
        :param init_cont: [batch_size, value_dim]
                          initial context vector.
        :param att_kwargs: additional parameters that should be passed to the
                           forward method of the attention module.
        :return: outs: [batch_size, out_seq_len, *]
                       hidden states concatenated with contxt_vecs
                 prev_hidden: [batch_size, hidden_dim]
                              last hidden layer.
                 att_weights: [batch_size, out_seq_len, inp_seq_len]
                              normalized scores of where the model was attending
                              over the 'out_seq_len' steps decoding.
                 prev_cont: [batch_size, value_dim]
                            last context vector produced by the attention.
        """
        assert len(x.shape) == 3
        assert len(init_hidden.shape) == 2
        assert len(att_values.shape) == 3
        assert len(att_mask.shape) == 2
        bs = x.size(0)
        out_seq_len = x.size(1)
        inp_seq_len = att_values.size(1)
        device = x.device
        init_hidden = init_hidden

        # collectors
        outs = T.empty((bs, out_seq_len, self.hidden_dim + self.contxt_dim),
                       dtype=T.float32, device=device)
        att_weights = T.empty((bs, out_seq_len, inp_seq_len), dtype=T.float32,
                              device=device)
        if init_cont is None:
            init_cont = T.zeros((bs, self.contxt_dim), device=device)

        if self.hidden_norm:
            init_hidden = self.hidden_norm_layer(init_hidden)

        prev_hidden = init_hidden
        prev_cont = init_cont
        
        mask = mask.unsqueeze(-1)
        
        for indx in range(out_seq_len):
            _x = x[:, indx]
            _m = mask[:, indx]
            hidden, cont, att_wts = self.step(_x, _m, prev_hidden=prev_hidden,
                                              prev_cont=prev_cont,
                                              att_keys=att_keys,
                                              att_values=att_values,
                                              att_mask=att_mask,
                                              **att_kwargs)

            att_weights[:, indx] = att_wts
            outs[:, indx] = T.cat((hidden, cont), dim=1)

            prev_hidden = hidden
            prev_cont = cont

        return outs, att_weights, prev_hidden, prev_cont

    def step(self, emb, m, prev_hidden, prev_cont, att_keys, att_values,
             att_mask, **att_kwargs):
        """
        Performs a step by running the decoder.

        :param emb: [batch_size, emb_dim]
                    word embeddings.
        :param m: [batch_size]
                  mask for the current time-step.
        :param prev_cont: [batch_size, conxt_dim]
        :param att_keys: [batch_size, seq_len, dim]
        :param att_values: [batch_size, seq_len, dim]
        :param att_mask: [batch_size]
        """
        if self.cat_conxt_to_inp:
            inp = T.cat((emb, prev_cont), dim=1)
        else:
            inp = emb

        # optional input layer normalization
        if self.input_norm:
            inp = self.inp_norm_layer(inp)

        curr_hidden = self.gru_cell(inp, prev_hidden)

        # optional hidden states normalization
        if self.hidden_norm:
            curr_hidden = self.hidden_norm_layer(curr_hidden)

        curr_cont, att_wts = self.att(query=curr_hidden, key=att_keys,
                                      value=att_values, mask=att_mask,
                                      **att_kwargs)

        # copying the hidden layer and context vectors for masked seqs
        new_hidden = m * curr_hidden + (1. - m) * prev_hidden
        new_cont = m * curr_cont + (1. - m) * prev_cont

        return new_hidden, new_cont, att_wts
