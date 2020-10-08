import warnings
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

def ifnone(x, y):
    if x is not None:
        return x
    else:
        return y

def first(x):
    "First element of `x`, or None if missing"
    try: return next(iter(x))
    except StopIteration: return None

def one_param(m):
    "First parameter in `m`"
    return first(m.parameters())

def dropout_mask(x:torch.Tensor, sz:Union[List, Tuple], p:float):
    "Return a dropout mask of the same type as `x`, size `sz`, with probability `p` to cancel an element."
    return x.new(*sz).bernoulli_(1-p).div_(1-p)

class RNNDropout(nn.Module):
    "Dropout with probability `p` that is consistent on the seq_len dimension."
    def __init__(self, p:float=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0.: return x
        return x * dropout_mask(x.data, (x.size(0), 1, *x.shape[2:]), self.p)

class EmbeddingDropout(nn.Module):
    def __init__(self, embed: nn.Embedding, embed_dropout_p: float):
        super().__init__()
        self.embed = embed
        self.embed_dropout_p = embed_dropout_p

    def forward(self, words:torch.Tensor):
        if self.training and self.embed_dropout_p !=0:
            size = (self.embed.weight.size(0), 1)
            mask = dropout_mask(self.embed.weight.data, size, self.embed_dropout_p)
            masked_embed = self.embed.weight * mask
        else:
            masked_embed = self.embed.weight
        
        return F.embedding(
            words, masked_embed, ifnone(self.embed.padding_idx, -1), self.embed.max_norm,
            self.embed.norm_type, self.embed.scale_grad_by_freq, self.embed.sparse
            )

class WeightDropout(nn.Module):
    def __init__(self, module:nn.Module, weight_p:float, layer_names=['weight_hh_l0']):
        super().__init__()
        self.module = module
        self.weight_p = weight_p
        self.layer_names = layer_names

        for layer in self.layer_names:
            #Makes a copy of the weights of the selected layers.
            w = getattr(self.module, layer)
            self.register_parameter(f'{layer}_raw', nn.Parameter(w.data))
            self.module._parameters[layer] = F.dropout(w, p=self.weight_p, training=False)

    def _setweights(self):
        "Apply dropout to the raw weights."
        for layer in self.layer_names:
            raw_w = getattr(self, f'{layer}_raw')
            self.module._parameters[layer] = F.dropout(raw_w, p=self.weight_p, training=self.training)

    def forward(self, *args):
        self._setweights()
        with warnings.catch_warnings():
            #To avoid the warning that comes because the weights aren't flattened.
            warnings.simplefilter("ignore")
            return self.module.forward(*args)

    def reset(self):
        for layer in self.layer_names:
            raw_w = getattr(self, f'{layer}_raw')
            self.module._parameters[layer] = F.dropout(raw_w, p=self.weight_p, training=False)
        if hasattr(self.module, 'reset'):
            self.module.reset()


class AWD_LSTM(nn.Module):
    def __init__(
        self,
        vocab_size:int,
        embed_dim:int,
        hidden_dim:int,
        num_layer:int,
        bi_dir:bool=True,
        hidden_dropout_p:float=0.2,
        embed_dropout_p:float=0.1,
        weight_dropout_p:float=0.5,
        input_dropout_p:float=0.6,
        padding_idx=1
        ):
        super().__init__()

        self.batch_size = 1
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layer = num_layer
        self.bi_dir = bi_dir
        self.hidden_dropout_p = hidden_dropout_p
        self.embed_dropout_p = embed_dropout_p
        self.weight_dropout_p = weight_dropout_p
        self.input_dropout_p = input_dropout_p
        self.padding_idx = padding_idx

        self.initrange = 0.1
        self.n_dir = 2 if self.bi_dir else 1

        self.embed = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=self.padding_idx)
        self.embed.weight.data.uniform_(-self.initrange, self.initrange)

        self.embed_dp = EmbeddingDropout(self.embed, self.embed_dropout_p)

        rnns = []
        for l in range(self.num_layer):
            weight_dropout_rnn = self._lstm_unit(
                self.embed_dim if l == 0 else self.hidden_dim,
                (self.hidden_dim if l != self.num_layer - 1 else embed_dim)//self.n_dir,
                self.bi_dir,
                self.weight_dropout_p,
                )
            rnns.append(weight_dropout_rnn)
        
        self.rnns = nn.ModuleList(rnns)
        self.input_dp = RNNDropout(self.input_dropout_p)
        self.hidden_dps = nn.ModuleList([RNNDropout(self.hidden_dropout_p) for _ in range(self.num_layer)])
        
        self.reset()
        

    def _lstm_unit(self, in_dim:int, out_dim:int, bi_dir:bool, weight_p):
        rnn = nn.LSTM(in_dim, out_dim, num_layers=1, bidirectional=bi_dir, batch_first=True)
        return WeightDropout(rnn, weight_p=weight_p)

    def reset(self):
        "Reset the hidden states"
        [r.reset() for r in self.rnns if hasattr(r, 'reset')]
        self.hidden = [self._one_hidden(l) for l in range(self.num_layer)]

    def _change_one_hidden(self, l, batch_size):
        if self.batch_size < batch_size:
            nh = (self.hidden_dim if l != self.num_layer - 1 else self.embed_dim) // self.n_dir
            return tuple(torch.cat([h, h.new_zeros(self.n_dir, batch_size-self.batch_size, nh)], dim=1) for h in self.hidden[l])
        if self.batch_size > batch_size:
            return (self.hidden[l][0][:, :batch_size].contiguous(), self.hidden[l][1][:, :batch_size].contiguous())
        return self.hidden[l]

    def _change_hidden(self, batch_size):
        self.hidden = [self._change_one_hidden(l, batch_size) for l in range(self.num_layer)]
        self.batch_size = batch_size

    def _one_hidden(self, l):
        "Return one hidden state"
        nh = (self.hidden_dim if l != self.num_layer - 1 else self.embed_dim) // self.n_dir
        return (one_param(self).new_zeros(self.n_dir, self.batch_size, nh), one_param(self).new_zeros(self.n_dir, self.batch_size, nh))

    def forward(self, x: torch.Tensor, from_embeds:bool=False):
        batch_size, seq_len = x[:2] if from_embeds else x.shape
        if batch_size != self.batch_size:
            self._change_hidden(batch_size)

        output = self.input_dp(x if from_embeds else self.embed_dp(x))
        new_hidden = []
        for l, (rnn,hid_dp) in enumerate(zip(self.rnns, self.hidden_dps)):
            output, new_h = rnn(output, self.hidden[l])
            new_hidden.append(new_h)
            if l != self.num_layer - 1: output = hid_dp(output)
        self.hidden = new_hidden
        return output