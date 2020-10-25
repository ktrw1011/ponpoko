import numpy as np
import torch
import torch.nn as nn

from .utils import WeightDrop, embedded_dropout

class RegLSTM(nn.Module):
    def __init__(
        self,
        vocab_size:int,
        embed_dim:int,
        hidden_dim:int,
        num_layers:int,
        bi_dir:bool=True,
        hidden_dropout_p:float=0.2,
        embed_dropout_p:float=0.1,
        weight_dropout_p:float=0.5,
        input_dropout_p:float=0.6,
        padding_idx=1
        ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bi_dir = bi_dir

        self.hidden_dropout_p = hidden_dropout_p
        self.embed_dropout_p = embed_dropout_p
        self.weight_dropout_p = weight_dropout_p
        self.input_dropout_p = input_dropout_p

        rand_embed_init = torch.Tensor(vocab_size, embed_dim).uniform_(-0.25, 0.25)
        self.embed = nn.Embedding.from_pretrained(rand_embed_init, freeze=False)

        self.lstm = nn.LSTM(self.embed_dim, self.hidden_dim, dropout=self.hidden_dropout_p, num_layers=self.num_layers,
                            bidirectional=self.bi_dir, batch_first=True)

        if self.weight_dropout_p:
            self.lstm = WeightDrop(self.lstm, ['weight_hh_l0'], dropout=self.weight_dropout_p)

    def set_pretrainend_embedding(self, embed_mat:np.ndarray, freeze:bool=True):
        self.embed.weight = nn.Parameter(torch.Tensor(embed_mat).float())
        if freeze:
            self.embed.weight.requires_grad = False

    def forward(self, x):
        if self.embed_dropout_p:
            x = embedded_dropout(
                self.embed,
                x,
                dropout=(self.embed_dropout_p if self.training else 0)
                )
        else:
            x = self.embed(0)

        rnn_outs, _ = self.lstm(x)
        return rnn_outs



        

