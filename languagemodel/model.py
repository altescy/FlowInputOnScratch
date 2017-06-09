# -*- coding: utf-8 -*-

import numpy as np
import chainer
from chainer import links as L, functions as F


class LangModel(chainer.Chain):
    def __init__(self, n_vocab, n_units, train=True):
        super(LangModel, self).__init__(
            embed  = L.EmbedID(n_vocab, n_units),
            l1  = L.LSTM(n_units, n_units),
            l2  = L.LSTM(n_units, n_units),
            l3 = L.Linear(n_units, n_vocab),
        )
        
        self.y = None
        self.train = train

    def __call__(self, x):
        h = self.embed(x)
        h = self.l1(F.dropout(h, train=self.train))
        h = self.l2(F.dropout(h, train=self.train))
        self.y = self.l3(F.dropout(h, train=self.train))
        return self.y

    def reset_state(self):
        self.y = None
        self.l1.reset_state()
        self.l2.reset_state()