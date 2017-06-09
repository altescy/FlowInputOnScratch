# -*- coding: utf-8 -*-

import numpy as np
from chainer import links as L, serializers

from matching.recognizer import recognize_locus
from languagemodel.model import LangModel
from languagemodel.predict import *

MODELFILE = './languagemodel/model/langmodel.npz'
lm = L.Classifier(LangModel(len(token2id), 650, train=False))
serializers.load_npz(MODELFILE, lm)


def softmax(x):
    x = np.asarray(x, dtype=np.float32)
    exp_x = np.exp(x)
    sum_exp = np.sum(exp_x)
    return exp_x / sum_exp


def get_lm_prob(candidates):
    y = lm.predictor.y
    if y is None:
        return np.zeros(len(candidates), dtype=np.float32)

    y = y[0].data
    ret = []
    for w in candidates:
        if w not in token2id:
            w = '<unk>'
        ret.append(y[token2id[w]])
    return np.array(ret, dtype=np.float32)


def compute_candidates(l):
    candidates, diffs = recognize_locus(l)
    if candidates is None:
        return None

    prob_lm = softmax(get_lm_prob(candidates))
    scores = diffs - prob_lm
    return [candidates[i] for i in np.argsort(scores)]


def predict_nexts(w):
    y = preinput(lm, w)
    nexts = []
    idxs = np.argsort(-y[0].data)
    for i in idxs:
        n = id2token[i]
        if n in ['<bos>', '<unk>']:
            continue
        if n == '<eos>':
            n = '.'
        nexts.append(n)
    return nexts


def reset_state():
    lm.predictor.reset_state()
