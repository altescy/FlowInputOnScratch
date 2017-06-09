# -*- coding: utf-8 -*-

import numpy as np
import chainer
from chainer import links as L, functions as F
from chainer import Variable, serializers

from languagemodel.model import LangModel

token2id = chainer.datasets.get_ptb_words_vocabulary()
id2token = {token2id[token]: token for token in token2id}

def convert(sentence):
    """
    与えられた文字列を言語モデルで処理できるよう変換する
    """
    sentence = sentence.lower().strip()
    
    for s in ',@#$%^&*(){}-=_+\|<>~`"':
        sentence = sentence.replace(s, '')
    for s in '.?!;:':
        sentence = sentence.replace(s, ' <eos> <bos> ')
    sentence = sentence.replace("'", " '")
    
    tokens = sentence.split()
    
    x = []
    for token in tokens:
        if token not in token2id:
            token = '<unk>'
        x.append(token2id[token])
    return x


def one_input(model, w):
    return model.predictor(Variable(np.array([w], np.int32), volatile='on'))


def id_with_highest_prob(y):
    return int(np.argmax(y[0].data))


def preinput(model, sentence):
    x = convert(sentence)
    y = np.expand_dims(np.random.uniform(size=len(token2id)).astype(np.int32), 0)
    y = Variable(y, volatile='on')
    for w in x:
        y = one_input(model, w)
    
    return y


def gentext(model, prevs, maxlen=20):
    print(prevs, end=' ')
    prev = id_with_highest_prob(preinput(model, prevs))
    
    for i in range(maxlen):
        if prev == token2id['<eos>']:
            print('.')
            break
        print(id2token[prev], end=' ')
        prev = id_with_highest_prob(one_input(model, prev))


if __name__ == '__main__':
    MODELFILE = './model/langmodel.npz'


    model = L.Classifier(LangModel(len(token2id), 650, train=False))
    serializers.load_npz(MODELFILE, model)

    gentext(model, "<bos>")
    
