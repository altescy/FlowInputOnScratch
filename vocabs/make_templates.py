# -*- coding: utf-8 -*-
"""
Created on Thu May  4 16:05:30 2017

@author: altescy
"""

import argparse
import json
import pickle
import numpy as np

from progressbar import ProgressBar

from locus_utils import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--vocab', type=str, default='./data/vocab.cleaned.txt')
    parser.add_argument('-k', '--keyconf', type=str, default='../keyboard/keyconf.json')
    parser.add_argument('-o', '--out', type=str, default='./data/vocablocus-angle.pkl')
    parser.add_argument('-s', '--step', type=int, default=40)
    args = parser.parse_args()
    
    with open(args.vocab, 'r') as f:
        vocabs = f.read().split()
    
    with open(args.keyconf, 'r') as f:
        keyconf = json.load(f)
    
    print('compute template locus ... ')
    pb = ProgressBar()
    locus = {chr(i): {} for i in range(97, 97 + 26)} #chr(97..97+25) -> a..z
    for i, s in enumerate(pb(vocabs)):
        initial = s[0].lower()
        if len(s) > 0 and s not in locus[initial]:
            l = get_locus(normalize(s), keyconf, step=args.step)
            a = compute_init_angle(l)
            locus[initial][s] = (a, l)
        pb.update(i + 1)
    
    keyconf = {key: np.array(keyconf[key], dtype=np.float32) for key in keyconf}
    
    meta = {'keyconf': keyconf,
            'step'   : args.step,}
    data = {'meta' : meta,
            'locus': locus,}
    with open(args.out, 'wb') as f:
        pickle.dump(data, f)
