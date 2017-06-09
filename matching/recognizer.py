# -*- coding: utf-8 -*-
"""
Created on Fri May  5 15:54:30 2017

@author: altescy
"""


import pickle
import numpy as np

from vocabs import locus_utils


LOCUS_FILE = 'vocabs/data/vocablocus-angle.pkl'

with open(LOCUS_FILE, 'rb') as f:
    LOCUS_DATA = pickle.load(f)

VLOCUS = LOCUS_DATA['locus']
INITIALS = LOCUS_DATA['meta']['keyconf']
SORTED_KEYS = sorted(INITIALS)
STEP = LOCUS_DATA['meta']['step']

THRESH_ANGLE = 12 # 初期移動角度のテンプレートとの許容誤差
THRESH_LENGTH = 4 # 系列長のテンプレートとの許容誤差


def get_initial(x):
    norms = [np.linalg.norm(x - INITIALS[key]) for key in SORTED_KEYS]
    return SORTED_KEYS[np.argmin(norms)]


def distance(x, y):
    return np.linalg.norm(x - y)


def compute_difference(a, b):
    # DP matching
    m, n = a.shape[0], b.shape[0]
    
    D = np.zeros((m, n), dtype=np.float32)
    
    D[0, 0] = distance(a[0], b[0])

    for i in range(1, m):
        D[i, 0] = D[i - 1, 0] + distance(a[i], b[0])

    for j in range(1, n):
        D[0, j] = D[0, j - 1] + distance(a[0], b[j])

    for i in range(1, m):
        for j in range(1, n):
            D[i, j] = min([D[i - 1, j - 1] + 2 * distance(a[i], b[j]),
                           D[i    , j - 1] +     distance(a[i], b[j]),
                           D[i - 1, j    ] +     distance(a[i], b[j]),])
    
    return D[-1, -1]


def choose_candidates(l):
    initial = get_initial(l[0])
    a = locus_utils.compute_init_angle(l)
    if a is None:
        return initial, [initial]
    
    candidates = []
    for w in VLOCUS[initial]:
        if VLOCUS[initial][w][0] is not None:
            if abs(VLOCUS[initial][w][0] - a) < THRESH_ANGLE:
                if abs(VLOCUS[initial][w][1].shape[0] - l.shape[0]) < THRESH_LENGTH:  
                    candidates.append(w)
    return initial, candidates


def recognize_locus(l):
    """
    　候補の単語と距離のタプルのリストを返す．
    """
    if len(l) == 0:
        return None
    
    l = np.asarray(l, dtype=np.float32)
    initial, candidates = choose_candidates(l)
    if len(candidates) < 1:
        return None, None
    
    print("#candidates: ", len(candidates))
    diffs = []
    for w in candidates:
        diff = compute_difference(l, VLOCUS[initial][w][1])
        diffs.append(diff)
        # print('l <-> {}: {}'.format(w, d))
    
    diffs = np.array(diffs, dtype=np.int32)
    return candidates, diffs

if __name__ == '__main__':
    l = VLOCUS['w']['was']
    print(recognize_locus(l))
