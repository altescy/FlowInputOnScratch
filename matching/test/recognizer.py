# -*- coding: utf-8 -*-
"""
Created on Fri May  5 15:54:30 2017

@author: altescy
"""


import pickle
import numpy as np



LOCUS_FILE = '../vocabs/data/vocablocus.pkl'

with open(LOCUS_FILE, 'rb') as f:
    LOCUS_DATA = pickle.load(f)

VLOCUS = LOCUS_DATA['locus']
INITIALS = LOCUS_DATA['meta']['keyconf']
SORTED_KEYS = sorted(INITIALS)



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
    return initial, list(VLOCUS[initial].keys())


def recognize_locus(l):
    """
    最も可能性の高いn個の単語を返す．
    """
    if len(l) == 0:
        return None
    
    l = np.asarray(l, dtype=np.float32)
    initial, candidates = choose_candidates(l)
    
    ds = []
    for w in candidates:
        d = compute_difference(l, VLOCUS[initial][w])
        ds.append(d)
        # print('l <-> {}: {}'.format(w, d))
    return [candidates[idx] for idx in np.argsort(ds)]


if __name__ == '__main__':
    l = VLOCUS['w']['was']
    print(recognize_locus(l))
