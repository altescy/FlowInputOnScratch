# -*- coding: utf-8 -*-
"""
Created on Sun May  7 07:40:56 2017

@author: Users001
"""

import numpy as np
import matplotlib.pyplot as plt

N_DIRCLASS = 7

def normalize(s):
    """
    文字列を正規化する
    """
    ret = ''
    prev = None
    for c in s:
        if c.isalpha() and c != prev:
            c = c.lower()
            ret += c
        prev = c
    return ret



def get_locus(s, keyconf, step=40):
    """
    文字列sをキーボードで入力する際の軌跡を求める
    """
    ret = []

    keys = []
    for c in s:
        keys.append(keyconf[c])
    
    for a, b in zip(keys[:-1], keys[1:]):
        ret.append(a)
        mids = get_midpoints(a, b, step)
        ret.extend(mids)
    
    ret.append(keys[-1])
    return np.array(ret, dtype=np.float32)


def get_midpoints(a, b, step):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    
    dx, dy = b - a
    alpha = dy / dx
    def f(x):
        return alpha * (x - a[0]) + a[1]
    
    step = dx / (np.linalg.norm(b - a) // step)
    x = np.arange(a[0] + step, b[0], step)
    
    return np.array([x, f(x)], dtype=np.float32).T


def show_locus(s, keyconf):
    s = normalize(s)
    ps = get_locus(s, keyconf)
    
    plt.plot(ps[:, 0], ps[:, 1])
    plt.plot(ps[:, 0], ps[:, 1], 'o')
    
    plt.xlim(-240, 240)
    plt.ylim(-180, 180)
    plt.show()


def compute_init_angle(l):
    if len(l) < 2:
        return None
    d = l[1] - l[0]
    return np.arctan2(d[1], d[0]) * 180 / np.pi

def compute_dir_class(a, b):
    """
    a -> b の移動方向を分類
    """
    d = b - a
    theta = np.arctan2(d[1], d[0]) * 180 / np.pi
    direction = 0

    if theta < -150:
        direction = 1
    elif theta < -90:
        direction = 2
    elif theta < -30:
        direction = 3
    elif theta < 30:
        direction = 4
    elif theta < 90:
        direction = 5
    elif theta < 150:
        direction = 6
    else:
        direction = 1
    
    return direction


def classify_init_dir(l):
    """
    a -> b の移動方向を4つに分類
    """
    if len(l) < 2:
        return 0
    
    return compute_dir_class(l[0], l[1])
