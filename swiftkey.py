# -*- coding: utf-8 -*-

import blockext
import numpy as np
from pandas.io.clipboard import to_clipboard

from core import compute_candidates, predict_nexts, reset_state
from matching.recognizer import STEP


def make_len(x, length, pad=None):
    if len(x) > length:
        return x[:length]

    ret = [pad] * length
    for i, e in enumerate(x):
        ret[i] = e
    return ret


class SwiftKey:
    def __init__(self):
        self.reset_all()

    def reset_all(self):
        self.upper = 0
        self.hold = 0
        self.locus = []
        self.result = [None, None, None]
        self.completed = 0
        reset_state()
        self.valid_result(predict_nexts('<bos>'))

    def append(self, p):
        if len(self.locus) > 0:
            tmp = np.array([self.locus[-1], p], dtype=np.int32)
            if np.linalg.norm(tmp[0] - tmp[1]) < STEP * 0.6:
                return
        self.locus.append(p)

    def reset(self):
        self.locus = []
        self.result = [None, None, None]
        self.completed = 0

    def recognize(self):
        res = compute_candidates(self.locus)
        if res is None:
            self.completed = -1
            return

        self.valid_result(res)
        self.completed = 1

    def input_prev(self, w):
        if w is not None:
            self.valid_result(predict_nexts(w))

    def valid_result(self, res):
        if res is None:
            return

        res = make_len(res, 3)
        for i, w in enumerate(res):
            if w is None:
                continue
            if self.upper:
                if self.hold:
                    w = w.upper()
                else:
                    w = w[0].upper() + w[1:] if len(w) > 1 else w.upper()
            else:
                w = w.lower()
            self.result[i] = w
        print(self.result)



sk = SwiftKey()

@blockext.command('reest')
def reset():
    sk.reset()

@blockext.command('initialize')
def initialize():
    sk.reset_all()

@blockext.command('append x: %n, y:%n')
def append(x, y):
    sk.append([x, y])

@blockext.command('recognize')
def recognize():
    sk.recognize()

@blockext.command('input this: %s')
def input_this(w):
    sk.input_prev(w)

@blockext.command('copy %s')
def copy(s):
    to_clipboard(s)

@blockext.reporter('first candidate')
def get_first():
    return '' if sk.result[0] is None else sk.result[0]

@blockext.reporter('second candidate')
def get_second():
    return '' if sk.result[1] is None else sk.result[1]

@blockext.reporter('third candidate')
def get_third():
    return '' if sk.result[2] is None else sk.result[2]

@blockext.reporter('completed')
def completed():
    return sk.completed

@blockext.command('change upper')
def change_upper():
    sk.upper = 1
    sk.valid_result(sk.result)

@blockext.command('change lower')
def change_lower():
    sk.upper = 0
    sk.valid_result(sk.result)

@blockext.command('hold on shift')
def hold_on_shift():
    sk.hold = 1
    sk.valid_result(sk.result)

@blockext.command('hold out shift')
def hold_out_shift():
    sk.hold = 0
    sk.valid_result(sk.result)

@blockext.reporter('shift')
def shift():
    return sk.upper

@blockext.reporter('hold')
def hold():
    return sk.hold

if __name__ == '__main__':
    blockext.run('SwiftKey', 'swiftkey', 5678)
