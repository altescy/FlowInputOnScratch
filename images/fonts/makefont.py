# -*- coding: utf-8 -*-

import argparse
import numpy as np
from scipy.misc import imread, imsave


def get_charname(c):
    symbols = {
        '\\': 'bs',
        '/' : 'sl',
        '*' : 'as',
        ':' : 'cl',
        '?' : 'qs',
        '<' : 'lt',
        '>' : 'gt',
        '|' : 'vl',
        '"' : 'dq',
        '.' : 'pd',
        ' ' : 'sp',
    }
    if c in symbols:
        return symbols[c]

    if c.isupper():
        return 'u' + c.lower()

    return c



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--fonts', type=str, default='./data/fonts.png',
                        help='Path to font-image file.')
    parser.add_argument('-c', '--chars', type=str, default='./data/chars.txt',
                        help='Path to character-set file')
    parser.add_argument('-o', '--out', type=str, default='./images')
    args = parser.parse_args()

    imgfp = args.fonts
    charsfp = args.chars
    outpath = args.out

    with open(charsfp, 'r') as f:
        chars = f.read().strip()
    n = len(chars)

    img = 255 - imread(imgfp)[:, :, -1]
    cw = img.shape[1] // n

    print('char image size:', img.shape[0], cw)

    print('characters:', chars)

    print('now saving ...', end=' ')
    mw = 15
    for i in range(n):
        cn = get_charname(chars[i])
        a = cw * i
        b = cw * (i + 1)

        nz = np.asarray([np.sum(img[:, j] == 0) for j in range(max(0, a - mw), min(a + mw, img.shape[1]))])
        m = min(nz)
        candidate = np.argwhere(nz == m) + max(0, a - mw)
        j = int(np.clip(candidate[np.argmin(np.abs(candidate - a))], 0, img.shape[1]))


        nz = np.asarray([np.sum(img[:, k] == 0) for k in range(max(0, b - mw), min(b + mw, img.shape[1]))])
        m = min(nz)
        candidate = np.argwhere(nz == m) + max(0, b - mw)
        k = int(np.clip(candidate[np.argmin(np.abs(candidate - b))], 0, img.shape[1]))

        imsave(outpath + '/%s.png'%cn, img[:, j:k])



    print('completed.')
