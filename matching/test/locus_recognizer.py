# -*- coding: utf-8 -*-

import blockext
from recognizer import recognize_locus

class Logger:
    def __init__(self):
        self.upper = False
        self.hold = False
        self.locus = []
        self.result = [None, None, None]
        self.completed = 0

    def append(self, p):
        self.locus.append(p)

    def reset(self):
        self.locus = []
        self.result = [None, None, None]
        self.completed = 0

    def recognize(self):
        res = recognize_locus(self.locus)[:3]
        for i, w in enumerate(res):
            if self.upper:
                if self.hold:
                    w = w.upper()
                else:
                    w = w[0].upper() + w[1:]
            self.result[i] = w
        self.completed = 1


logger = Logger()

@blockext.command('reest')
def reset():
    logger.reset();

@blockext.command('append x: %n, y:%n')
def append(x, y):
    logger.append([x, y])

@blockext.command('recognize')
def recognize():
    logger.recognize()

@blockext.reporter('first')
def get_first():
    return '' if logger.result[0] is None else logger.result[0]

@blockext.reporter('second')
def get_second():
    return '' if logger.result[1] is None else logger.result[1]

@blockext.reporter('third')
def get_third():
    return '' if logger.result[2] is None else logger.result[2]

@blockext.reporter('completed')
def completed():
    return logger.completed

@blockext.command('change upper')
def change_upper():
    logger.upper = True

@blockext.command('change lower')
def change_lower():
    logger.upper = False

@blockext.command('hold on shift')
def hold_on_shift():
    logger.hold = True

@blockext.command('hold out shift')
def hold_out_shift():
    logger.hold = False

@blockext.reporter('shift')
def shift():
    return 1 if logger.upper else 0

@blockext.reporter('hold')
def hold():
    return 1 if logger.hold else 0

if __name__ == '__main__':
    blockext.run('LocusRecognizer', 'locus_recognizer', 5678)