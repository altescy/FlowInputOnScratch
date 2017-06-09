# -*- coding: utf-8 -*-

import json
import blockext

keylocfile = '../keyconf.json'

class LocLogger(object):
    def __init__(self):
        self.log = {}

    def init_log(self):
        self.log = {}

    def add_loc(self, loc):
        name, x, y = loc.split(',')
        self.log[name] = [round(float(x)), round(float(y))]

    def save(self):
        with open(keylocfile, 'w') as f:
            json.dump(self.log, f)


logger = LocLogger()

@blockext.command("init log")
def init_log():
    logger.init_log()

@blockext.command("add loc %s")
def add_loc(loc):
    logger.add_loc(loc)

@blockext.command("save")
def save():
    logger.save()


if __name__ == '__main__':
    blockext.run("Loc Logger", "setup_keyboard", port=5678)
