import errno
import json
import logging
import os


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def save_config(cfg, path):
    with open(path, 'w') as f:
        f.write(cfg.dump())
