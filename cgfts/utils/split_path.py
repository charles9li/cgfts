from __future__ import absolute_import

import os

__all__ = ['split_path']


def split_path(path):
    folders = []
    while True:
        path, folder = os.path.split(path)
        if folder != "":
            folders.append(folder)
        elif path == "":
            break
    folders.reverse()
    return folders
