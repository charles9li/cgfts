from __future__ import absolute_import


class BinarySet(object):

    def __init__(self, item1, item2):
        self._items = {item1, item2}

    @property
    def items(self):
        return self._items

    def __eq__(self, other):
        return self.items == other.items

    def __iter__(self):
        return iter(self._items)

    def sorted(self):
        return sorted(self._items)
