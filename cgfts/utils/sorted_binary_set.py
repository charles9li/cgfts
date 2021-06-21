from __future__ import absolute_import


class SortedBinarySet(object):

    def __init__(self, item1, item2):
        self._items = sorted([item1, item2])

    @property
    def items(self):
        return self._items

    def __eq__(self, other):
        return self.items == other.items

    def __iter__(self):
        return iter(self._items)
