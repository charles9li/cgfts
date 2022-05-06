from __future__ import absolute_import


class SortedBinarySet(object):

    def __init__(self, item1, item2):
        self._items = sorted([item1, item2])

    @property
    def items(self):
        return self._items

    def __eq__(self, other):
        for i, j in zip(self.items, other.items):
            if i != j:
                return False
        return True

    def __iter__(self):
        return iter(self._items)

    def __hash__(self):
        return hash(tuple(self._items))
