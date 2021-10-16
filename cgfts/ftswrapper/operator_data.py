from __future__ import absolute_import

import numpy as np


class OperatorData(object):

    def __init__(self, filename):

        # open file and read data
        f = open(filename, 'r')

        # get column headers from first line
        first_line = f.readline()
        self.column_names = first_line[1:].strip().split(' ')

        # read data
        data = np.loadtxt(f)

        # create dictionary of data
        self._data = dict()
        for i, col_name in enumerate(self.column_names):
            self._data[col_name] = data[:, i]

        # close file
        f.close()

    def get_by_column_name(self, column_name):
        return self._data[column_name]

    def get_by_column_index(self, column_index):
        column_name = self.column_names[column_index]
        return self.get_by_column_name(column_name)

    @property
    def n_columns(self):
        return len(self.column_names)
