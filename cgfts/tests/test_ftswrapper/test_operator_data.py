from __future__ import absolute_import

import unittest

from cgfts.ftswrapper import OperatorData


class TestOperatorData(unittest.TestCase):

    def test1(self):

        # initialize OperatorData instance
        op = OperatorData("example_operator_data/operators1.dat")

        # check that number of columns is correct
        self.assertEqual(15, op.n_columns)

        # check that length of data is correct
        self.assertEqual(84, len(op.get_by_column_index(0)))

        # check that final values for operators are correct
        self.assertEqual(8.8339026450e+01, op.get_by_column_name('Hamiltonian')[-1])
        self.assertEqual(8.8336704588e+01, op.get_by_column_name('Pressure')[-1])


if __name__ == '__main__':
    unittest.main()
