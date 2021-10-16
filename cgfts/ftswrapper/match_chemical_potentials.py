from __future__ import absolute_import, division, print_function

import os

from scipy.optimize import root
import numpy as np

from .operator_data import OperatorData

__all__ = ['match_chemical_potentials']


def match_chemical_potentials(run_fts, target_chemical_potentials, directory="match_chemical_potentials"):

    # store current npw and read_input_fields of run_fts
    tmp_npw = run_fts.cell.npw
    tmp_input_fields_file = run_fts.init_fields.input_fields_file
    tmp_read_input_fields = run_fts.init_fields.read_input_fields

    # running single point simulations
    run_fts.cell.npw = 1
    run_fts.init_fields.read_input_fields = False

    # use root method
    x0 = np.append(np.array([run_fts.composition.c_chain_density]), run_fts.composition.chain_vol_frac[:-1])
    sol = root(_compute_chemical_potential_differences, x0, args=(target_chemical_potentials, run_fts, directory))
    run_fts.composition.c_chain_density = sol.x[0]
    run_fts.composition.chain_vol_frac[:-1] = sol.x[1:]
    run_fts.composition.chain_vol_frac[-1] = 1. - np.sum(sol.x[1:])

    # restore npw and read_input_fields
    run_fts.cell.npw = tmp_npw
    run_fts.init_fields.input_fields_file = tmp_input_fields_file
    run_fts.init_fields.read_input_fields = tmp_read_input_fields

    # return
    return sol.x


def _compute_chemical_potential_differences(x0, target_chemical_potentials, run_fts, directory):

    # set parameters
    run_fts.composition.c_chain_density = x0[0]
    run_fts.composition.chain_vol_frac[:-1] = x0[1:]
    run_fts.composition.chain_vol_frac[-1] = 1. - np.sum(x0[1:])

    # run fts
    run_fts.run(filename="params.in", directory=directory)

    # store field configuration as seed for future runs
    run_fts.init_fields.input_fields_file = "fields_k.bin"
    run_fts.init_fields.read_input_fields = True

    # get chemical potentials from operators file
    operators_path = os.path.join(directory, "operators.dat")
    od = OperatorData(operators_path)
    chemical_potentials = np.array([od.get_by_column_name('ChemicalPotential{}'.format(i+1) for i in range(len(target_chemical_potentials)))])
    return chemical_potentials - target_chemical_potentials
