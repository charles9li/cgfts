from __future__ import absolute_import, division, print_function

import os

import numpy as np

from cgfts.system import System
from cgfts.utils import split_path

__all__ = ['RunFTS']


_TAB = "  "


class RunFTS(object):

    # __slots__ = ['polyFTS_directory', 'system', 'input_file_version', 'num_models']

    def __init__(self, system):

        self.polyFTS_directory = "~/code/PolyFTS_feature_linkers/bin/Release"

        self.system = system

        # model parameters
        self.input_file_version = 3
        self.num_models = 1
        self.model_type = 'MOLECULAR'

        # model sections
        self.cell = _Cell()
        self.interactions = _Interactions(system)
        self.composition = _Composition(system)
        self.operators = _Operators()
        self.init_fields = _InitFields(system)

        # simulation
        self.simulation = _Simulation(system)

        # parallel
        self.parallel = _Parallel()

    @property
    def system(self):
        return self._system

    @system.setter
    def system(self, value):
        if isinstance(value, System):
            self._system = value
        else:
            raise TypeError("system attribute must of type 'cgfts.system.System' not {}".format(type(value).__name__))

    def create_input_file(self, filename="params.in", directory="run"):

        # file header
        s = "InputFileVersion = {} \n".format(self.input_file_version)
        s += "\n"

        # models
        s += "models { \n"
        s += _TAB + "NumModels = {} \n".format(self.num_models)
        s += _TAB + "ModelType = {} \n".format(self.model_type)
        s += "\n"

        # monomers
        s += self._system.force_field.to_PolyFTS_monomer(tab=_TAB)

        # add chains
        s += _TAB + "chains { \n"
        s += _TAB*2 + "NChains = {} \n".format(len(list(self._system.molecule_types)))
        s += _TAB*2 + "Contourds = 1 \n"
        s += _TAB*2 + "PolymerReferenceN = 1 \n"
        s += "\n"
        chain_index = 1
        for chain in self._system.molecule_types:
            s += chain.to_PolyFTS(self._system.force_field, chain_index, tab=_TAB)
            chain_index += 1
        s += _TAB + "} \n"
        s += "\n"

        # model
        s += _TAB + "model1 { \n"
        s += "\n"
        s += self.cell.to_PolyFTS(tab=_TAB)
        s += "\n"
        s += self.interactions.to_PolyFTS(tab=_TAB)
        s += "\n"
        s += self.composition.to_PolyFTS(tab=_TAB)
        s += "\n"
        s += self.operators.to_PolyFTS(tab=_TAB)
        s += "\n"
        s += self.init_fields.to_PolyFTS(tab=_TAB)
        s += _TAB + "} \n"

        # close model
        s += "} \n"
        s += "\n"

        # simulation
        s += self.simulation.to_PolyFTS(tab=_TAB)
        s += "\n"

        # parallel
        s += self.parallel.to_PolyFTS(tab=_TAB)

        # make directory if it doesn't exist
        if not os.path.isdir(directory):
            split_dir = split_path(directory)
            for i in range(len(split_dir)):
                d = os.path.join(*split_dir[0:i+1])
                if not os.path.exists(d):
                    os.mkdir(d)

        # create parameters file
        print(s, file=open(os.path.join(directory, filename), 'w'))
        return s

    def run(self, filename="params.in", directory="run"):

        # create input file
        self.create_input_file(filename=filename, directory=directory)

        # get current working directory
        cwd = os.getcwd()

        # output file directory
        run_dir = os.path.join(cwd, directory)

        # output log path
        output_log_path = os.path.join(run_dir, "output.log")

        # input file path
        input_file_path = os.path.join(run_dir, filename)

        # PolyFTS path
        polyFTS_path = os.path.join(self.polyFTS_directory, "PolyFTS.x")

        # run
        os.chdir(run_dir)
        os.system("{} {} > {}".format(polyFTS_path, input_file_path, output_log_path))
        os.chdir(cwd)

    def run_mpi(self, filename="params.in", directory="run", num_processors=1):

        if num_processors == 1:
            self.run(filename=filename, directory=directory)
        else:

            raise(NotImplementedError("run_mpi method is not implemented for multiple processors"))
        #
        #     # set number of parallel processors
        #     self.parallel.openmp_n_threads = num_processors
        #
        #     # create input file
        #     self.create_input_file(filename=filename, directory=directory)
        #
        #     # get current working directory
        #     cwd = os.getcwd()
        #
        #     # output file directory
        #     run_dir = os.path.join(cwd, directory)
        #
        #     # output log path
        #     output_log_path = os.path.join(run_dir, "output.log")
        #
        #     # input file path
        #     input_file_path = os.path.join(run_dir, filename)
        #
        #     # PolyFTS path
        #     polyFTS_path = os.path.join(self.polyFTS_directory, "PolyFTSPLL.x")
        #
        #     # run
        #     os.chdir(run_dir)
        #     os.system("{} {} > {}".format(polyFTS_path, input_file_path, output_log_path))
        #     os.chdir(cwd)


class _Cell(object):

    def __init__(self):
        self.dim = 1
        self.cell_scaling = 1.
        self.cell_lengths = 10.
        self.cell_angles = 90
        self.npw = 32
        self.space_group_name = None
        self.symmetrize = "off"

    def to_PolyFTS(self, tab="  "):
        # cell
        s = tab*2 + "cell { \n"
        s += tab*3 + "Dim = {} \n".format(self.dim)
        s += tab*3 + "CellScaling = {} \n".format(self.cell_scaling)
        # TODO: implement way to independently
        s += tab*3 + "CellLengths = {} \n".format(" ".join([str(float(self.cell_lengths))]*self.dim))
        if self.dim == 2:
            s += tab*3 + "CellAngles = {} \n".format(self.cell_angles)
        else:
            s += tab*3 + "CellAngles = {} \n".format(" ".join([str(self.cell_angles)]*self.dim))
        s += tab*3 + "NPW = {} \n".format(" ".join([str(self.npw)]*self.dim))
        if self.space_group_name is not None:
            s += "\n"
            s += tab*3 + "SpaceGroupName = {} \n".format(self.space_group_name)
        s += tab*3 + "Symmetrize = {} \n".format(self.symmetrize)
        s += tab*2 + "} \n"
        return s


class _Interactions(object):

    def __init__(self, system):
        self._system = system
        self.apply_compressibility_constraint = False

    def to_PolyFTS(self, tab="  "):
        s = tab*2 + "interactions { \n"
        bead_names = list(self._system.force_field.bead_names)
        for i, bead_name_1 in enumerate(bead_names):
            for j, bead_name_2 in enumerate(bead_names[i:]):
                gaussian = self._system.force_field.get_pair_potential("Gaussian", bead_name_1, bead_name_2)
                s += tab*3 + "BExclVolume{}{} = {} \n".format(i+1, i+j+1, gaussian.excl_vol.value)
        s += "\n"
        s += tab*3 + "ApplyCompressibilityConstraint = {} \n".format(str(self.apply_compressibility_constraint).lower())
        s += tab*2 + "} \n"
        return s


class _Composition(object):

    def __init__(self, system):
        # TODO: add way to incorporate small molecules
        self._system = system
        self.ensemble = "canonical"
        self.chain_vol_frac = None
        self.compute_chain_vol_frac()
        self.c_chain_density = 8.0

    def compute_chain_vol_frac(self):
        mol_nums = np.array(list(self._system.molecule_nums))
        mol_num_beads = np.array([c.n_beads for c in self._system.molecule_types])
        chain_vol_frac = mol_nums * mol_num_beads
        chain_vol_frac = chain_vol_frac / np.sum(chain_vol_frac)
        self.chain_vol_frac = chain_vol_frac

    def to_PolyFTS(self, tab="  "):
        s = tab*2 + "composition { \n"
        s += tab*3 + "Ensemble = {} \n".format(self.ensemble)
        s += tab*3 + "ChainVolFrac = {} \n".format(" ".join([str(cvf) for cvf in self.chain_vol_frac]))
        s += tab*3 + "CChainDensity = {} \n".format(self.c_chain_density)
        s += tab*2 + "} \n"
        return s


class _Operators(object):

    def __init__(self):
        self.calc_hamiltonian = True
        self.calc_stress_tensor = False
        self.calc_pressure = True
        self.calc_chemical_potential = True
        self.calc_structure_factor = False
        self.calc_density_operator = False
        self.include_ideal_gas_terms = True
        self.calc_orientation_correlator = False
        self.orientation_corr_spatial_average_range = 0.25

    def to_PolyFTS(self, tab="  "):
        s = tab*2 + "operators { \n"
        s += tab*3 + "CalcHamiltonian = {} \n".format(str(self.calc_hamiltonian).lower())
        s += tab*3 + "CalcStressTensor = {} \n".format(str(self.calc_stress_tensor).lower())
        s += tab*3 + "CalcPressure = {} \n".format(str(self.calc_pressure).lower())
        s += tab*3 + "CalcStructureFactor = {} \n".format(str(self.calc_structure_factor).lower())
        s += tab*3 + "CalcDensityOperator = {} \n".format(str(self.calc_density_operator).lower())
        s += tab*3 + "IncludeIdealGasOperators = {} \n".format(str(self.include_ideal_gas_terms).lower())
        s += "\n"
        s += tab*3 + "CalcOrientationCorrelator = {} \n".format(str(self.calc_orientation_correlator).lower())
        s += tab*3 + "OrientationCorr_SpatialAverageRange = {} \n".format(self.orientation_corr_spatial_average_range)
        s += tab*2 + "} \n"
        return s


class _InitFields(object):

    def __init__(self, system):
        self._system = system
        self.read_input_fields = False
        self.input_fields_file = "fields0_k.bin"
        self.init_fields = [self._InitField() for _ in range(len(list(self._system.force_field.bead_names)))]

    class _InitField(object):

        def __init__(self):
            self.init_type = "urng"
            self.parameters = None

        def to_PolyFTS(self, init_field_index, tab="  "):
            s = tab*3 + "initfield{} {{ \n".format(init_field_index)
            s += tab*4 + "inittype = {} \n".format(self.init_type)
            if self.parameters is not None:
                s += tab*4 + "parameters = {} \n".format(" ".join([str(p) for p in self.parameters]))
            s += tab*3 + "} \n"
            return s

    def to_PolyFTS(self, tab="  "):
        s = tab*2 + "initfields { \n"
        s += tab*3 + "ReadInputFields = {} \n".format(str(self.read_input_fields).lower())
        s += tab*3 + "InputFieldsFile = {} \n".format(self.input_fields_file)
        s += "\n"
        init_field_index = 1
        for init_field in self.init_fields:
            s += init_field.to_PolyFTS(init_field_index, tab=_TAB)
            init_field_index += 1
        s += tab*2 + "} \n"
        return s


class _Simulation(object):

    def __init__(self, system):
        self.job_type = "SCFT"
        self.field_updater = "ETD"
        self.time_step_dt = 0.01
        self.lambda_force_scale = [1.0] * len(list(system.force_field.bead_names))
        self.num_time_steps_per_block = 100
        self.num_blocks = 36000
        self.random_seed = 0
        self.scft_force_stopping_tol = 5e-05
        self.scft_stress_stopping_tol = 0.0001
        self.variable_cell = False
        self.cell_updater = None
        self.IO = _IO()

    def to_PolyFTS(self, tab="  "):
        s = "\n"
        s += "simulation { \n"
        s += tab + "JobType = {} \n".format(self.job_type)
        s += tab + "FieldUpdater = {} \n".format(self.field_updater)
        s += tab + "TimeStepDT = {} \n".format(self.time_step_dt)
        s += tab + "LambdaForceScale = {} \n".format(" ".join([str(lfs) for lfs in self.lambda_force_scale]))
        s += "\n"
        s += tab + "NumTimeStepsPerBlock = {} \n".format(self.num_time_steps_per_block)
        s += tab + "NumBlocks = {} \n".format(self.num_blocks)
        s += "\n"
        s += tab + "RandomSeed = {} \n".format(self.random_seed)
        s += "\n"
        s += tab + "SCFTForceStoppingTol = {} \n".format(self.scft_force_stopping_tol)
        s += tab + "SCFTStressStoppingTol = {} \n".format(self.scft_stress_stopping_tol)
        s += "\n"
        s += tab + "VariableCell = {} \n".format(str(self.variable_cell).lower())
        if self.cell_updater is not None:
            s += tab + "CellUpdater = {} \n".format(self.cell_updater)
        s += "\n"

        # IO
        s += self.IO.to_PolyFTS()

        # close simulation
        s += "}\n"
        return s


class _IO(object):

    def __init__(self):
        self.keep_density_history = False
        self.keep_field_history = False
        self.density_output_by_chain = False
        self.output_formatted_fields = False
        self.output_fields = "HFields"
        self.field_output_space = "both"

    def to_PolyFTS(self, tab="  "):
        s = tab + "IO { \n"
        s += tab*2 + "KeepDensityHistory = {} \n".format(str(self.keep_density_history).lower())
        s += tab*2 + "KeepFieldHistory = {} \n".format(str(self.keep_field_history).lower())
        s += tab*2 + "DensityOutputByChain = {} \n".format(str(self.density_output_by_chain).lower())
        s += tab*2 + "OutputFormattedFields = {} \n".format(str(self.output_formatted_fields).lower())
        s += "\n"
        s += tab*2 + "OutputFields = {} \n".format(self.output_fields)
        s += tab*2 + "FieldOutputSpace = {} \n".format(self.field_output_space)
        s += tab + "} \n"
        return s


class _Parallel(object):

    def __init__(self):
        self.cuda_select_device = 0
        self.cuda_thread_block_size = 64
        self.openmp_n_threads = 1

    def to_PolyFTS(self, tab="  "):
        s = "parallel { \n"
        s += tab + "CUDA_SelectDevice = {} \n".format(self.cuda_select_device)
        s += tab + "CUDA_ThreadBlockSize = {} \n".format(self.cuda_thread_block_size)
        s += "\n"
        s += tab + "OpenMP_nthreads = {} \n".format(self.openmp_n_threads)
        s += "}"
        return s
