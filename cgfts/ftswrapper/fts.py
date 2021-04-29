from __future__ import absolute_import, division, print_function

from collections import defaultdict, OrderedDict

from scipy.constants import R
import numpy as np

from cgfts.forcefield import ForceField
from cgfts.utils import *


class FTS(object):

    def __init__(self, temperature, one_bead=False):
        super(FTS, self).__init__()

        self._one_bead = one_bead

        # forcefield and system
        self._force_field = ForceField(temperature)
        self._mol_num_dict = OrderedDict()
        self._mol_vol_dict = OrderedDict()
        self._mol_num_bead_dict = OrderedDict()
        self._num_dodecane_1bead = 0
        self._num_dodecane_2bead = 0
        self._num_dodecane_3bead = 0

        # fts system parameters
        self._input_file_version = 3
        self._num_models = 1
        self._model_type = "MOLECULAR"

        # cell parameters
        self._dim = 3
        self._cell_scaling = 1
        self._cell_lengths = 10
        self._cell_angles = 90
        self._npw = 32

        # operators
        self._calc_hamiltonian = True
        self._calc_stress_tensor = False
        self._calc_pressure = True
        self._calc_chemical_potential = True
        self._calc_structure_factor = False
        self._calc_density_operator = False
        self._include_ideal_gas_terms = True
        self._calc_orientation_correlator = False
        self._orientation_corr_spatial_average_range = 0.25

        # initfields
        self._read_input_fields = "false"
        self._input_fields_file = "fields0_k.bin"
        self._init_field_type = defaultdict(lambda: "urng")
        self._init_field_parameters = defaultdict(lambda: None)

        # simulation
        self._job_type = "SCFT"
        self._field_updater = "ETD"
        self._time_step_dt = 0.02
        self._lambda_force_scale = defaultdict(lambda: 1.0)
        self._num_time_steps_per_block = 1000
        self._num_blocks = 36000
        self._random_seed = 0
        self._scft_force_stopping_tol = 5e-5
        self._scft_stress_stopping_tol = 1e-4
        self._variable_cell = False

        # io
        self._keep_density_history = False
        self._keep_field_history = False
        self._density_output_by_chain = False
        self._output_formatted_fields = False
        self._output_fields = "HFields"
        self._field_output_space = "both"

        # parallel
        self._cuda_select_device = 0
        self._cuda_thread_block_size = 64
        self._openmp_nthreads = 6

        # file formatting
        self._tab = "  "

    @classmethod
    def from_file(cls, temperature, ff_file, one_bead=False):
        fts = cls(temperature, one_bead=one_bead)
        fts._force_field = ForceField.from_file(temperature, ff_file)
        return fts

    @property
    def force_field(self):
        return self._force_field

    @property
    def read_input_fields(self):
        return self._read_input_fields

    @read_input_fields.setter
    def read_input_fields(self, value):
        self._read_input_fields = value

    @property
    def input_fields_file(self):
        return self._input_fields_file

    @input_fields_file.setter
    def input_fields_file(self, value):
        self._input_fields_file = value

    def set_init_field(self, bead_name, init_type, init_parameters=None):
        self._init_field_type[bead_name] = init_type
        self._init_field_parameters[bead_name] = init_parameters

    @property
    def time_step_dt(self):
        return self._time_step_dt

    @time_step_dt.setter
    def time_step_dt(self, value):
        self._time_step_dt = value

    def set_lambda_force_scale(self, bead_name, lambda_force_scale=1.0):
        self._lambda_force_scale[bead_name] = lambda_force_scale

    @property
    def num_time_steps_per_block(self):
        return self._num_time_steps_per_block

    @num_time_steps_per_block.setter
    def num_time_steps_per_block(self, value):
        self._num_time_steps_per_block = value

    @property
    def num_blocks(self):
        return self._num_blocks

    @num_blocks.setter
    def num_blocks(self, value):
        self._num_blocks = value

    def add_polyacrylate(self, sequence, num_mol=1):
        self._mol_num_dict[sequence] = num_mol

    def add_dodecane_1bead(self, num_mol=1):
        self._num_dodecane_1bead = num_mol

    def add_dodecane_2bead(self, num_mol=1):
        self._num_dodecane_2bead = num_mol

    def add_dodecane_3bead(self, num_mol=1):
        self._num_dodecane_3bead = num_mol

    def create_input_file(self, filepath="params.in"):

        # file header
        s = "InputFileVersion = {}".format(self._input_file_version)
        s += "\n"
        s += "\n"

        # add sections
        s += self.models_string()
        s += self.simulation_string()
        s += self.parallel_string()

        print(s, file=open(filepath, 'w'))

    def models_string(self):

        self._mol_vol_dict = OrderedDict()
        self._mol_num_bead_dict = OrderedDict()

        # section header
        s = "models {"
        s += "\n"
        s += self._tab + "NumModels = 1"
        s += "\n"
        s += self._tab + "ModelType = {}".format(self._model_type)
        s += "\n"
        s += "\n"

        # bead names and smear lengths
        bead_types = self._force_field.bead_types[:]
        bead_names = [b.name for b in self._force_field.bead_types]
        smear_lengths = [b.smear_length for b in self._force_field.bead_types]
        if self._num_dodecane_2bead == 0:
            if 'D6' in bead_names:
                D6_index = bead_names.index('D6')
                bead_types.pop(D6_index)
                bead_names.pop(D6_index)
                smear_lengths.pop(D6_index)

        # create monomers
        # TODO: find way to remove D6 bead when dodecane not present
        kuhn_lengths = self._determine_kuhn_lengths(bead_types)
        s += self._tab + "monomers {{ # {}".format(" ".join(bead_names))
        s += "\n"
        s += self._tab*2 + "NSpecies = {}".format(len(bead_names))
        s += "\n"
        s += self._tab*2 + "KuhnLen = {}".format(" ".join([str(k) for k in kuhn_lengths]))
        s += "\n"
        s += self._tab*2 + "Charge = " + "0. "*len(bead_names)
        s += "\n"
        s += self._tab*2 + "GaussSmearWidth = {}".format(" ".join([str(s) for s in smear_lengths]))
        s += "\n"
        s += self._tab + "}"
        s += "\n"
        s += "\n"

        # create chains
        s += self._tab + "chains {"
        s += "\n"
        n_chains = len(self._mol_num_dict.keys())
        if self._num_dodecane_2bead > 0:
            n_chains += 1
        s += self._tab*2 + "NChains = {}".format(n_chains)
        s += "\n"
        s += self._tab*2 + "Contourds = 1"
        s += "\n"
        s += self._tab*2 + "PolymerReferenceN = 1"
        s += "\n"

        # add polyacrylate chains
        chain_index = 1
        for sequence in self._mol_num_dict.keys():
            if self._one_bead:
                s += self._create_polyacrylate_chain_1bead(sequence, chain_index, bead_names)
            else:
                s += self._create_polyacrylate_chain(sequence, chain_index, bead_names)
            chain_index += 1
            
        # add 2-bead dodecane
        if self._num_dodecane_2bead > 0:
            s += self._create_dodecane_2bead(chain_index, bead_names)
            chain_index += 1

        # add 3-bead dodecane
        if self._num_dodecane_3bead > 0:
            s += self._create_dodecane_3bead(chain_index, bead_names)
            chain_index += 1

        s += self._tab + "}"
        s += "\n"
        s += "\n"

        # add 1-bead dodecane
        if self._num_dodecane_1bead > 0:
            s += self._tab + "smallmolecules {"
            s += "\n"
            s += self._tab*2 + "PolymerReferenceN = 1"
            s += "\n"
            s += self._tab*2 + "NSmallMoleculeTypes = 1"
            s += "\n"
            s += "\n"
            s += self._tab*2 + "smallmolecule1 {"
            s += "\n"
            s += self._tab*3 + "Species = {}".format(bead_names.index('D12') + 1)
            s += "\n"
            s += self._tab*2 + "}"
            s += "\n"
            s += self._tab + "}"
            s += "\n"
            s += "\n"
            self._mol_vol_dict['D12'] = self._force_field.get_bead_volume('D12')
            self._mol_num_bead_dict['D12'] = 1
            self._mol_num_dict['D12'] = self._num_dodecane_1bead

        # create model
        s += self._tab + "model1 {"
        s += "\n"
        s += "\n"

        # cell
        s += self._tab*2 + "cell {"
        s += "\n"
        s += self._tab*3 + "Dim = {}".format(self._dim)
        s += "\n"
        s += self._tab*3 + "CellScaling = {}".format(self._cell_scaling)
        s += "\n"
        # TODO: implement way to independently
        s += self._tab*3 + "CellLengths = {}".format(" ".join([str(float(self._cell_lengths))]*self._dim))
        s += "\n"
        s += self._tab*3 + "CellAngles = {}".format(" ".join([str(self._cell_angles)]*self._dim))
        s += "\n"
        s += self._tab*3 + "NPW = {}".format(" ".join([str(self._npw)]*self._dim))
        s += "\n"
        s += self._tab*2 + "}"
        s += "\n"

        # interactions
        s += "\n"
        s += self._tab*2 + "interactions {"
        s += "\n"
        for i, bead_type_1 in enumerate(bead_types):
            for j, bead_type_2 in enumerate(bead_types[i:]):
                gaussian = self._force_field.get_gaussian_potential(bead_type_1.name, bead_type_2.name)
                B = gaussian.B * (np.pi / gaussian.Kappa)**(3./2.) / (R * 1.e-3 * self._force_field.temperature)
                s += self._tab*3 + "BExclVolume{}{} = {}".format(i+1, i+j+1, B)
                s += "\n"
        s += "\n"
        s += self._tab*3 + "ApplyCompressibilityConstraint = false"
        s += "\n"
        s += self._tab*2 + "}"
        s += "\n"

        # compositions
        s += "\n"
        s += self._tab*2 + "composition {"
        s += "\n"
        s += self._tab*3 + "Ensemble = canonical"
        s += "\n"
        chain_vol_frac = np.array(self._mol_num_dict.values()) * np.array(self._mol_num_bead_dict.values())
        chain_vol_frac /= np.sum(chain_vol_frac)
        s += self._tab*3 + "ChainVolFrac = {}".format(" ".join([str(c) for c in chain_vol_frac]))
        s += "\n"
        c_chain_density = np.sum(np.array(self._mol_num_dict.values()) * np.array(self._mol_num_bead_dict.values()))
        c_chain_density /= np.sum(np.array(self._mol_num_dict.values()) * np.array(self._mol_vol_dict.values()))
        s += self._tab*3 + "CChainDensity = {}".format(c_chain_density)
        s += "\n"
        s += self._tab*2 + "}"
        s += "\n"

        # operators
        s += "\n"
        s += self._tab*2 + "operators {"
        s += "\n"
        s += self._tab*3 + "CalcHamiltonian = {}".format(str(self._calc_hamiltonian).lower())
        s += "\n"
        s += self._tab*3 + "CalcStressTensor = {}".format(str(self._calc_stress_tensor).lower())
        s += "\n"
        s += self._tab*3 + "CalcPressure = {}".format(str(self._calc_pressure).lower())
        s += "\n"
        s += self._tab*3 + "CalcStructureFactor = {}".format(str(self._calc_structure_factor).lower())
        s += "\n"
        s += self._tab*3 + "CalcDensityOperator = {}".format(str(self._calc_density_operator).lower())
        s += "\n"
        s += self._tab*3 + "IncludeIdealGasOperators = {}".format(str(self._include_ideal_gas_terms).lower())
        s += "\n"
        s += "\n"
        s += self._tab*3 + "CalcOrientationCorrelator = {}".format(str(self._calc_orientation_correlator).lower())
        s += "\n"
        s += self._tab*3 + "OrientationCorr_SpatialAverageRange = {}".format(self._orientation_corr_spatial_average_range)
        s += "\n"
        s += self._tab*2 + "}"
        s += "\n"

        # initfields
        s += "\n"
        s += self._tab*2 + "initfields {"
        s += "\n"
        s += self._tab*3 + "ReadInputFields = {}".format(self._read_input_fields)
        s += "\n"
        s += self._tab*3 + "InputFieldsFile = {}".format(self._input_fields_file)
        s += "\n"
        s += "\n"
        for i, bead_type in enumerate(bead_types):
            bead_name = bead_type.name
            s += self._tab*3 + "initfield{} {{".format(i+1)
            s += "\n"
            s += self._tab*4 + "inittype = {}".format(self._init_field_type[bead_name])
            s += "\n"
            init_parameters = self._init_field_parameters[bead_name]
            if init_parameters is not None:
                s += self._tab*4 + "initparameters = {}".format(init_parameters)
                s += "\n"
            s += self._tab*3 + "}"
            s += "\n"
        s += self._tab*2 + "}"
        s += "\n"

        # close model1
        s += self._tab + "}"
        s += "\n"

        # close models
        s += "}"
        s += "\n"

        return s

    def simulation_string(self):

        bead_types = self._force_field.bead_types
        bead_names = [b.name for b in bead_types]
        if self._num_dodecane_2bead == 0:
            if 'D6' in bead_names:
                D6_index = bead_names.index('D6')
                bead_types.pop(D6_index)
                bead_names.pop(D6_index)

        s = "\n"
        s += "simulation {"
        s += "\n"
        s += self._tab + "JobType = {}".format(self._job_type)
        s += "\n"
        s += self._tab + "FieldUpdater = {}".format(self._field_updater)
        s += "\n"
        s += self._tab + "TimeStepDT  = {}".format(self._time_step_dt)
        s += "\n"
        s += self._tab + "LambdaForceScale = {}".format(" ".join([str(self._lambda_force_scale[b]) for b in bead_names]))
        s += "\n"
        s += "\n"
        s += self._tab + "NumTimeStepsPerBlock = {}".format(self._num_time_steps_per_block)
        s += "\n"
        s += self._tab + "NumBlocks = {}".format(self._num_blocks)
        s += "\n"
        s += "\n"
        s += self._tab + "RandomSeed = {}".format(self._random_seed)
        s += "\n"
        s += "\n"
        s += self._tab + "SCFTForceStoppingTol = {}".format(self._scft_force_stopping_tol)
        s += "\n"
        s += self._tab + "SCFTStressStoppingTol = {}".format(self._scft_stress_stopping_tol)
        s += "\n"
        s += "\n"
        s += self._tab + "VariableCell = {}".format(str(self._variable_cell).lower())
        s += "\n"

        # IO
        s += "\n"
        s += self._tab + "IO {"
        s += "\n"
        s += self._tab*2 + "KeepDensityHistory = {}".format(str(self._keep_density_history).lower())
        s += "\n"
        s += self._tab*2 + "KeepFieldHistory = {}".format(str(self._keep_field_history).lower())
        s += "\n"
        s += self._tab*2 + "DensityOutputByChain = {}".format(str(self._density_output_by_chain).lower())
        s += "\n"
        s += self._tab*2 + "OutputFormattedFields = {}".format(str(self._output_formatted_fields).lower())
        s += "\n"
        s += "\n"
        s += self._tab*2 + "OutputFields = {}".format(self._output_fields)
        s += "\n"
        s += self._tab*2 + "FieldOutputSpace = {}".format(self._field_output_space)
        s += "\n"
        s += self._tab + "}"
        s += "\n"

        # close simulation
        s += "}"
        s += "\n"
        return s

    def parallel_string(self):
        s = "\n"
        s += "parallel {"
        s += "\n"
        s += self._tab + "CUDA_SelectDevice = {}".format(self._cuda_select_device)
        s += "\n"
        s += self._tab + "CUDA_ThreadBlockSize = {}".format(self._cuda_thread_block_size)
        s += "\n"
        s += "\n"
        s += self._tab + "OpenMP_nthreads = {}".format(self._openmp_nthreads)
        s += "\n"
        s += "}"
        return s

    def _create_polyacrylate_chain_1bead(self, sequence, index, bead_name_list):

        # initialize chain vol and bead count
        chain_volume = 0.0
        num_beads = 0

        # determine blocks
        monomer_list = acrylate_sequence_to_list(sequence)
        block_species = []
        n_per_block = []
        backbone_grafting_positions = defaultdict(list)
        curr_monomer = None
        curr_block_length = 0
        for i, monomer in enumerate(monomer_list):

            # add to volume and bead count
            chain_volume += self._force_field.get_bead_volume(monomer)
            num_beads += 1

            # add backbone graft position for side chain
            backbone_grafting_positions[monomer].append(i)

            # add to current block or start new one
            if curr_monomer is None:
                curr_monomer = monomer
            if monomer != curr_monomer:
                block_species.append(bead_name_list.index(curr_monomer) + 1)
                n_per_block.append(curr_block_length)
                curr_monomer = monomer
                curr_block_length = 0
            curr_block_length += 1
        block_species.append(bead_name_list.index(curr_monomer) + 1)
        n_per_block.append(curr_block_length)

        # chain settings
        s = "\n"
        s += self._tab*2 + "chain{} {{".format(index)
        s += "\n"
        s += self._tab*3 + "Label = {}".format(sequence)
        s += "\n"
        s += self._tab*3 + "Architecture = linear"
        s += "\n"
        s += self._tab*3 + "Statistics = FJC"
        s += "\n"
        s += "\n"

        # backbone
        s += self._tab*3 + "NBlocks = {}".format(len(block_species))
        s += "\n"
        s += self._tab*3 + "BlockSpecies = {}".format(" ".join([str(b) for b in block_species]))
        s += "\n"
        s += self._tab*3 + "NBeads = {}".format(np.sum(n_per_block))
        s += "\n"
        s += self._tab*3 + "NPerBlock = {}".format(" ".join([str(n) for n in n_per_block]))
        s += "\n"

        s += self._tab*2 + "}"
        s += "\n"

        # add volume and bead count to dictionaries
        self._mol_vol_dict[sequence] = chain_volume
        self._mol_num_bead_dict[sequence] = num_beads

        return s

    _MONOMER_TO_BEAD_NAME = {'A4': ['Bpba', 'C4'],
                             'A12': ['Bpla', 'C6', 'E6'],
                             'mA12': ['Bplma', 'C6', 'E6']}

    _MONOMER_TO_BEAD_NAME_V2 = {'A4': ['Bpba', 'D4'],
                                'A12': ['Bpla', 'D4', 'D4', 'D4'],
                                'mA12': ['Bplma', 'D4', 'D4', 'D4']}

    def _create_polyacrylate_chain(self, sequence, index, bead_name_list, version=1):

        # initialize chain vol and bead count
        chain_volume = 0.0
        num_beads = 0

        # determine monomer to bead name
        if version == 1:
            monomer_to_bead_name = self._MONOMER_TO_BEAD_NAME
        elif version == 2:
            monomer_to_bead_name = self._MONOMER_TO_BEAD_NAME_V2
        else:
            raise ValueError("Invalid version.")

        # determine blocks
        monomer_list = acrylate_sequence_to_list(sequence)
        block_species = []
        n_per_block = []
        backbone_grafting_positions = defaultdict(list)
        curr_monomer = None
        curr_block_length = 0
        for i, monomer in enumerate(monomer_list):

            # add to volume and bead count
            chain_volume += np.sum([self._force_field.get_bead_volume(b) for b in monomer_to_bead_name[monomer]])
            num_beads += len(monomer_to_bead_name[monomer])

            # add backbone graft position for side chain
            backbone_grafting_positions[monomer].append(i)

            # add to current block or start new one
            if curr_monomer is None:
                curr_monomer = monomer
            if monomer != curr_monomer:
                block_species.append(bead_name_list.index(monomer_to_bead_name[curr_monomer][0]) + 1)
                n_per_block.append(curr_block_length)
                curr_monomer = monomer
                curr_block_length = 0
            curr_block_length += 1
        block_species.append(bead_name_list.index(monomer_to_bead_name[curr_monomer][0]) + 1)
        n_per_block.append(curr_block_length)

        # chain settings
        s = "\n"
        s += self._tab*2 + "chain{} {{".format(index)
        s += "\n"
        s += self._tab*3 + "Label = {}".format(sequence)
        s += "\n"
        s += self._tab*3 + "Architecture = comb"
        s += "\n"
        s += self._tab*3 + "Statistics = FJC"
        s += "\n"
        s += self._tab*3 + "NumSideArmTypes = {}".format(len(backbone_grafting_positions.keys()))
        s += "\n"
        s += "\n"

        # backbone
        s += self._tab*3 + "backbone {"
        s += "\n"
        s += self._tab*4 + "Statistics = FJC"
        s += "\n"
        s += self._tab*4 + "NBlocks = {}".format(len(block_species))
        s += "\n"
        s += self._tab*4 + "BlockSpecies = {}".format(" ".join([str(b) for b in block_species]))
        s += "\n"
        s += self._tab*4 + "NBeads = {}".format(np.sum(n_per_block))
        s += "\n"
        s += self._tab*4 + "NPerBlock = {}".format(" ".join([str(n) for n in n_per_block]))
        s += "\n"
        s += self._tab*3 + "}"
        s += "\n"

        # side arms
        side_arm_index = 1
        for monomer, grafting_positions in backbone_grafting_positions.items():
            side_arm_bead_names = monomer_to_bead_name[monomer][1:]
            side_arm_block_species = [bead_name_list.index(b) + 1 for b in side_arm_bead_names]
            s += "\n"
            s += self._tab*3 + "sidearmtype{} {{".format(side_arm_index)
            s += "\n"
            s += self._tab*4 + "Statistics = FJC"
            s += "\n"
            s += self._tab*4 + "NBeads = {}".format(len(side_arm_block_species))
            s += "\n"
            s += self._tab*4 + "NBlocks = {}".format(len(side_arm_block_species))
            s += "\n"
            s += self._tab*4 + "BlockSpecies = {}".format(" ".join([str(b) for b in side_arm_block_species]))
            s += "\n"
            s += self._tab*4 + "NPerBlock =" + " 1"*len(side_arm_block_species)
            s += "\n"
            s += self._tab*4 + "NumArms = {}".format(len(grafting_positions))
            s += "\n"
            s += self._tab*4 + "BackboneGraftingPositions = {}".format(" ".join([str(g) for g in grafting_positions]))
            s += "\n"
            s += self._tab*3 + "}"
            s += "\n"
            side_arm_index += 1

        s += self._tab*2 + "}"
        s += "\n"

        # add volume and bead count to dictionaries
        self._mol_vol_dict[sequence] = chain_volume
        self._mol_num_bead_dict[sequence] = num_beads

        return s

    def _create_dodecane_2bead(self, index, bead_name_list):

        # create string
        s = "\n"
        s += self._tab*2 + "chain{} {{".format(index)
        s += "\n"
        s += self._tab*3 + "Label = dodecane"
        s += "\n"
        s += self._tab*3 + "Architecture = linear"
        s += "\n"
        s += self._tab*3 + "Statistics = FJC"
        s += "\n"
        s += "\n"
        s += self._tab*3 + "Nbeads = 2"
        s += "\n"
        s += self._tab*3 + "NBlocks = 1"
        s += "\n"
        s += self._tab*3 + "BlockSpecies = {}".format(bead_name_list.index('D6') + 1)
        s += "\n"
        s += self._tab*3 + "NPerBlock = 2"
        s += "\n"
        s += self._tab*2 + "}"
        s += "\n"

        # add vol and bead count
        self._mol_num_dict["dodecane"] = self._num_dodecane_2bead
        self._mol_vol_dict["dodecane"] = 2*self._force_field.get_bead_volume('D6')
        self._mol_num_bead_dict["dodecane"] = 2

        return s

    def _create_dodecane_3bead(self, index, bead_name_list):

        # create string
        s = "\n"
        s += self._tab*2 + "chain{} {{".format(index)
        s += "\n"
        s += self._tab*3 + "Label = dodecane"
        s += "\n"
        s += self._tab*3 + "Architecture = linear"
        s += "\n"
        s += self._tab*3 + "Statistics = FJC"
        s += "\n"
        s += "\n"
        s += self._tab*3 + "Nbeads = 3"
        s += "\n"
        s += self._tab*3 + "NBlocks = 1"
        s += "\n"
        s += self._tab*3 + "BlockSpecies = {}".format(bead_name_list.index('D4') + 1)
        s += "\n"
        s += self._tab*3 + "NPerBlock = 3"
        s += "\n"
        s += self._tab*2 + "}"
        s += "\n"

        # add vol and bead count
        self._mol_num_dict["dodecane"] = self._num_dodecane_2bead
        self._mol_vol_dict["dodecane"] = 3*self._force_field.get_bead_volume('D4')
        self._mol_num_bead_dict["dodecane"] = 3

        return s

    def _determine_kuhn_lengths(self, bead_types):

        # initialize array
        kuhn_lengths = np.zeros(len(bead_types))

        # determine kuhn lengths from like-like bonded interactions if possible
        like_bead_names = []
        for i, bead_type in enumerate(bead_types):
            try:
                p = self._force_field.get_bonded_potential(bead_type.name, bead_type.name)
                kuhn_lengths[i] = p.Dist0 / np.sqrt(6)
                like_bead_names.append(bead_type.name)
            except ValueError:
                pass

        # determine kuhn lengths for other beads
        for i, bead_type in enumerate(bead_types):
            if kuhn_lengths[i] == 0:
                for p in self._force_field.bonded_potentials:
                    if bead_type.name in [p.bead_name_1, p.bead_name_2]:
                        if bead_type.name == p.bead_name_1:
                            other_bead_name = p.bead_name_2
                        else:
                            other_bead_name = p.bead_name_1
                        if other_bead_name not in like_bead_names:
                            kuhn_lengths[i] = p.Dist0 / np.sqrt(6)

        # catch any stragglers
        for i, bead_type in enumerate(bead_types):
            if kuhn_lengths[i] == 0:
                for p in self._force_field.bonded_potentials:
                    if bead_type.name in [p.bead_name_1, p.bead_name_2]:
                        kuhn_lengths[i] = p.Dist0 / np.sqrt(6)

        return kuhn_lengths


if __name__ == '__main__':
    fts = FTS.from_file(313.15, "25A4_25A12_rand1_NPT_313K_3866bar_10wt_ff.dat")
    fts.force_field.reorder_bead_types(['Bpba', 'C4', 'Bpla', 'C6', 'E6', 'D6'])
    m_poly = 5*128.17 + 5*240.38
    m_dod = 170.33
    n_poly = 1.0
    n_dod = .9/.1 * m_poly/m_dod * n_poly
    print(n_poly, n_dod)
    fts.add_polyacrylate("5*A4+5*A12", num_mol=n_poly)
    # fts.add_dodecane_2bead(num_mol=n_dod)
    print(fts.models_string())
    print(fts.simulation_string())
