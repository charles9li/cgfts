from collections import OrderedDict
import os

from scipy.constants import N_A, R
import mdtraj as md
import numpy as np

import sim


def map_topology(top_list, dcd_list_list, residue_map, temperature, pressure, potential_fix={}, cut=4.0, stride=1, include_hydrogen=False):

    ForceFieldFile = "ff.dat"

    # Default Simulation Package Settings
    sim.export.lammps.NeighOne = 8000
    sim.export.lammps.UseTable2 = True
    sim.export.lammps.InnerCutoff = 1.e-6
    sim.export.lammps.NPairPotentialBins = 1000
    sim.export.lammps.LammpsExec = 'lmp_omp'
    sim.export.lammps.UseLangevin = True
    sim.export.lammps.OMP_NumThread = 8
    sim.export.lammps.TableInterpolationStyle = 'spline'    # More robust than spline for highly CG-ed systems
    sim.srel.optimizetrajlammps.LammpsDelTempFiles = False
    sim.srel.optimizetrajlammps.UseLangevin = True
    sim.export.omm.platformName = 'CUDA'                    # or 'OpenCL' or 'GPU' or 'CUDA'
    sim.export.omm.device = -1                              # -1 is default, let openmm choose its own platform.
    sim.export.omm.NPairPotentialKnots = 500                # number of points used to spline-interpolate the potential
    sim.export.omm.InnerCutoff = 0.001                      # 0.001 is default. Note that a small value is not necessary, like in the lammps export, because the omm export interpolates to zero
    sim.srel.optimizetrajomm.OpenMMStepsMin = 0 #number of steps to minimize structure, 0 is default
    sim.srel.optimizetrajomm.OpenMMDelTempFiles = False #False is Default
    sim.export.omm.UseTabulated = True

    CG_atom_type_dict = {}
    Opts = []
    ForceField = None
    system_index = 0
    for top, dcd_list in zip(top_list, dcd_list_list):

        # load topology
        traj_list = []
        for dcd in dcd_list:
            traj = md.load_dcd(dcd, top, stride=stride)
            traj_list.append(traj)
        traj_union = md.join(traj_list)
        traj_union.xyz = traj_union.xyz / 10.
        traj_union.unitcell_lengths = traj_union.unitcell_lengths / 10.
        topology = traj_union.topology

        # create map
        Map = sim.atommap.PosMap()

        # map CG bead indices to
        CG_bead_index = 0
        AA_to_CG = {}
        CG_to_AA = {}
        CG_to_chain = {}
        chain_to_CG = {}
        CGAtomNameList = []
        MolTypeList = []
        MolTypeDict = OrderedDict()
        MolNumDict = OrderedDict()
        for chain in topology.chains:

            bead_type_list = []
            bead_index_list = []

            for residue in chain.residues:

                # get corresponding data from residue_map
                try:
                    bead_types, num_atoms_per_bead, masses_residue = residue_map[residue.name]
                except KeyError:
                    raise ValueError("Cannot find {} residue in residue_map provided.".format(residue.name))

                # check that bead_types and num_atoms_per_bead are the same length
                if len(bead_types) != len(num_atoms_per_bead):
                    raise ValueError("Bead type list and atoms per bead list "
                                     "don't have same length for {} residue".format(residue.name))

                # get indices of heavy atoms in residue
                heavy_atom_indices = topology.select('resid {} and element != H'.format(residue.index))

                # check that sum of num_atoms_per_bead equals the number of heavy atoms
                if np.sum(num_atoms_per_bead) != len(heavy_atom_indices):
                    raise ValueError("Sum of atoms per bead list does not match the "
                                     "number of heavy atoms in residue {}.".format(residue.name))

                # check that sum of num_atoms_per_bead equals length of mass of heavy atoms
                if np.sum(num_atoms_per_bead) != len(masses_residue):
                    raise ValueError("Sum of atoms per bead list does not match the "
                                     "number of heavy atom masses in residue {}.".format(residue.name))

                # create maps
                indices_split = np.split(heavy_atom_indices, np.cumsum(num_atoms_per_bead[:-1]))
                masses_split = np.split(masses_residue, np.cumsum(num_atoms_per_bead[:-1]))
                for bead_type, indices, masses in zip(bead_types, indices_split, masses_split):
                    AtomMap = sim.atommap.AtomMap(list(indices), CG_bead_index, Mass1=masses, Atom2Name=bead_type)
                    Map.append(AtomMap)
                    if bead_type not in CG_atom_type_dict.keys():
                        CG_atom_type_dict[bead_type] = sim.chem.AtomType(bead_type)
                    for i in indices:
                        AA_to_CG[i] = CG_bead_index
                    CG_to_AA[CG_bead_index] = indices
                    CG_to_chain[CG_bead_index] = chain.index
                    bead_index_list.append(CG_bead_index)
                    bead_type_list.append(CG_atom_type_dict[bead_type])
                    CGAtomNameList.append(bead_type)
                    CG_bead_index += 1

            chain_to_CG[chain.index] = bead_index_list

            MolType = sim.chem.MolType(str(chain.index), bead_type_list)
            if tuple(bead_type_list) not in MolTypeDict.keys():
                MolTypeDict[tuple(bead_type_list)] = MolType
                MolNumDict[tuple(bead_type_list)] = 1
            else:
                MolNumDict[tuple(bead_type_list)] += 1
            MolTypeList.append(MolTypeDict[tuple(bead_type_list)])

        # create mapped trajectory
        print("start mapping trajectory")
        filename = "{}.lammpstrj".format("system")
        traj_union.save_lammpstrj(filename)
        Trj = sim.traj.Lammps(filename)
        BoxL = traj_union.unitcell_lengths[0][0]
        MappedTrj = sim.traj.Mapped(Trj, Map, AtomNames=CGAtomNameList, BoxL=BoxL)
        print("finish mapping trajectory")

        # add bonds to MolTypes
        print("start adding bonds")
        BondTypes = []
        for bond in topology.bonds:
            AA_index_1 = bond[0].index
            AA_index_2 = bond[1].index
            CG_index_1 = AA_to_CG[AA_index_1]
            CG_index_2 = AA_to_CG[AA_index_2]
            if CG_index_1 != CG_index_2:
                chain_index = CG_to_chain[CG_index_1]
                MolType = MolTypeList[chain_index]
                bead_index_list = chain_to_CG[chain_index]
                try:
                    MolType.Bond(bead_index_list.index(CG_index_1), bead_index_list.index(CG_index_2))
                except ValueError:
                    pass
                AtomType1 = CGAtomNameList[CG_index_1]
                AtomType2 = CGAtomNameList[CG_index_2]
                BondPair = tuple(np.sort([AtomType1, AtomType2]))
                if BondPair not in BondTypes:
                    BondTypes.append(BondPair)
        print(BondTypes)
        print("finish adding bonds")

        # define world
        print("creating world")
        World = sim.chem.World(MolTypeDict.values(), Dim=3, Units=sim.units.DimensionlessUnits)

        # create system
        print("creating system")
        Sys = sim.system.System(World, Name="system")

        # add molecules to system
        print("adding molecules to system")
        for i, MolType in zip(MolNumDict.values(), MolTypeDict.values()):
            for _ in range(i):
                Sys += MolType.New()

        # Set system box length
        print("seting system box length")
        Sys.BoxL = BoxL

        # Set system temperature and charges
        Temp = temperature * R / 1000.0
        Sys.TempSet = Temp
        Sys.PresSet = pressure * 1e5 * 1.0e-27 / 1000.0 * N_A
        Sys.ForceField.Globals.Charge.Fixed = True

        # initialize force field
        ForceField = []

        # create Gaussian potentials
        print("creating Gaussian potentials")
        BeadNameList = CG_atom_type_dict.keys()
        for i, BeadName1 in enumerate(BeadNameList):
            for BeadName2 in BeadNameList[i:]:
                BeadName1_sorted, BeadName2_sorted = np.sort([BeadName1, BeadName2])
                Label = "Gaussian_{}{}".format(BeadName1_sorted, BeadName2_sorted)
                BeadType1 = CG_atom_type_dict[BeadName1_sorted]
                BeadType2 = CG_atom_type_dict[BeadName2_sorted]
                Filter = sim.atomselect.PolyFilter([BeadType1, BeadType2])
                Gaussian = sim.potential.LJGaussian(Sys, Cut=cut, Filter=Filter,
                                                    B=1.0, Kappa=1.0, Dist0=0.0, Sigma=1.0, Epsilon=0.0,
                                                    Label=Label)
                if 'B' in potential_fix.keys() and Label in potential_fix['B']:
                    Gaussian.Param.B.Fixed = True
                Gaussian.Param.Kappa.Fixed = True
                Gaussian.Param.Dist0.Fixed = True
                Gaussian.Param.Sigma.Fixed = True
                Gaussian.Param.Epsilon.Fixed = True
                ForceField.append(Gaussian)

        # create Bonded potentials
        print("creating bonded potentials")
        for BondType in BondTypes:
            BeadName1, BeadName2 = BondType
            Label = "Bonded_{}{}".format(BeadName1, BeadName2)
            BeadType1 = CG_atom_type_dict[BeadName1]
            BeadType2 = CG_atom_type_dict[BeadName2]
            Filter = sim.atomselect.PolyFilter([BeadType1, BeadType2], Bonded=True)
            Bonded = sim.potential.Bond(Sys, Filter=Filter,
                                        Dist0=1.0, FConst=1.0,
                                        Label=Label)
            if 'Dist0' in potential_fix.keys() and (potential_fix['Dist0'] == 'all' or Label in potential_fix['Dist0']):
                Bonded.Param.Dist0.Fixed = True
            if 'FConst' in potential_fix.keys() and (potential_fix['FConst'] == 'all' or Label in potential_fix['FConst']):
                Bonded.Param.FConst.Fixed = True
            ForceField.append(Bonded)

        # add potentials to forcefield
        Sys.ForceField.extend(ForceField)

        # set up the histograms
        for P in Sys.ForceField:
            P.Arg.SetupHist(NBin=10000, ReportNBin=100)

        Sys.World.GetBondOrdMatrix(ShowRigid=True)

        # lock and load
        print("loading system")
        Sys.Load()

        # initial positions and velocities
        print("initializing positions and velocities")
        sim.system.positions.CubicLattice(Sys)
        sim.system.velocities.Canonical(Sys, Temp=Temp)

        # configure integrator
        print("configuring integrator")
        Int = Sys.Int
        Int.Method = Int.Methods.VVIntegrate
        Int.Method.TimeStep = 0.005
        Int.Method.Thermostat = Int.Method.ThermostatLangevin
        Int.Method.LangevinGamma = 1.0
        Int.Method.Barostat = Int.Method.BarostatMonteCarlo

        # import parameters from file
        print("loading forcefield file")
        Sys.ForceField.SetParamString(open(ForceFieldFile, 'r').read())

        # run Srel
        print("creating optimizer")
        Map = sim.atommap.PosMap()
        for i, a in enumerate(Sys.Atom):
            Map.append(sim.atommap.AtomMap(Atoms1=i, Atom2=a))
        Opt = sim.srel.optimizetrajomm.OptimizeTrajOpenMMClass(Sys, Map,
                                                               Traj=MappedTrj, FilePrefix="{}{}".format(Sys.Name, system_index),
                                                               TempFileDir=os.getcwd(),
                                                               UseTarHists=False)
        Opt.ConstrainNeutralCharge()
        Opt.StepsEquil = 50000
        Opt.StepsProd = 500000
        Opt.StepsStride = 500
        Opts.append(Opt)
        system_index += 1

    Weights = [1.]*len(Opts)
    Optimizer = sim.srel.OptimizeMultiTrajClass(Opts, Weights=Weights)
    Optimizer.FilePrefix = ("system")
    print("running optimizer")
    Optimizer.RunConjugateGradient(MaxIter=None, SteepestIter=0)


if __name__ == '__main__':
    top_list = ["/home/charlesli/dodecane_acrylate/ba_la_dodecane/50A4_50A12_rand0_NPT_313K_1bar_4wt.pdb", "/home/charlesli/dodecane_acrylate/ba_la_dodecane/50A4_50A12_rand1_NPT_313K_1bar_4wt.pdb"]
    dcd_list_list = [["/home/charlesli/dodecane_acrylate/ba_la_dodecane/50A4_50A12_rand0_NPT_313K_1bar_4wt_0.dcd",
                      "/home/charlesli/dodecane_acrylate/ba_la_dodecane/50A4_50A12_rand0_NPT_313K_1bar_4wt_1.dcd"],
                     ["/home/charlesli/dodecane_acrylate/ba_la_dodecane/50A4_50A12_rand1_NPT_313K_1bar_4wt_0.dcd",
                      "/home/charlesli/dodecane_acrylate/ba_la_dodecane/50A4_50A12_rand1_NPT_313K_1bar_4wt_1.dcd"]]
    residue_map = {'A4': (['B4', 'C4'], [5, 4], [14.027, 13.019, 12.011, 15.999, 15.999] + [14.027]*3 + [15.035]),
                   'A12': (['B12', 'C6', 'E6'], [5, 6, 6], [14.027, 13.019, 12.011, 15.999, 15.999] + [14.027]*11 + [15.035]),
                   'C12': (['D', 'D'], [6, 6], [15.035] + [14.027]*10 + [15.035])}
    potential_fix = {'B': ['Gaussian_B4B4', 'Gaussian_B4C4', 'Gaussian_B4D', 
                           'Gaussian_C4C4', 'Gaussian_C4D',
                           'Gaussian_B12B12', 'Gaussian_B12C6', 'Gaussian_B12E6', 'Gaussian_B12D',
                           'Gaussian_C6C6', 'Gaussian_C6E6', 'Gaussian_C6D',
                           'Gaussian_E6E6',
                           'Gaussian_DE6', 'Gaussian_DD'],
                     'Dist0': 'all',
                     'FConst': 'all'}
    map_topology(top_list, dcd_list_list, residue_map, 313.15, 3866.4, potential_fix, cut=2.5, stride=10)

