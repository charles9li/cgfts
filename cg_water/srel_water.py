from collections import OrderedDict as ODict
import mdtraj as md
import numpy as np
import sim

# Parameters
Units = sim.units.MKSUnits
AAtrajs = ["water.dcd"]
AAtops = ["tip3p.pdb"]

maxIter = None
steepestIter = 0

sysName = "water"
NMolsDicts = [[('H2O', 895)]]
MolTypesDicts = [{'H2O': ['H2O']}]

# ============== #
# Map trajectory #
# ============== #

traj = md.load_dcd("water.dcd", top="tip3p.pdb")
top = traj.topology
AAatomId = []
CGatomTypes = []
AAatomsInThisCGBead = []
for atom in list(top.atoms):
    AAatomsInThisCGBead.append(atom.index)
    if atom.name == 'H2':
        CGatomTypes.append('H2O')
        AAatomId.append(AAatomsInThisCGBead)
        AAatomsInThisCGBead = []
UniqueCGAtomTypes = np.unique(CGatomTypes)

# create mapped object
Map = sim.atommap.PosMap()
for i, CGatomType in enumerate(CGatomTypes):
    Atoms1 = AAatomId[i]
    Atom2 = i
    this_map = sim.atommap.AtomMap(Atoms1, Atom2)
    Map.append(this_map)
traj.save("water.lammpstrj")
Trj = sim.traj.Lammps("water.lammpstrj")
MappedTrj = sim.traj.Mapped(Trj, Map, AtomNames=CGatomTypes, BoxL=traj.unitcell_lengths[0][0]*10.)

# ============= #
# Create system #
# ============= #

MolTypesDict = MolTypesDicts[0]
NMolsDict = ODict(NMolsDicts[0])

AtomTypes = {}
MolTypes = []
for AtomName in UniqueCGAtomTypes:
    AtomType = sim.chem.AtomType(AtomName, Mass=18.02, Charge=0.0)
    AtomTypes.update({AtomName: AtomType})

for MolName, NMol in NMolsDict.items():
    if NMol == 0:
        break
    AtomsInMol = []
    AtomNames = MolTypesDict[MolName]
    for AtomName in AtomNames:
        AtomsInMol.append(AtomTypes[AtomName])
    MolType = sim.chem.MolType(MolName, AtomsInMol)
    MolTypes.append(MolType)
world = sim.chem.World(MolTypes, Dim=3, Units=Units)
sys = sim.system.System(world, Name=sysName)

# optimizer = sim.srel.OptimizeMultiTrajClass(opts, Weights=weights)
# optimizer.FilePrefix = ("{}".format(sysName))
# optimizer.RunConjugateGradient(MaxIter=maxIter, SteepestIter=steepestIter)
