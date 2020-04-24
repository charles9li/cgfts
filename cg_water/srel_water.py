import mdtraj as md
import sim

maxIter = None
steepestIter = 0

sysName = "water"

# ============= #
# Create system #
# ============= #

AtomTypes = {}
MolTypes = []
NMons = []
for MolName, NMol in NMolsDict.items():
    if NMol == 0:
        break
    AtomsInMol = []
    AtomNames = MolTypesDict[MolName]
    NMons.append(len(AtomNames))
    for AtomName in AtomNames:
        AtomsInMol.append(AtomTypes[AtomName])
    MolType = sim.chem.MolType(MolName, AtomsInMol)
    MolTypes.append(MolType)
world = sim.chem.World(MolTypes, Dim=3, Units=units)
sys = sim.system.System(world, Name=sysName)

optimizer = sim.srel.OptimizeMultiTrajClass(opts, Weights=weights)
optimizer.FilePrefix = ("{}".format(sysName))
optimizer.RunConjugateGradient(MaxIter=maxIter, SteepestIter=steepestIter)
