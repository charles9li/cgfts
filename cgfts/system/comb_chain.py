from __future__ import absolute_import, division, print_function

from ._molecule import _Molecule
from .linear_chain import LinearChain
from cgfts.utils import blockify

__all__ = ['CombChain']


class CombChain(_Molecule):

    def __init__(self, name):
        super(CombChain, self).__init__(name)
        self._backbone_bead_names = []
        self._side_arm_types = []

    @property
    def backbone_beads(self):
        return self._backbone_bead_names

    def add_backbone_bead(self, bead_name):
        self._backbone_bead_names.append(bead_name)

    def add_side_arm_type(self, bead_names, grafting_positions):
        name = str(len(self._side_arm_types))
        side_arm_type = SideArmType(name)
        for bn in bead_names:
            side_arm_type.add_bead(bn)
        self._side_arm_types.append((side_arm_type, grafting_positions))

    def get_side_arm_type(self, side_arm_index):
        return self._side_arm_types[side_arm_index][0]

    def get_side_arm_grafting_indices(self, side_arm_index):
        return self._side_arm_types[side_arm_index][1]

    @property
    def side_arm_types(self):
        for i in range(self.n_side_arm_types):
            yield self.get_side_arm_type(i)

    @property
    def side_arm_grafting_indices(self):
        for i in range(self.n_side_arm_types):
            yield self.get_side_arm_grafting_indices(i)

    @property
    def n_side_arm_types(self):
        return len(self._side_arm_types)

    @property
    def n_beads(self):
        n = len(self._backbone_bead_names)
        for i in range(self.n_side_arm_types):
            n += self.get_side_arm_type(i).n_beads * len(self.get_side_arm_grafting_indices(i))
        return n

    def to_sim(self, force_field):
        import sim
        curr_bead_index = 0
        backbone_bead_indices = []
        sim_MolType = sim.chem.MolType(Name=self.name)
        for bb_index, bb in enumerate(self.backbone_beads):
            sim_BackboneAtomType = force_field.get_bead_type(bb).to_sim()
            sim_MolType.append(sim_BackboneAtomType)
            backbone_bead_indices.append(curr_bead_index)
            curr_bead_index += 1
            for (sat, sagi) in zip(self.side_arm_types, self.side_arm_grafting_indices):
                if bb_index in sagi:
                    for sb in sat.beads:
                        sim_SideArmAtomType = force_field.get_bead_type(sb).to_sim()
                        sim_MolType.append(sim_SideArmAtomType)
                        sim_MolType.Bond(curr_bead_index - 1, curr_bead_index)
                        curr_bead_index += 1
        for i in range(len(backbone_bead_indices) - 1):
            sim_MolType.Bond(backbone_bead_indices[i], backbone_bead_indices[i+1])
        return sim_MolType

    def to_PolyFTS(self, force_field, chain_index, tab="  "):

        bead_names = list(force_field.bead_names)

        s = tab*2 + "chain{} {{".format(chain_index)
        s += "\n"
        s += tab*3 + "Label = {}".format(self.name)
        s += "\n"
        s += tab*3 + "Architecture = comb"
        s += "\n"
        s += tab*3 + "Statistics = FJC"
        s += "\n"
        s += "\n"

        # create backbone section
        n_beads_bb = len(self._backbone_bead_names)
        block_names_bb, n_per_block_bb = blockify(self._backbone_bead_names)
        block_species_bb = [bead_names.index(n)+1 for n in block_names_bb]
        s += tab*3 + "backbone {"
        s += "\n"
        s += tab*4 + "Statistics = FJC"
        s += "\n"
        s += tab*4 + "Nbeads = {}".format(n_beads_bb)
        s += "\n"
        s += tab*4 + "NBlocks = {}".format(len(block_names_bb))
        s += "\n"
        s += tab*4 + "BlockSpecies = {}".format(" ".join([str(s) for s in block_species_bb]))
        s += "\n"
        s += tab*4 + "NPerBlock = {}".format(" ".join([str(n) for n in n_per_block_bb]))
        s += "\n"
        s += tab*3 + "}"
        s += "\n"

        # create side arm types
        side_arm_index = 1
        for (sat, sagi) in zip(self.side_arm_types, self.side_arm_grafting_indices):
            block_names_sat, n_per_block_sat = blockify(sat.beads)
            block_species_sat = [bead_names.index(n)+1 for n in block_names_sat]
            s += "\n"
            s += tab*3 + "sidearmtype{} {{".format(side_arm_index)
            s += "\n"
            s += tab*4 + "Statistics = FJC"
            s += "\n"
            s += tab*4 + "NBeads = {}".format(sat.n_beads)
            s += "\n"
            s += tab*4 + "NBlocks = {}".format(len(block_names_sat))
            s += "\n"
            s += tab*4 + "BlockSpecies = {}".format(" ".join([str(b) for b in block_species_sat]))
            s += "\n"
            s += tab*4 + "NPerBlock = {}".format(" ".join([str(n) for n in n_per_block_sat]))
            s += "\n"
            s += tab*4 + "NumArms = {}".format(len(sagi))
            s += "\n"
            s += tab*4 + "BackboneGraftingPositions = {}".format(" ".join([str(g) for g in sagi]))
            s += "\n"
            s += tab*3 + "}"
            s += "\n"
            side_arm_index += 1

        s += tab*2 + "}"
        s += "\n"

        return s


class SideArmType(LinearChain):

    def __init__(self, name):
        super(SideArmType, self).__init__(name)
