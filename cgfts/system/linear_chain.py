from __future__ import absolute_import, division, print_function

from ._molecule import _Molecule
from cgfts.utils import blockify

__all__ = ['LinearChain']


class LinearChain(_Molecule):

    def __init__(self, name):
        super(LinearChain, self).__init__(name)
        self._beads = []

    @property
    def beads(self):
        return self._beads

    def add_bead(self, bead_name):
        self._beads.append(bead_name)

    @property
    def n_beads(self):
        return len(self._beads)

    def to_graph(self, force_field):
        import networkx as nx

        # initialize graph
        G = nx.Graph()

        # compute kuhn lengths
        kuhn_lengths = force_field._determine_kuhn_lengths()
        bead_types = list(force_field.bead_types)

        # create graph
        for i, bead_name in enumerate(self._beads):
            G.add_node(i, name=bead_name)
            if i > 0:
                b = kuhn_lengths[bead_types.index(bead_name)]
                G.add_edge(i, i - 1, weight=b)

    def to_sim(self, force_field):
        import sim
        sim_AtomTypes = [force_field.get_bead_type(bn).to_sim() for bn in self._beads]
        sim_MolType = sim.chem.MolType(Name=self.name, AtomTypes=sim_AtomTypes)
        for i in range(self.n_beads - 1):
            sim_MolType.Bond(i, i + 1)
        return sim_MolType

    def to_PolyFTS(self, force_field, chain_index, tab="  "):

        bead_names = list(force_field.bead_names)
        block_names, n_per_block = blockify(self._beads)
        block_species = [bead_names.index(n)+1 for n in block_names]

        s = tab*2 + "chain{} {{".format(chain_index)
        s += "\n"
        s += tab*3 + "Label = {}".format(self.name)
        s += "\n"
        s += tab*3 + "Architecture = linear"
        s += "\n"
        s += tab*3 + "Statistics = FJC"
        s += "\n"
        s += "\n"
        s += tab*3 + "Nbeads = {}".format(self.n_beads)
        s += "\n"
        s += tab*3 + "NBlocks = {}".format(len(block_names))
        s += "\n"
        s += tab*3 + "BlockSpecies = {}".format(" ".join([str(s) for s in block_species]))
        s += "\n"
        s += tab*3 + "NPerBlock = {}".format(" ".join([str(n) for n in n_per_block]))
        s += "\n"
        s += tab*2 + "}"
        s += "\n"

        return s
