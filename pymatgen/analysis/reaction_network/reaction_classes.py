# coding: utf-8
# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.


import abc
import logging
import copy
import itertools
import heapq
import numpy as np
from typing import List
from monty.json import MSONable, MontyDecoder
from pymatgen.analysis.graphs import MoleculeGraph, MolGraphSplitError
from pymatgen.analysis.local_env import OpenBabelNN
from pymatgen.io.babel import BabelMolAdaptor
from pymatgen import Molecule
from pymatgen.analysis.fragmenter import metal_edge_extender
import networkx as nx
from networkx.algorithms import bipartite
from pymatgen.entries.mol_entry import MoleculeEntry
from pymatgen.core.composition import CompositionError


__author__ = "Samuel Blau"
__copyright__ = "Copyright 2019, The Materials Project"
__version__ = "1.0"
__maintainer__ = "Samuel Blau"
__email__ = "samblau1@gmail.com"
__status__ = "Alpha"
__date__ = "11/8/19"

reaction_tag_dict = {
    "one_electron_reduction": "one_electron_oxidation",
    "one_electron_oxidation": "one_electron_reduction",
    "intramol_single_bond_formation": "intramol_single_bond_breakage",
    "intramol_single_bond_breakage": "intramol_single_bond_formation",
    "intermol_single_bond_formation": "intermol_single_bond_breakage",
    "intermol_single_bond_breakage": "intermol_single_bond_formation",
    "coordination_bond_formation": "coordination_bond_breakage",
    "coordination_bond_breakage": "coordination_bond_formation",
}


class AbstractReaction(MSONable, metaclass=abc.ABCMeta):
    """
    Abstract reaction object class.
    """

    @abc.abstractproperty
    def energy(self) -> float:
        """
        Returns the reaction's delta E.
        """
        return

    @abc.abstractproperty
    def free_energy(self) -> float:
        """
        Returns the reaction's delta G.
        """
        return

    @abc.abstractproperty
    def build_graph_representation(self) -> nx.DiGraph:
        """
        Returns an nx.DiGraph object of the reaction.
        """
        return


class OneElectronRedoxReaction(AbstractReaction):
    """
    Redox reaction object class.
    """

    def __init__(
        self,
        reactant: MoleculeEntry,
        product: MoleculeEntry,
        electron_free_energy: float = -2.15,
    ):
        """
        Initializes the redox reaction object. 

        Args:
            reactants:
                Reactant MoleculeEntry object.
            products:
                Product MoleculeEntry object.
            electron_free_energy:
                Float that defines the electron free energy.
        """
        assert (
            abs(reactant.charge - product.charge) == 1
        ), "One electron redox entries must have a charge difference of one!"
        self.electron_free_energy = electron_free_energy
        self.reactant = reactant
        self.product = product

    @property
    def tag(self):
        return (
            "one_electron_reduction"
            if self.reactant.charge < self.product.charge
            else "one_electron_oxidation"
        )

    @property
    def energy(self):
        """
        Returns the reaction's delta E.
        """
        if "reduction" in self.tag:
            return self.entry1.energy - self.entry0.energy - self.electron_free_energy
        else:
            return self.entry1.energy + self.electron_free_energy - self.entry0.energy

    @property
    def free_energy(self):
        """
        Returns the reaction's delta G.
        """
        if "reduction" in self.tag:
            return (
                self.entry1.free_energy
                - self.entry0.free_energy
                - self.electron_free_energy
            )
        else:
            return (
                self.entry1.free_energy
                + self.electron_free_energy
                - self.entry0.free_energy
            )

    def add_reaction(self, RN_graph):
        return add_reaction_1_1(
            RN_graph, self.entries0, self.entries1, self.tag, self.free_energy
        )


def add_reaction_1_1(RN_graph, entries0, entries1, tag, dG):
    """
    Adds the reaction to the reaction network graph.

    Args:
        RN_graph:
            nx.DiGraph previously initialized by a Reaction
            Network class object.
        entries0:

    Returns:
        new_RN_graph
            Identical to RN_graph except with the additional
            nodes and edges added for this reaction.
    """
    new_RN_graph = copy.deepcopy(RN_graph)
    # Do stuff!
    return new_RN_graph
