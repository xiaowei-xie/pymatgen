import json
from json import JSONEncoder

from networkx.readwrite import json_graph
import time
import yaml
from networkx.readwrite import json_graph
import networkx.algorithms.isomorphism as iso

from abc import ABCMeta, abstractproperty, abstractmethod, abstractclassmethod
from abc import ABC, abstractmethod
from time import time
from gunicorn.util import load_class
from monty.json import MSONable
import logging
import copy
import itertools
import heapq
import numpy as np
from monty.json import MSONable, MontyDecoder
from monty.serialization import dumpfn, loadfn

from pymatgen.analysis.graphs import MoleculeGraph, MolGraphSplitError
from pymatgen.analysis.local_env import OpenBabelNN
from pymatgen.io.babel import BabelMolAdaptor
from pymatgen import Molecule
from pymatgen.analysis.fragmenter import metal_edge_extender
import networkx as nx
from networkx.algorithms import bipartite
from pymatgen.entries.mol_entry import MoleculeEntry
from pymatgen.core.composition import CompositionError
from typing import List, Dict, Tuple, Generator
from pymatgen.analysis.reaction_network.extract_reactions import *
import os

MappingDict = Dict[str, Dict[int, Dict[int, List[MoleculeEntry]]]]
Mapping_Energy_Dict = Dict[str, float]
Mapping_ReactionType_Dict = Dict[str, str]
Mapping_Record_Dict = Dict[str, List[str]]


class Reaction(MSONable, metaclass=ABCMeta):
    """
       Abstract class for subsequent types of reaction class
       Args:
           reactants ([MoleculeEntry]): A list of MoleculeEntry objects of len 1.
           products ([MoleculeEntry]): A list of MoleculeEntry objects of max len 2
       """

    def __init__(self, reactants: List[MoleculeEntry], products: List[MoleculeEntry]):
        self.reactants = reactants
        self.products = products
        self.entry_ids = {e.entry_id for e in self.reactants}

    def __in__(self, entry: MoleculeEntry):
        return entry.entry_id in self.entry_ids

    def __len__(self):
        return len(self.reactants)

    @classmethod
    @abstractmethod
    def generate(cls, entries: MappingDict):
        pass

    @abstractmethod
    def graph_representation(self) -> nx.DiGraph:
        pass

    @abstractmethod
    def reaction_type(self) -> Mapping_ReactionType_Dict:
        pass

    @abstractmethod
    def energy(self) -> Mapping_Energy_Dict:
        pass

    @abstractmethod
    def free_energy(self) -> Mapping_Energy_Dict:
        pass

    @abstractmethod
    def rate(self):
        pass


#     @abstractmethod
#     def num_entries(self):
#         """
# 		number of molecule entries that are intertacting
# 		num reactants + num products not including electron
# 		"""
#
#     @abstractmethod
#     def virtual_edges(self, from_entry):
#         """
# 		Returns the virtual networkx edge that goes from the molecule node to the PR reaction node
# 		Virtual edge - (from_node,to_node)
# 		"""
#         pass
#
#     @abstractmethod
#     def update_weights(self, preq_costs: Dict[str, float]) -> Dict[Tuple(str, str), Dict[str, float]]:
#         """
#
# 		Arguments:
# 			prereq_costs: dictionary mapping entry_id to the new/delta cost
#
# 		Returns:
# 			Dictionary to update weights of edges in networkx graph
# 			Dictionary key is a tuple for the two nodes that define the edge
# 			Dictionary value is a dictionary with key of the weight function(s)
# 				and value of the new weight in new/delta cost
# # 		"""

def graph_rep_2_2(reaction: Reaction) -> nx.DiGraph:
    """
    A method to convert a reaction type object into graph representation. Reaction much be of type 2 reactants -> 2
    products
    Args:
       :param reaction: (any of the reaction class object, ex. RedoxReaction, IntramolSingleBondChangeReaction, Concerted)
    """

    if len(reaction.reactants) != 2 or len(reaction.products) != 2:
        raise ValueError("Must provide reaction with 2 reactants and 2 products for graph_rep_2_2")

    reactant_0 = reaction.reactants[0]
    reactant_1 = reaction.reactants[1]
    product_0 = reaction.products[0]
    product_1 = reaction.products[1]
    graph = nx.DiGraph()

    if product_0.parameters["ind"] <= product_1.parameters["ind"]:
        two_prod_name = str(product_0.parameters["ind"]) + "+" + str(product_1.parameters["ind"])
        two_prod_name_entry_ids = str(product_0.entry_id) + "+" + str(product_1.entry_id)
    else:
        two_prod_name = str(product_1.parameters["ind"]) + "+" + str(product_0.parameters["ind"])
        two_prod_name_entry_ids = str(product_1.entry_id) + "+" + str(product_0.entry_id)

    if reactant_0.parameters["ind"] <= reactant_1.parameters["ind"]:
        two_reac_name = str(reactant_0.parameters["ind"]) + "+" + str(reactant_1.parameters["ind"])
        two_reac_name_entry_ids = str(reactant_0.entry_id) + "+" + str(reactant_1.entry_id)
    else:
        two_reac_name = str(reactant_1.parameters["ind"]) + "+" + str(reactant_0.parameters["ind"])
        two_reac_name_entry_ids = str(reactant_1.entry_id) + "+" + str(reactant_0.entry_id)

    two_prod_name0 = str(product_0.parameters["ind"]) + "+PR_" + str(product_1.parameters["ind"])
    two_prod_name1 = str(product_1.parameters["ind"]) + "+PR_" + str(product_0.parameters["ind"])

    two_reac_name0 = str(reactant_0.parameters["ind"]) + "+PR_" + str(reactant_1.parameters["ind"])
    two_reac_name1 = str(reactant_1.parameters["ind"]) + "+PR_" + str(reactant_0.parameters["ind"])

    node_name_A0 = two_reac_name0 + "," + two_prod_name
    node_name_A1 = two_reac_name1 + "," + two_prod_name
    node_name_B0 = two_prod_name0 + "," + two_reac_name
    node_name_B1 = two_prod_name1 + "," + two_reac_name

    two_prod_entry_ids0 = str(product_0.entry_id) + "+PR_" + str(product_1.entry_id)
    two_prod_entry_ids1 = str(product_1.entry_id) + "+PR_" + str(product_0.entry_id)
    
    two_reac_entry_ids0 = str(reactant_0.entry_id) + "+PR_" + str(reactant_1.entry_id)
    two_reac_entry_ids1 = str(reactant_1.entry_id) + "+PR_" + str(reactant_0.entry_id)
    
    entry_ids_name_A0 = two_reac_entry_ids0 + "," + two_prod_name_entry_ids
    entry_ids_name_A1 = two_reac_entry_ids1 + "," + two_prod_name_entry_ids
    entry_ids_name_B0 = two_prod_entry_ids0 + "," + two_reac_name_entry_ids
    entry_ids_name_B1 = two_prod_entry_ids1 + "," + two_reac_name_entry_ids

    rxn_type_A = reaction.reaction_type()["rxn_type_A"]
    rxn_type_B = reaction.reaction_type()["rxn_type_B"]
    energy_A = reaction.energy()["energy_A"]
    energy_B = reaction.energy()["energy_B"]
    free_energy_A = reaction.free_energy()["free_energy_A"]
    free_energy_B = reaction.free_energy()["free_energy_B"]

    graph.add_node(node_name_A0, rxn_type=rxn_type_A, bipartite=1, energy=energy_A, free_energy=free_energy_A,
                   entry_ids=entry_ids_name_A0)

    graph.add_edge(reactant_0.parameters["ind"],
                   node_name_A0,
                   softplus=ReactionNetwork.softplus(free_energy_A),
                   exponent=ReactionNetwork.exponent(free_energy_A),
                   weight=1.0
                   )

    graph.add_edge(node_name_A0,
                   product_0.parameters["ind"],
                   softplus=0.0,
                   exponent=0.0,
                   weight=1.0
                   )
    graph.add_edge(node_name_A0,
                   product_1.parameters["ind"],
                   softplus=0.0,
                   exponent=0.0,
                   weight=1.0
                   )

    graph.add_node(node_name_A1, rxn_type=rxn_type_A, bipartite=1, energy=energy_A, free_energy=free_energy_A,
                   entry_ids=entry_ids_name_A1)

    graph.add_edge(reactant_1.parameters["ind"],
                   node_name_A1,
                   softplus=ReactionNetwork.softplus(free_energy_A),
                   exponent=ReactionNetwork.exponent(free_energy_A),
                   weight=1.0
                   )

    graph.add_edge(node_name_A1,
                   product_0.parameters["ind"],
                   softplus=0.0,
                   exponent=0.0,
                   weight=1.0
                   )
    graph.add_edge(node_name_A1,
                   product_1.parameters["ind"],
                   softplus=0.0,
                   exponent=0.0,
                   weight=1.0
                   )

    graph.add_node(node_name_B0, rxn_type=rxn_type_B, bipartite=1, energy=energy_B, free_energy=free_energy_B,
                   entry_ids=entry_ids_name_B0)

    graph.add_edge(product_0.parameters["ind"],
                   node_name_B0,
                   softplus=ReactionNetwork.softplus(free_energy_B),
                   exponent=ReactionNetwork.exponent(free_energy_B),
                   weight=1.0
                   )

    graph.add_edge(node_name_B0,
                   reactant_0.parameters["ind"],
                   softplus=0.0,
                   exponent=0.0,
                   weight=1.0
                   )
    graph.add_edge(node_name_B0,
                   reactant_1.parameters["ind"],
                   softplus=0.0,
                   exponent=0.0,
                   weight=1.0
                   )

    graph.add_node(node_name_B1, rxn_type=rxn_type_B, bipartite=1, energy=energy_B, free_energy=free_energy_B,
                   entry_ids=entry_ids_name_B1)

    graph.add_edge(product_1.parameters["ind"],
                   node_name_B1,
                   softplus=ReactionNetwork.softplus(free_energy_B),
                   exponent=ReactionNetwork.exponent(free_energy_B),
                   weight=1.0
                   )

    graph.add_edge(node_name_B1,
                   reactant_0.parameters["ind"],
                   softplus=0.0,
                   exponent=0.0,
                   weight=1.0
                   )
    graph.add_edge(node_name_B1,
                   reactant_1.parameters["ind"],
                   softplus=0.0,
                   exponent=0.0,
                   weight=1.0
                   )

    return graph


def graph_rep_1_2(reaction: Reaction) -> nx.DiGraph:
    """
    A method to convert a reaction type object into graph representation. Reaction much be of type 1 reactant -> 2
    products
    Args:
       :param reaction: (any of the reaction class object, ex. RedoxReaction, IntramolSingleBondChangeReaction)
    """

    if len(reaction.reactants) != 1 or len(reaction.products) != 2:
        raise ValueError("Must provide reaction with 1 reactant and 2 products for graph_rep_1_2")

    reactant_0 = reaction.reactants[0]
    product_0 = reaction.products[0]
    product_1 = reaction.products[1]
    graph = nx.DiGraph()

    if product_0.parameters["ind"] <= product_1.parameters["ind"]:
        two_mol_name = str(product_0.parameters["ind"]) + "+" + str(product_1.parameters["ind"])
        two_mol_name_entry_ids = str(product_0.entry_id) + "+" + str(product_1.entry_id)
    else:
        two_mol_name = str(product_1.parameters["ind"]) + "+" + str(product_0.parameters["ind"])
        two_mol_name_entry_ids = str(product_1.entry_id) + "+" + str(product_0.entry_id)

    two_mol_name0 = str(product_0.parameters["ind"]) + "+PR_" + str(product_1.parameters["ind"])
    two_mol_name1 = str(product_1.parameters["ind"]) + "+PR_" + str(product_0.parameters["ind"])
    node_name_A = str(reactant_0.parameters["ind"]) + "," + two_mol_name
    node_name_B0 = two_mol_name0 + "," + str(reactant_0.parameters["ind"])
    node_name_B1 = two_mol_name1 + "," + str(reactant_0.parameters["ind"])

    two_mol_entry_ids0 = str(product_0.entry_id) + "+PR_" + str(product_1.entry_id)
    two_mol_entry_ids1 = str(product_1.entry_id) + "+PR_" + str(product_0.entry_id)
    entry_ids_name_A = str(reactant_0.entry_id) + "," + two_mol_name_entry_ids
    entry_ids_name_B0 = two_mol_entry_ids0 + "," + str(reactant_0.entry_id)
    entry_ids_name_B1 = two_mol_entry_ids1 + "," + str(reactant_0.entry_id)

    rxn_type_A = reaction.reaction_type()["rxn_type_A"]
    rxn_type_B = reaction.reaction_type()["rxn_type_B"]
    energy_A = reaction.energy()["energy_A"]
    energy_B = reaction.energy()["energy_B"]
    free_energy_A = reaction.free_energy()["free_energy_A"]
    free_energy_B = reaction.free_energy()["free_energy_B"]

    graph.add_node(node_name_A, rxn_type=rxn_type_A, bipartite=1, energy=energy_A, free_energy=free_energy_A,
                   entry_ids=entry_ids_name_A)

    graph.add_edge(reactant_0.parameters["ind"],
                   node_name_A,
                   softplus=ReactionNetwork.softplus(free_energy_A),
                   exponent=ReactionNetwork.exponent(free_energy_A),
                   weight=1.0
                   )

    graph.add_edge(node_name_A,
                   product_0.parameters["ind"],
                   softplus=0.0,
                   exponent=0.0,
                   weight=1.0
                   )
    graph.add_edge(node_name_A,
                   product_1.parameters["ind"],
                   softplus=0.0,
                   exponent=0.0,
                   weight=1.0
                   )

    graph.add_node(node_name_B0, rxn_type=rxn_type_B, bipartite=1, energy=energy_B, free_energy=free_energy_B,
                   entry_ids=entry_ids_name_B0)
    graph.add_node(node_name_B1, rxn_type=rxn_type_B, bipartite=1, energy=energy_B, free_energy=free_energy_B,
                   entry_ids=entry_ids_name_B1)

    graph.add_edge(node_name_B0,
                   reactant_0.parameters["ind"],
                   softplus=0.0,
                   exponent=0.0,
                   weight=1.0
                   )
    graph.add_edge(node_name_B1,
                   reactant_0.parameters["ind"],
                   softplus=0.0,
                   exponent=0.0,
                   weight=1.0
                   )

    graph.add_edge(product_0.parameters["ind"],
                   node_name_B0,
                   softplus=ReactionNetwork.softplus(free_energy_B),
                   exponent=ReactionNetwork.exponent(free_energy_B),
                   weight=1.0
                   )
    graph.add_edge(product_1.parameters["ind"],
                   node_name_B1,
                   softplus=ReactionNetwork.softplus(free_energy_B),
                   exponent=ReactionNetwork.exponent(free_energy_B),
                   weight=1.0)
    return graph


def graph_rep_1_1(reaction: Reaction) -> nx.DiGraph:
    """
    A method to convert a reaction type object into graph representation. Reaction much be of type 1 reactant -> 1
    product
    Args:
       :param reaction:(any of the reaction class object, ex. RedoxReaction, IntramolSingleBondChangeReaction)
    """

    if len(reaction.reactants) != 1 or len(reaction.products) != 1:
        raise ValueError("Must provide reaction with 1 reactant and product for graph_rep_1_1")

    reactant_0 = reaction.reactants[0]
    product_0 = reaction.products[0]
    graph = nx.DiGraph()
    node_name_A = str(reactant_0.parameters["ind"]) + "," + str(product_0.parameters["ind"])
    node_name_B = str(product_0.parameters["ind"]) + "," + str(reactant_0.parameters["ind"])
    rxn_type_A = reaction.reaction_type()["rxn_type_A"]
    rxn_type_B = reaction.reaction_type()["rxn_type_B"]
    energy_A = reaction.energy()["energy_A"]
    energy_B = reaction.energy()["energy_B"]
    free_energy_A = reaction.free_energy()["free_energy_A"]
    free_energy_B = reaction.free_energy()["free_energy_B"]
    entry_ids_A = str(reactant_0.entry_id) + "," + str(product_0.entry_id)
    entry_ids_B = str(product_0.entry_id) + "," + str(reactant_0.entry_id)

    graph.add_node(node_name_A, rxn_type=rxn_type_A, bipartite=1, energy=energy_A, free_energy=free_energy_A,
                   entry_ids=entry_ids_A)
    graph.add_edge(reactant_0.parameters["ind"],
                   node_name_A,
                   softplus=ReactionNetwork.softplus(free_energy_A),
                   exponent=ReactionNetwork.exponent(free_energy_A),
                   weight=1.0)
    graph.add_edge(node_name_A,
                   product_0.parameters["ind"],
                   softplus=0.0,
                   exponent=0.0,
                   weight=1.0)
    graph.add_node(node_name_B, rxn_type=rxn_type_B, bipartite=1, energy=energy_B, free_energy=free_energy_B,
                   entry_ids=entry_ids_B)
    graph.add_edge(product_0.parameters["ind"],
                   node_name_B,
                   softplus=ReactionNetwork.softplus(free_energy_B),
                   exponent=ReactionNetwork.exponent(free_energy_B),
                   weight=1.0)
    graph.add_edge(node_name_B,
                   reactant_0.parameters["ind"],
                   softplus=0.0,
                   exponent=0.0,
                   weight=1.0)

    return graph


class RedoxReaction(Reaction):
    """
    A class to define redox reactions as follows:
    One electron oxidation / reduction without change to bonding
        A^n ±e- <-> A^n±1
        Two entries with:
        identical composition
        identical number of edges
        a charge difference of 1
        isomorphic molecule graphs
    Args:
       reactant([MolecularEntry]): list of single molecular entry
       product([MoleculeEntry]): list of single molecular entry
    """

    def __init__(self, reactant: MoleculeEntry, product: MoleculeEntry):
        """
            Initilizes RedoxReaction.reactant to be in the form of a MolecularEntry,
            RedoxReaction.product to be in the form of MolecularEntry,
            Reaction.reactant to be in the form of a of a list of MolecularEntry of length 1
            Reaction.products to be in the form of a of a list of MolecularEntry of length 1
          Args:
            :param reactant MolecularEntry object
            :param product MolecularEntry object
        """
        self.reactant = reactant
        self.product = product
        self.electron_free_energy = None
        super().__init__([self.reactant], [self.product])

    def graph_representation(self) -> nx.DiGraph:
        """
            A method to convert a RedoxReaction class object into graph representation (nx.Digraph object).
            Redox Reaction must be of type 1 reactant -> 1 product
            :return nx.Digraph object of a single Redox Reaction
        """

        return graph_rep_1_1(self)

    @classmethod
    def generate(cls, entries: MappingDict) -> List[Reaction]:

        """
        A method to generate all the possible redox reactions from given entries
        Args:
           :param entries: ReactionNetwork(input_entries).entries, entries = {[formula]:{[Nbonds]:{[charge]:MoleculeEntry}}}
           :return list of RedoxReaction class objects
        """

        reactions = []
        for formula in entries:
            for Nbonds in entries[formula]:
                charges = list(entries[formula][Nbonds].keys())
                if len(charges) > 1:
                    for ii in range(len(charges) - 1):
                        charge0 = charges[ii]
                        charge1 = charges[ii + 1]
                        if charge1 - charge0 == 1:
                            for entry0 in entries[formula][Nbonds][charge0]:
                                for entry1 in entries[formula][Nbonds][charge1]:
                                    if entry0.mol_graph.isomorphic_to(entry1.mol_graph):
                                        r = cls(entry0, entry1)
                                        reactions.append(r)

        return reactions

    def reaction_type(self) -> Mapping_ReactionType_Dict:
        """
        A method to identify type of redox reaction (oxidation or reduction)
        Args:
           :return dictionary of the form {"class": "RedoxReaction", "rxn_type_A": rxn_type_A, "rxn_type_B": rxn_type_B}
           where rnx_type_A is the primary type of the reaction based on the reactant and product of the RedoxReaction
           object, and the backwards of this reaction would be rnx_type_B
        """

        if self.product.charge < self.reactant.charge:
            rxn_type_A = "One electron reduction"
            rxn_type_B = "One electron oxidation"
        else:
            rxn_type_A = "One electron oxidation"
            rxn_type_B = "One electron reduction"

        reaction_type = {"class": "RedoxReaction", "rxn_type_A": rxn_type_A, "rxn_type_B": rxn_type_B}
        return reaction_type

    def free_energy(self) -> Mapping_Energy_Dict:
        """
           A method to determine the free energy of the redox reaction. Note to set RedoxReaction.eletron_free_energy a value.
           Args:
              :return dictionary of the form {"free_energy_A": free_energy_A, "free_energy_B": free_energy_B}
              where free_energy_A is the primary type of the reaction based on the reactant and product of the RedoxReaction
              object, and the backwards of this reaction would be free_energy_B.
        """

        if self.product.free_energy is not None and self.reactant.free_energy is not None:
            free_energy_A = self.product.free_energy - self.reactant.free_energy
            free_energy_B = self.reactant.free_energy - self.product.free_energy

            if self.reaction_type()["rxn_type_A"] == "One electron reduction":
                free_energy_A += -self.electron_free_energy
                free_energy_B += self.electron_free_energy
            else:
                free_energy_A += self.electron_free_energy
                free_energy_B += -self.electron_free_energy
        else:
            free_energy_A = None
            free_energy_B = None

        return {"free_energy_A": free_energy_A, "free_energy_B": free_energy_B}

    def energy(self) -> Mapping_Energy_Dict:
        """
           A method to determine the energy of the redox reaction
           Args:
              :return dictionary of the form {"energy_A": energy_A, "energy_B": energy_B}
              where energy_A is the primary type of the reaction based on the reactant and product of the RedoxReaction
              object, and the backwards of this reaction would be energy_B.
        """
        if self.product.energy is not None and self.reactant.energy is not None:
            energy_A = self.product.energy - self.reactant.energy
            energy_B = self.reactant.energy - self.product.energy
        else:
            energy_A = None
            energy_B = None

        return {"energy_A": energy_A, "energy_B": energy_B}

    def rate(self):
        pass


class IntramolSingleBondChangeReaction(Reaction):
    """
    A class to define intramolecular single bond change as follows:
        Intramolecular formation / breakage of one bond
        A^n <-> B^n
        Two entries with:
            identical composition
            number of edges differ by 1
            identical charge
            removing one of the edges in the graph with more edges yields a graph isomorphic to the other entry
    Args:
       reactant([MolecularEntry]): list of single molecular entry
       product([MoleculeEntry]): list of single molecular entry
    """

    def __init__(self, reactant: MoleculeEntry, product: MoleculeEntry):
        """
            Initilizes IntramolSingleBondChangeReaction.reactant to be in the form of a MolecularEntry,
            IntramolSingleBondChangeReaction.product to be in the form of MolecularEntry,
            Reaction.reactant to be in the form of a of a list of MolecularEntry of length 1
            Reaction.products to be in the form of a of a list of MolecularEntry of length 1
          Args:
            :param reactant MolecularEntry object
            :param product MolecularEntry object
        """

        self.reactant = reactant
        self.product = product
        super().__init__([self.reactant], [self.product])

    def graph_representation(self) -> nx.DiGraph:
        """
            A method to convert a IntramolSingleBondChangeReaction class object into graph representation (nx.Digraph object).
           IntramolSingleBondChangeReaction must be of type 1 reactant -> 1 product
            :return nx.Digraph object of a single IntramolSingleBondChangeReaction object
        """

        return graph_rep_1_1(self)

    @classmethod
    def generate(cls, entries: MappingDict) -> List[Reaction]:

        """
            A method to generate all the possible intermolecular single bond change reactions from given entries
            Args:
               :param entries: ReactionNetwork(input_entries).entries, entries = {[formula]:{[Nbonds]:{[charge]:MoleculeEntry}}}
               :return list of IntramolSingleBondChangeReaction class objects
        """

        reactions = []
        for formula in entries:
            Nbonds_list = list(entries[formula].keys())
            if len(Nbonds_list) > 1:
                for ii in range(len(Nbonds_list) - 1):
                    Nbonds0 = Nbonds_list[ii]
                    Nbonds1 = Nbonds_list[ii + 1]
                    if Nbonds1 - Nbonds0 == 1:
                        for charge in entries[formula][Nbonds0]:
                            if charge in entries[formula][Nbonds1]:
                                for entry1 in entries[formula][Nbonds1][charge]:
                                    for edge in entry1.edges:
                                        mg = copy.deepcopy(entry1.mol_graph)
                                        mg.break_edge(edge[0], edge[1], allow_reverse=True)
                                        if nx.is_weakly_connected(mg.graph):
                                            for entry0 in entries[formula][Nbonds0][charge]:
                                                if entry0.mol_graph.isomorphic_to(mg):
                                                    r = cls(entry0, entry1)
                                                    reactions.append(r)
                                                    break

        return reactions

    def reaction_type(self) -> Mapping_ReactionType_Dict:
        """
            A method to identify type of intramolecular single bond change reaction (bond breakage or formation)
            Args:
               :return dictionary of the form {"class": "IntramolSingleBondChangeReaction", "rxn_type_A": rxn_type_A, "rxn_type_B": rxn_type_B}
               where rnx_type_A is the primary type of the reaction based on the reactant and product of the IntramolSingleBondChangeReaction
               object, and the backwards of this reaction would be rnx_type_B
        """
        if self.product.charge < self.reactant.charge:
            rxn_type_A = "Intramolecular single bond breakage"
            rxn_type_B = "Intramolecular single bond formation"
        else:
            rxn_type_A = "Intramolecular single bond formation"
            rxn_type_B = "Intramolecular single bond breakage"

        reaction_type = {"class": "IntramolSingleBondChangeReaction", "rxn_type_A": rxn_type_A,
                         "rxn_type_B": rxn_type_B}
        return reaction_type

    def free_energy(self) -> Mapping_Energy_Dict:
        """
          A method to  determine the free energy of the intramolecular single bond change reaction
          Args:
             :return dictionary of the form {"free_energy_A": energy_A, "free_energy_B": energy_B}
             where free_energy_A is the primary type of the reaction based on the reactant and product of the IntramolSingleBondChangeReaction
             object, and the backwards of this reaction would be free_energy_B.
        """
        if self.product.free_energy is not None and self.reactant.free_energy is not None:
            free_energy_A = self.product.free_energy - self.reactant.free_energy
            free_energy_B = self.reactant.free_energy - self.product.free_energy
        else:
            free_energy_A = None
            free_energy_B = None

        return {"free_energy_A": free_energy_A, "free_energy_B": free_energy_B}

    def energy(self) -> Mapping_Energy_Dict:
        """
          A method to determine the energy of the intramolecular single bond change reaction
          Args:
             :return dictionary of the form {"energy_A": energy_A, "energy_B": energy_B}
             where energy_A is the primary type of the reaction based on the reactant and product of the IntramolSingleBondChangeReaction
             object, and the backwards of this reaction would be energy_B.
         """

        if self.product.energy is not None and self.reactant.energy is not None:
            energy_A = self.product.energy - self.reactant.energy
            energy_B = self.reactant.energy - self.product.energy

        else:
            energy_A = None
            energy_B = None

        return {"energy_A": energy_A, "energy_B": energy_B}

    def rate(self):
        pass


class IntermolecularReaction(Reaction):
    """
        A class to define intermolecular bond change as follows:
            Intermolecular formation / breakage of one bond
            A <-> B + C aka B + C <-> A
            Three entries with:
                comp(A) = comp(B) + comp(C)
                charge(A) = charge(B) + charge(C)
                removing one of the edges in A yields two disconnected subgraphs that are isomorphic to B and C
        Args:
           reactant([MolecularEntry]): list of single molecular entry
           product([MoleculeEntry]): list of two molecular entries
    """

    def __init__(self, reactant: MoleculeEntry, product: List[MoleculeEntry]):
        """
            Initilizes IntermolecularReaction.reactant to be in the form of a MolecularEntry,
            IntermolecularReaction.product to be in the form of [MolecularEntry_0, MolecularEntry_1],
            Reaction.reactant to be in the form of a of a list of MolecularEntry of length 1
            Reaction.products to be in the form of a of a list of MolecularEntry of length 2
          Args:
            :param reactant MolecularEntry object
            :param product list of MolecularEntry object of length 2
        """

        self.reactant = reactant
        self.product_0 = product[0]
        self.product_1 = product[1]
        super().__init__([self.reactant], [self.product_0, self.product_1])

    def graph_representation(self) -> nx.DiGraph:  # temp here, use graph_rep_1_2 instead

        """
            A method to convert a IntermolecularReaction class object into graph representation (nx.Digraph object).
            IntermolecularReaction must be of type 1 reactant -> 2 products
            :return nx.Digraph object of a single IntermolecularReaction object
        """

        return graph_rep_1_2(self)

    @classmethod
    def generate(cls, entries: MappingDict) -> List[Reaction]:

        """
           A method to generate all the possible intermolecular reactions from given entries
           Args:
              :param entries: ReactionNetwork(input_entries).entries, entries = {[formula]:{[Nbonds]:{[charge]:MoleculeEntry}}}
              :return list of IntermolecularReaction class objects
        """
        reactions = []
        for formula in entries:
            for Nbonds in entries[formula]:
                if Nbonds > 0:
                    for charge in entries[formula][Nbonds]:
                        for entry in entries[formula][Nbonds][charge]:
                            for edge in entry.edges:
                                bond = [(edge[0], edge[1])]
                                try:
                                    frags = entry.mol_graph.split_molecule_subgraphs(bond, allow_reverse=True)
                                    formula0 = frags[0].molecule.composition.alphabetical_formula
                                    Nbonds0 = len(frags[0].graph.edges())
                                    formula1 = frags[1].molecule.composition.alphabetical_formula
                                    Nbonds1 = len(frags[1].graph.edges())
                                    if formula0 in entries and formula1 in entries:
                                        if Nbonds0 in entries[formula0] and Nbonds1 in entries[formula1]:
                                            for charge0 in entries[formula0][Nbonds0]:
                                                for entry0 in entries[formula0][Nbonds0][charge0]:
                                                    if frags[0].isomorphic_to(entry0.mol_graph):
                                                        charge1 = charge - charge0
                                                        if charge1 in entries[formula1][Nbonds1]:
                                                            for entry1 in entries[formula1][Nbonds1][charge1]:
                                                                if frags[1].isomorphic_to(entry1.mol_graph):
                                                                    # r1 = ReactionEntry([entry], [entry0, entry1])
                                                                    r = cls(entry, [entry0, entry1])
                                                                    reactions.append(r)
                                                                    break
                                                        break
                                except MolGraphSplitError:
                                    pass

        return reactions

    def reaction_type(self) -> Mapping_ReactionType_Dict:

        """
           A method to identify type of intermoleular reaction (bond decomposition from one to two or formation from two to one molecules)
           Args:
              :return dictionary of the form {"class": "IntermolecularReaction", "rxn_type_A": rxn_type_A, "rxn_type_B": rxn_type_B}
              where rnx_type_A is the primary type of the reaction based on the reactant and product of the IntermolecularReaction
              object, and the backwards of this reaction would be rnx_type_B
        """

        rxn_type_A = "Molecular decomposition breaking one bond A -> B+C"
        rxn_type_B = "Molecular formation from one new bond A+B -> C"

        reaction_type = {"class": "IntermolecularReaction", "rxn_type_A": rxn_type_A, "rxn_type_B": rxn_type_B}
        return reaction_type

    def free_energy(self) -> Mapping_Energy_Dict:
        """
          A method to determine the free energy of the intermolecular reaction
          Args:
             :return dictionary of the form {"free_energy_A": energy_A, "free_energy_B": energy_B}
             where free_energy_A is the primary type of the reaction based on the reactant and product of the IntermolecularReaction
             object, and the backwards of this reaction would be free_energy_B.
         """
        if self.product_1.free_energy is not None and self.product_0.free_energy is not None and self.reactant.free_energy is not None:
            free_energy_A = self.product_0.free_energy + self.product_1.free_energy - self.reactant.free_energy
            free_energy_B = self.reactant.free_energy - self.product_0.free_energy - self.product_1.free_energy

        else:
            free_energy_A = None
            free_energy_B = None

        return {"free_energy_A": free_energy_A, "free_energy_B": free_energy_B}

    def energy(self) -> Mapping_Energy_Dict:
        """
          A method to determine the energy of the intermolecular reaction
          Args:
             :return dictionary of the form {"energy_A": energy_A, "energy_B": energy_B}
             where energy_A is the primary type of the reaction based on the reactant and product of the IntermolecularReaction
             object, and the backwards of this reaction would be energy_B.
        """
        if self.product_1.energy is not None and self.product_0.energy is not None and self.reactant.energy is not None:
            energy_A = self.product_0.energy + self.product_1.energy - self.reactant.energy
            energy_B = self.reactant.energy - self.product_0.energy - self.product_1.energy

        else:
            energy_A = None
            energy_B = None

        return {"energy_A": energy_A, "energy_B": energy_B}

    def rate(self):
        pass


class CoordinationBondChangeReaction(Reaction):
    """
    A class to define coordination bond change as follows:
        Simultaneous formation / breakage of multiple coordination bonds
        A + M <-> AM aka AM <-> A + M
        Three entries with:
            M = Li or Mg
            comp(AM) = comp(A) + comp(M)
            charge(AM) = charge(A) + charge(M)
            removing two M-containing edges in AM yields two disconnected subgraphs that are isomorphic to B and C
    Args:
       reactant([MolecularEntry]): list of single molecular entry
       product([MoleculeEntry]): list of two molecular entries
    """

    def __init__(self, reactant: MoleculeEntry, product: List[MoleculeEntry]):
        """
            Initilizes CoordinationBondChangeReaction.reactant to be in the form of a MolecularEntry,
            CoordinationBondChangeReaction.product to be in the form of [MolecularEntry_0, MolecularEntry_1],
            Reaction.reactant to be in the form of a of a list of MolecularEntry of length 1
            Reaction.products to be in the form of a of a list of MolecularEntry of length 2
          Args:
            :param reactant MolecularEntry object
            :param product list of MolecularEntry object of length 2
        """
        self.reactant = reactant
        self.product_0 = product[0]
        self.product_1 = product[1]
        super().__init__([self.reactant], [self.product_0, self.product_1])

    def graph_representation(self) -> nx.DiGraph:
        """
            A method to convert a CoordinationBondChangeReaction class object into graph representation (nx.Digraph object).
            CoordinationBondChangeReaction must be of type 1 reactant -> 2 products
            :return nx.Digraph object of a single CoordinationBondChangeReaction object
        """

        return graph_rep_1_2(self)

    @classmethod
    def generate(cls, entries: MappingDict) -> List[Reaction]:
        """
          A method to generate all the possible coordination bond chamge reactions from given entries
          Args:
             :param entries: ReactionNetwork(input_entries).entries, entries = {[formula]:{[Nbonds]:{[charge]:MoleculeEntry}}}
             :return list of CoordinationBondChangeReaction class objects
        """
        reactions = []
        M_entries = {}
        for formula in entries:
            if formula == "Li1" or formula == "Mg1":
                if formula not in M_entries:
                    M_entries[formula] = {}
                for charge in entries[formula][0]:
                    assert (len(entries[formula][0][charge]) == 1)
                    M_entries[formula][charge] = entries[formula][0][charge][0]
        if M_entries != {}:
            for formula in entries:
                if "Li" in formula or "Mg" in formula:
                    for Nbonds in entries[formula]:
                        if Nbonds > 2:
                            for charge in entries[formula][Nbonds]:
                                for entry in entries[formula][Nbonds][charge]:
                                    nosplit_M_bonds = []
                                    for edge in entry.edges:
                                        if str(entry.molecule.sites[edge[0]].species) in M_entries or str(
                                                entry.molecule.sites[edge[1]].species) in M_entries:
                                            M_bond = (edge[0], edge[1])
                                            try:
                                                frags = entry.mol_graph.split_molecule_subgraphs([M_bond],
                                                                                                 allow_reverse=True)
                                            except MolGraphSplitError:
                                                nosplit_M_bonds.append(M_bond)
                                    bond_pairs = itertools.combinations(nosplit_M_bonds, 2)
                                    for bond_pair in bond_pairs:
                                        try:
                                            frags = entry.mol_graph.split_molecule_subgraphs(bond_pair,
                                                                                             allow_reverse=True)
                                            M_ind = None
                                            M_formula = None
                                            for ii, frag in enumerate(frags):
                                                frag_formula = frag.molecule.composition.alphabetical_formula
                                                if frag_formula in M_entries:
                                                    M_ind = ii
                                                    M_formula = frag_formula
                                                    break
                                            if M_ind != None:
                                                for ii, frag in enumerate(frags):
                                                    if ii != M_ind:
                                                        # nonM_graph = frag.graph
                                                        nonM_formula = frag.molecule.composition.alphabetical_formula
                                                        nonM_Nbonds = len(frag.graph.edges())
                                                        if nonM_formula in entries:
                                                            if nonM_Nbonds in entries[nonM_formula]:
                                                                for nonM_charge in entries[nonM_formula][
                                                                    nonM_Nbonds]:
                                                                    M_charge = entry.charge - nonM_charge
                                                                    if M_charge in M_entries[M_formula]:
                                                                        for nonM_entry in \
                                                                                entries[nonM_formula][nonM_Nbonds][
                                                                                    nonM_charge]:
                                                                            if frag.isomorphic_to(nonM_entry.mol_graph):
                                                                                r = cls(entry, [nonM_entry,
                                                                                                M_entries[
                                                                                                    M_formula][
                                                                                                    M_charge]])
                                                                                reactions.append(r)
                                                                                break
                                        except MolGraphSplitError:
                                            pass
        return reactions

    def reaction_type(self) -> Mapping_ReactionType_Dict:
        """
           A method to identify type of coordination bond change reaction (bond breaking from one to two or forming from two to one molecules)
           Args:
              :return dictionary of the form {"class": "CoordinationBondChangeReaction", "rxn_type_A": rxn_type_A, "rxn_type_B": rxn_type_B}
              where rnx_type_A is the primary type of the reaction based on the reactant and product of the CoordinationBondChangeReaction
              object, and the backwards of this reaction would be rnx_type_B
        """

        rxn_type_A = "Coordination bond breaking AM -> A+M"
        rxn_type_B = "Coordination bond forming A+M -> AM"

        reaction_type = {"class": "CoordinationBondChangeReaction", "rxn_type_A": rxn_type_A, "rxn_type_B": rxn_type_B}
        return reaction_type

    def free_energy(self) -> Mapping_Energy_Dict:
        """
              A method to determine the free energy of the coordination bond chnage reaction
              Args:
                 :return dictionary of the form {"free_energy_A": energy_A, "free_energy_B": energy_B}
                 where free_energy_A is the primary type of the reaction based on the reactant and product of the CoordinationBondChangeReaction
                 object, and the backwards of this reaction would be free_energy_B.
         """
        if self.product_1.free_energy is not None and self.product_0.free_energy is not None and self.reactant.free_energy is not None:
            free_energy_A = self.product_0.free_energy + self.product_1.free_energy - self.reactant.free_energy
            free_energy_B = self.reactant.free_energy - self.product_0.free_energy - self.product_1.free_energy

        else:
            free_energy_A = None
            free_energy_B = None

        return {"free_energy_A": free_energy_A, "free_energy_B": free_energy_B}

    def energy(self) -> Mapping_Energy_Dict:
        """
              A method to determine the energy of the coordination bond change reaction
              Args:
                 :return dictionary of the form {"energy_A": energy_A, "energy_B": energy_B}
                 where energy_A is the primary type of the reaction based on the reactant and product of the CoordinationBondChangeReaction
                 object, and the backwards of this reaction would be energy_B.
        """
        if self.product_1.energy is not None and self.product_0.energy is not None and self.reactant.energy is not None:
            energy_A = self.product_0.energy + self.product_1.energy - self.reactant.energy
            energy_B = self.reactant.energy - self.product_0.energy - self.product_1.energy

        else:
            energy_A = None
            energy_B = None

        return {"energy_A": energy_A, "energy_B": energy_B}

    def rate(self):
        pass

class ConcertedReaction(Reaction):
    """
        A class to define concerted reactions.
        User can specify either allowing <=1 bond breakage + <=1 bond formation OR <=2 bond breakage + <=2 bond formation.
        User can also specify how many electrons are allowed to involve in a reaction.
        Can only deal with <= 2 reactants and <=2 products for now.
        For 1 reactant -> 1 product reactions, a maximum 1 bond breakage and 1 bond formation is allowed,
        even when the user specify "<=2 bond breakage + <=2 bond formation".
        Args:
           reactant([MolecularEntry]): list of 1-2 molecular entries
           product([MoleculeEntry]): list of 1-2 molecular entries
    """

    def __init__(self, reactant: List[MoleculeEntry], product: List[MoleculeEntry]):
        """
            Initilizes IntermolecularReaction.reactant to be in the form of a MolecularEntry,
            IntermolecularReaction.product to be in the form of [MolecularEntry_0, MolecularEntry_1],
            Reaction.reactant to be in the form of a of a list of MolecularEntry of length 1
            Reaction.products to be in the form of a of a list of MolecularEntry of length 2
          Args:
            :param reactant MolecularEntry object
            :param product list of MolecularEntry object of length 2
        """

        self.reactants = reactant
        self.products = product
        self.electron_free_energy = None
        self.electron_energy = None
        super().__init__(reactant, product)

    def graph_representation(self) -> nx.DiGraph:  # temp here, use graph_rep_1_2 instead

        """
            A method to convert a Concerted class object into graph representation (nx.Digraph object).
            IntermolecularReaction must be of type 1 reactant -> 2 products
            :return nx.Digraph object of a single IntermolecularReaction object
        """
        if len(self.reactants) == len(self.products) == 1:
            return graph_rep_1_1(self)
        elif len(self.reactants) == 1 and len(self.products) == 2:
            return graph_rep_1_2(self)
        elif len(self.reactants) == 2 and len(self.products) == 1:
            self.reactants, self.products = self.products, self.reactants
            return graph_rep_1_2(self)
        elif len(self.reactants) == len(self.products) == 2:
            return graph_rep_2_2(self)

    @classmethod
    def generate(cls, entries_list: [MoleculeEntry], name="nothing", read_file=True, num_processors=16, reaction_type="break2_form2", allowed_charge_change=0, restart=False) -> List[Reaction]:

        """
           A method to generate all the possible concerted reactions from given entries_list.
           Args:
              :param entries_list, entries_list = [MoleculeEntry]
              :param name(str): The name to put in FindConcertedReactions class. For reading in the files generated from that class.
              :param read_file(bool): whether to read in the file generated from the FindConcertedReactions class.
                                     If true, name+'_concerted_rxns.json' has to be present in the running directory.
                                     If False, will find concerted reactions on the fly.
                                     Note that this will take a couple hours when running on 16 CPU with < 100 entries.
              :param num_processors:
              :param reaction_type: Can choose from "break2_form2" and "break1_form1"
              :param allowed_charge_change: How many charge changes are allowed in a concerted reaction.
                          If zero, sum(reactant total charges) = sun(product total charges). If n(non-zero), allow n-electron redox reactions.
              :param restart (bool): whether load already determined concerted reactions or not. If a job was killed prematurely for
                             this function, restart can be called to save the effort.
              :return list of IntermolecularReaction class objects
        """
        if read_file:
            all_concerted_reactions = loadfn(name+'_concerted_rxns.json')
        else:
            from pymatgen.analysis.reaction_network.extract_reactions import FindConcertedReactions
            FCR = FindConcertedReactions(entries_list, name)
            all_concerted_reactions = FCR.get_final_concerted_reactions(name, num_processors, reaction_type, restart=restart)

        reactions = []
        for reaction in all_concerted_reactions:
            reactants = reaction[0].split("_")
            products = reaction[1].split("_")
            entries0 = [entries_list[int(item)] for item in reactants]
            entries1 = [entries_list[int(item)] for item in products]
            reactant_total_charge = np.sum([item.charge for item in entries0])
            product_total_charge = np.sum([item.charge for item in entries1])
            total_charge_change = product_total_charge - reactant_total_charge
            if abs(total_charge_change) <= allowed_charge_change:
                r = cls(entries0,entries1)
                reactions.append(r)

        return reactions

    def reaction_type(self) -> Mapping_ReactionType_Dict:

        """
           A method to identify type of intermoleular reaction (bond decomposition from one to two or formation from two to one molecules)
           Args:
              :return dictionary of the form {"class": "IntermolecularReaction", "rxn_type_A": rxn_type_A, "rxn_type_B": rxn_type_B}
              where rnx_type_A is the primary type of the reaction based on the reactant and product of the IntermolecularReaction
              object, and the backwards of this reaction would be rnx_type_B
        """

        rxn_type_A = "Concerted"
        rxn_type_B = "Concerted"

        reaction_type = {"class": "ConcertedReaction", "rxn_type_A": rxn_type_A, "rxn_type_B": rxn_type_B}
        return reaction_type

    def free_energy(self) -> Mapping_Energy_Dict:
        """
          A method to determine the free energy of the concerted reaction
          Args:
             :return dictionary of the form {"free_energy_A": energy_A, "free_energy_B": energy_B}
             where free_energy_A is the primary type of the reaction based on the reactant and product of the ConcertedReaction
             object, and the backwards of this reaction would be free_energy_B.
         """
        if all(reactant.free_energy != None for reactant in self.reactants) and all(product.free_energy != None for product in self.products):
            reactant_total_charge = np.sum([item.charge for item in self.reactants])
            product_total_charge = np.sum([item.charge for item in self.products])
            reactant_total_free_energy = np.sum([item.free_energy for item in self.reactants])
            product_total_free_energy = np.sum([item.free_energy for item in self.products])
            total_charge_change = product_total_charge - reactant_total_charge
            free_energy_A = product_total_free_energy - reactant_total_free_energy + total_charge_change * self.electron_free_energy
            free_energy_B = reactant_total_free_energy - product_total_free_energy - total_charge_change * self.electron_free_energy

        else:
            free_energy_A = None
            free_energy_B = None

        return {"free_energy_A": free_energy_A, "free_energy_B": free_energy_B}

    def energy(self) -> Mapping_Energy_Dict:
        """
          A method to determine the energy of the concerted reaction
          Args:
             :return dictionary of the form {"energy_A": energy_A, "energy_B": energy_B}
             where energy_A is the primary type of the reaction based on the reactant and product of the ConcertedReaction
             object, and the backwards of this reaction would be energy_B.
             Electron electronic energy set to 0 for now.
        """
        if all(reactant.energy != None for reactant in self.reactants) and all(
                product.energy != None for product in self.products):
            reactant_total_charge = np.sum([item.charge for item in self.reactants])
            product_total_charge = np.sum([item.charge for item in self.products])
            reactant_total_energy = np.sum([item.energy for item in self.reactants])
            product_total_energy = np.sum([item.energy for item in self.products])
            total_charge_change = product_total_charge - reactant_total_charge
            energy_A = product_total_energy - reactant_total_energy #+ total_charge_change * self.electron_energy
            energy_B = reactant_total_energy - product_total_energy #- total_charge_change * self.electron_energy

        else:
            energy_A = None
            energy_B = None

        return {"energy_A": energy_A, "energy_B": energy_B}

    def rate(self):
        pass


class ReactionPath(MSONable):
    """
        A class to define path object within the reaction network which constains all the associated characteristic attributes of a given path
        :param path - a list of nodes that defines a path from node A to B within a graph built using ReactionNetwork.build()
    """

    def __init__(self, path):
        """
        initializes the ReactionPath object attributes for a given path
        :param path: a list of nodes that defines a path from node A to B within a graph built using ReactionNetwork.build()
        """

        self.path = path
        self.byproducts = []
        self.unsolved_prereqs = []
        self.solved_prereqs = []
        self.all_prereqs = []
        self.cost = 0.0
        self.overall_free_energy_change = 0.0
        self.hardest_step = None
        self.description = ""
        self.pure_cost = 0.0
        self.full_path = None
        self.hardest_step_deltaG = None
        self.path_dict = {"byproducts": self.byproducts, "unsolved_prereqs": self.unsolved_prereqs,
                          "solved_prereqs": self.solved_prereqs, "all_prereqs": self.all_prereqs, "cost": self.cost,
                          "path": self.path, "overall_free_energy_change": self.overall_free_energy_change,
                          "hardest_step": self.hardest_step, "description": self.description,
                          "pure_cost": self.pure_cost,
                          "hardest_step_deltaG": self.hardest_step_deltaG, "full_path": self.full_path}

    @property
    def as_dict(self) -> dict:
        """
            A method to convert ReactionPath objection into a dictionary
        :return: d: dictionary containing all te ReactionPath attributes
        """
        d = {"@module": self.__class__.__module__,
             "@class": self.__class__.__name__,
             "byproducts": self.byproducts,
             "unsolved_prereqs": self.unsolved_prereqs,
             "solved_prereqs": self.solved_prereqs,
             "all_prereqs": self.all_prereqs,
             "cost": self.cost,
             "path": self.path,
             "overall_free_energy_change": self.overall_free_energy_change,
             "hardest_step": self.hardest_step,
             "description": self.description,
             "pure_cost": self.pure_cost,
             "hardest_step_deltaG": self.hardest_step_deltaG,
             "full_path": self.full_path,
             "path_dict": self.path_dict
             }
        return d

    @classmethod
    def from_dict(cls, d):
        """
            A method to convert dict to ReactionPath object
        :param d:  dict retuend from ReactionPath.as_dict() method
        :return: ReactionPath object
        """
        x = cls(d.get("path"))
        x.byproducts = d.get("byproducts")
        x.unsolved_prereqs = d.get("unsolved_prereqs")
        x.solved_prereqs = d.get("solved_prereqs")
        x.all_prereqs = d.get("all_prereqs")
        x.cost = d.get("cost", 0)

        x.overall_free_energy_change = d.get("overall_free_energy_change", 0)
        x.hardest_step = d.get("hardest_step")
        x.description = d.get("description")
        x.pure_cost = d.get("pure_cost", 0)
        x.hardest_step_deltaG = d.get("hardest_step_deltaG")
        x.full_path = d.get("full_path")
        x.path_dict = d.get("path_dict")

        return x

    @classmethod
    def characterize_path(cls, path: List[str], weight: str, min_cost: Dict[str, float], graph: nx.DiGraph,
                          PR_paths=[]):  # -> ReactionPath
        """
            A method to define ReactionPath attributes based on the inputs
        :param path: a list of nodes that defines a path from node A to B within a graph built using ReactionNetwork.build()
        :param weight: string (either "softplus" or "exponent")
        :param min_cost: dict with minimum cost from path start to a node, of from {node: float}
        :param graph: nx.Digraph
        :param PR_paths: list of already solved PRs
        :return: ReactionPath object
        """

        if path is None:
            class_instance = cls(None)
        else:
            class_instance = cls(path)
            for ii, step in enumerate(path):
                if ii != len(path) - 1:
                    class_instance.cost += graph[step][path[ii + 1]][weight]
                    if ii % 2 == 1:
                        rxn = step.split(",")
                        if "+PR_" in rxn[0]:
                            PR = int(rxn[0].split("+PR_")[1])
                            class_instance.all_prereqs.append(PR)
                        if "+" in rxn[1]:
                            desired_prod_satisfied = False
                            prods = rxn[1].split("+")
                            for prod in prods:
                                if int(prod) != path[ii + 1]:
                                    class_instance.byproducts.append(int(prod))
                                elif desired_prod_satisfied:
                                    class_instance.byproducts.append(int(prod))
                                else:
                                    desired_prod_satisfied = True
            for PR in class_instance.all_prereqs:
                if PR in class_instance.byproducts:
                    # Note that we're ignoring the order in which BPs are made vs they come up as PRs...
                    class_instance.all_prereqs.remove(PR)
                    class_instance.byproducts.remove(PR)

                    if PR in min_cost:
                        class_instance.cost -= min_cost[PR]
                    else:
                        print("Missing PR cost to remove:", PR)
            for PR in class_instance.all_prereqs:
                if str(PR) in PR_paths or PR in PR_paths: # XX
                    class_instance.solved_prereqs.append(PR)
                else:
                    class_instance.unsolved_prereqs.append(PR)

            class_instance.path_dict = {"byproducts": class_instance.byproducts,
                                        "unsolved_prereqs": class_instance.unsolved_prereqs,
                                        "solved_prereqs": class_instance.solved_prereqs,
                                        "all_prereqs": class_instance.all_prereqs, "cost": class_instance.cost,
                                        "path": class_instance.path,
                                        "overall_free_energy_change": class_instance.overall_free_energy_change,
                                        "hardest_step": class_instance.hardest_step,
                                        "description": class_instance.description,
                                        "pure_cost": class_instance.pure_cost,
                                        "hardest_step_deltaG": class_instance.hardest_step_deltaG,
                                        "full_path": class_instance.full_path}
        return class_instance

    @classmethod
    def characterize_path_final(cls, path: List[str], weight: str, min_cost: Dict[str, float], graph: nx.DiGraph,
                                PR_paths):  # Mapping_PR_Dict): -> ReactionPath
        """
            A method to define all the attributes of a given path once all the PRs are solved
        :param path: a list of nodes that defines a path from node A to B within a graph built using ReactionNetwork.build()
        :param weight: string (either "softplus" or "exponent")
        :param min_cost: dict with minimum cost from path start to a node, of from {node: float},
        if no path exist, value is "no_path", if path is unsolved yet, value is "unsolved_path"
        :param graph: nx.Digraph
        :param PR_paths: dict that defines a path from each node to a start,
               of the form {int(node1): {int(start1}: {ReactionPath object}, int(start2): {ReactionPath object}}, int(node2):...}
        :return: ReactionPath object
        """

        class_instance = cls.characterize_path(path, weight, min_cost, graph, PR_paths)
        if path is None:
            class_instance = cls(None)
        else:
            assert (len(class_instance.solved_prereqs) == len(class_instance.all_prereqs))
            assert (len(class_instance.unsolved_prereqs) == 0)

            PRs_to_join = copy.deepcopy(class_instance.all_prereqs)
            full_path = copy.deepcopy(path)
            while len(PRs_to_join) > 0:
                new_PRs = []
                for PR in PRs_to_join:
                    PR_path = None
                    PR_min_cost = 1000000000000000.0
                    for start in PR_paths[PR]:
                        if PR_paths[PR][start].path != None:
                            # print(PR_paths[PR][start].path_dict)
                            # print(PR_paths[PR][start].cost, PR_paths[PR][start].overall_free_energy_change, PR_paths[PR][start].path)
                            if PR_paths[PR][start].cost < PR_min_cost:
                                PR_min_cost = PR_paths[PR][start].cost
                                PR_path = PR_paths[PR][start]
                    assert (len(PR_path.solved_prereqs) == len(PR_path.all_prereqs))
                    for new_PR in PR_path.all_prereqs:
                        new_PRs.append(new_PR)
                        class_instance.all_prereqs.append(new_PR)
                    for new_BP in PR_path.byproducts:
                        class_instance.byproducts.append(new_BP)
                    full_path = PR_path.path + full_path
                PRs_to_join = copy.deepcopy(new_PRs)

            for PR in class_instance.all_prereqs:
                if PR in class_instance.byproducts:
                    print("WARNING: Matching prereq and byproduct found!", PR)

            for ii, step in enumerate(full_path):
                if graph.nodes[step]["bipartite"] == 1:
                    if weight == "softplus":
                        class_instance.pure_cost += ReactionNetwork.softplus(graph.nodes[step]["free_energy"])
                    elif weight == "exponent":
                        class_instance.pure_cost += ReactionNetwork.exponent(graph.nodes[step]["free_energy"])

                    class_instance.overall_free_energy_change += graph.nodes[step]["free_energy"]

                    if class_instance.description == "":
                        class_instance.description += graph.nodes[step]["rxn_type"]
                    else:
                        class_instance.description += ", " + graph.nodes[step]["rxn_type"]

                    if class_instance.hardest_step is None:
                        class_instance.hardest_step = step
                    elif graph.nodes[step]["free_energy"] > graph.nodes[class_instance.hardest_step]["free_energy"]:
                        class_instance.hardest_step = step

            class_instance.full_path = full_path

            if class_instance.hardest_step is None:
                class_instance.hardest_step_deltaG = None
            else:
                class_instance.hardest_step_deltaG = graph.nodes[class_instance.hardest_step]["free_energy"]

        class_instance.path_dict = {"byproducts": class_instance.byproducts,
                                    "unsolved_prereqs": class_instance.unsolved_prereqs,
                                    "solved_prereqs": class_instance.solved_prereqs,
                                    "all_prereqs": class_instance.all_prereqs, "cost": class_instance.cost,
                                    "path": class_instance.path,
                                    "overall_free_energy_change": class_instance.overall_free_energy_change,
                                    "hardest_step": class_instance.hardest_step,
                                    "description": class_instance.description, "pure_cost": class_instance.pure_cost,
                                    "hardest_step_deltaG": class_instance.hardest_step_deltaG,
                                    "full_path": class_instance.full_path}

        return class_instance


Mapping_PR_Dict = Dict[int, Dict[int, ReactionPath]]


class ReactionNetwork(MSONable):
    """
       Class to build a reaction network from entries
    """

    def __init__(self, input_entries: List[MoleculeEntry], electron_free_energy=-2.15):
        """
        Initilize the ReacitonNetwork object attributes
        :param input_entries: [MoleculeEntry]: List of MoleculeEntry objects
        :param electron_free_energy: The Gibbs free energy of an electron. Defaults to -2.15 eV, the value at which the LiEC SEI forms
        """

        self.graph = nx.DiGraph()
        self.reactions = []
        self.input_entries = input_entries
        self.entry_ids = {e.entry_id for e in self.input_entries}
        self.electron_free_energy = electron_free_energy
        self.entries = {}
        self.entries_list = []
        self.num_starts = 0
        self.weight = None
        self.PR_record = None
        self.Reactant_record = None
        self.min_cost = {}

        print(len(self.input_entries), "input entries")

        connected_entries = []
        for entry in self.input_entries:
            if len(entry.molecule) > 1:
                if nx.is_weakly_connected(entry.graph):
                    connected_entries.append(entry)
            else:
                connected_entries.append(entry)
        print(len(connected_entries), "connected entries")

        get_formula = lambda x: x.formula
        get_Nbonds = lambda x: x.Nbonds
        get_charge = lambda x: x.charge
        get_free_energy = lambda x: x.free_energy

        sorted_entries_0 = sorted(connected_entries, key=get_formula)
        for k1, g1 in itertools.groupby(sorted_entries_0, get_formula):
            sorted_entries_1 = sorted(list(g1), key=get_Nbonds)
            self.entries[k1] = {}

            for k2, g2 in itertools.groupby(sorted_entries_1, get_Nbonds):
                sorted_entries_2 = sorted(list(g2), key=get_charge)
                self.entries[k1][k2] = {}
                for k3, g3 in itertools.groupby(sorted_entries_2, get_charge):
                    sorted_entries_3 = list(g3)
                    sorted_entries_3.sort(key=get_free_energy)
                    if len(sorted_entries_3) > 1:
                        unique = []
                        for entry in sorted_entries_3:
                            isomorphic_found = False
                            for ii, Uentry in enumerate(unique):
                                if entry.mol_graph.isomorphic_to(Uentry.mol_graph):
                                    isomorphic_found = True
                                    # print("Isomorphic entries with equal charges found!")
                                    if entry.free_energy != None and Uentry.free_energy != None:
                                        if entry.free_energy < Uentry.free_energy:
                                            unique[ii] = entry
                                            # if entry.energy > Uentry.energy:
                                            #     print("WARNING: Free energy lower but electronic energy higher!")
                                    elif entry.free_energy != None:
                                        unique[ii] = entry
                                    elif entry.energy < Uentry.energy:
                                        unique[ii] = entry
                                    break
                            if not isomorphic_found:
                                unique.append(entry)
                        self.entries[k1][k2][k3] = unique
                    else:
                        self.entries[k1][k2][k3] = sorted_entries_3
                    for entry in self.entries[k1][k2][k3]:
                        self.entries_list.append(entry)

        print(len(self.entries_list), "unique entries")
        for ii, entry in enumerate(self.entries_list):
            if "ind" in entry.parameters.keys():
                pass
            else:
                entry.parameters["ind"] = ii

        self.entries_list = sorted(self.entries_list, key=lambda x: x.parameters["ind"])

    @staticmethod
    def softplus(free_energy: float) -> float:
        """
            Method to determine edge weight using softplus cost function
        :param free_energy: float
        :return: float
        """
        return float(np.log(1 + (273.0 / 500.0) * np.exp(free_energy)))

    @staticmethod
    def exponent(free_energy: float) -> float:
        """
            Method to determine edge weight using exponent cost function
        :param free_energy: float
        :return: float
        """
        return float(np.exp(free_energy))

    def build(self, reaction_types={"RedoxReaction", "IntramolSingleBondChangeReaction", "IntermolecularReaction",
                                    "CoordinationBondChangeReaction"}) -> nx.DiGraph:
        """
            A method to build the reaction network graph
        :param reaction_types: set of all the reactions class to include while building the graph
        :return: nx.DiGraph
        """

        self.graph.add_nodes_from(range(len(self.entries_list)), bipartite=0)
        reaction_types = [load_class(str(self.__module__) + "." + s) for s in reaction_types]
        self.reactions = [r.generate(self.entries) for r in reaction_types]
        self.reactions = [i for i in self.reactions if i]
        self.reactions = list(itertools.chain.from_iterable(self.reactions))

        for r in self.reactions:
            if r.reaction_type()["class"] == "RedoxReaction":
                r.electron_free_energy = self.electron_free_energy
            self.add_reaction(r.graph_representation())

        self.PR_record = self.build_PR_record()
        self.Reactant_record = self.build_reactant_record()

        return self.graph

    def build_concerted_reactions(self, name="nothing", read_file=True, num_processors=16, reaction_type="break2_form2", allowed_charge_change=0, restart=False) -> nx.DiGraph:
        """
            A method to refine the reaction network graph by adding concerted reactions.
            This has to be called after self.build, since concerted reactions also include elementary reactions.
            If a concerted reaction is already labeled as one of the elementary reaction types, then it will be removed.
        :return: nx.DiGraph
        """
        self.concerted_reactions = [ConcertedReaction.generate(self.entries_list,name, read_file, num_processors, reaction_type, allowed_charge_change, restart)]
        self.concerted_reactions = [i for i in self.concerted_reactions if i]
        self.concerted_reactions = list(itertools.chain.from_iterable(self.concerted_reactions))

        for r in self.concerted_reactions:
            r.electron_free_energy = self.electron_free_energy
            # Determine whether it's already labeled as elementary reaction. If so, not adding to the graph.
            if len(r.reactants) == len(r.products) == 1:
                node_name = str(r.reactants[0].parameters["ind"]) + "," + str(r.products[0].parameters["ind"])

            elif len(r.reactants) == 2 and len(r.products) == 1:
                reactant_0 = r.reactants[0]
                reactant_1 = r.reactants[1]
                if reactant_0.parameters["ind"] <= reactant_1.parameters["ind"]:
                    two_reac_name = str(reactant_0.parameters["ind"]) + "+" + str(reactant_1.parameters["ind"])
                else:
                    two_reac_name = str(reactant_1.parameters["ind"]) + "+" + str(reactant_0.parameters["ind"])
                node_name = str(r.products[0].parameters["ind"]) + "," + two_reac_name

            elif len(r.reactants) == 1 and len(r.products) == 2:
                product_0 = r.products[0]
                product_1 = r.products[1]
                if product_0.parameters["ind"] <= product_1.parameters["ind"]:
                    two_prod_name = str(product_0.parameters["ind"]) + "+" + str(product_1.parameters["ind"])
                else:
                    two_prod_name = str(product_1.parameters["ind"]) + "+" + str(product_0.parameters["ind"])
                node_name = str(r.reactants[0].parameters["ind"]) + "," + two_prod_name

            elif len(r.reactants) == 2 and len(r.products) == 2:
                reactant_0 = r.reactants[0]
                reactant_1 = r.reactants[1]
                product_0 = r.products[0]
                product_1 = r.products[1]
                if product_0.parameters["ind"] <= product_1.parameters["ind"]:
                    two_prod_name = str(product_0.parameters["ind"]) + "+" + str(product_1.parameters["ind"])
                else:
                    two_prod_name = str(product_1.parameters["ind"]) + "+" + str(product_0.parameters["ind"])
                two_reac_name0 = str(reactant_0.parameters["ind"]) + "+PR_" + str(reactant_1.parameters["ind"])
                node_name = two_reac_name0 + "," + two_prod_name

            if node_name not in self.graph.nodes:
                self.add_reaction(r.graph_representation())

        self.PR_record = self.build_PR_record()
        self.Reactant_record = self.build_reactant_record()

        return self.graph


    def add_reaction(self, graph_representation: nx.DiGraph):
        """
            A method to add a single reaction to the ReactionNetwork.graph attribute
        :param graph_representation: Graph representation of a reaction, obtained from ReactionClass.graph_representation
        """
        self.graph.add_nodes_from(graph_representation.nodes(data=True))
        self.graph.add_edges_from(graph_representation.edges(data=True))

    def build_PR_record(self) -> Mapping_Record_Dict:
        """
            A method to determine all the reaction nodes that have a the same PR in the ReactionNetwork.graph
        :return: a dict of the form {int(node1): [all the reaction nodes with PR of node1, ex "2+PR_node1, 3"]}
        """
        PR_record = {}
        for node in self.graph.nodes():
            if self.graph.nodes[node]["bipartite"] == 0:
                PR_record[node] = []
        for node in self.graph.nodes():
            if self.graph.nodes[node]["bipartite"] == 1:
                if "+PR_" in node.split(",")[0]:
                    PR = int(node.split(",")[0].split("+PR_")[1])
                    PR_record[PR].append(node)
        return PR_record

    def build_reactant_record(self) -> Mapping_Record_Dict:
        """
            A method to determine all the reaction nodes that have the same non PR reactant node in the ReactionNetwork.graph
        :return: a dict of the form {int(node1): [all the reaction nodes with non PR reactant of node1, ex "node1+PR_2, 3"]}
        """
        Reactant_record = {}
        for node in self.graph.nodes():
            if self.graph.nodes[node]["bipartite"] == 0:
                Reactant_record[node] = []
        for node in self.graph.nodes():
            if self.graph.nodes[node]["bipartite"] == 1:
                non_PR_reactant = node.split(",")[0].split("+PR_")[0]
                Reactant_record[int(non_PR_reactant)].append(node)
        return Reactant_record

    def solve_prerequisites(self, starts: List[int], target: int, weight: str, max_iter=20, save=False,
                            filename=None):  # -> Tuple[Union[Dict[Union[int, Any], dict], Any], Any]:
        """
            A method to solve the all the prerequisites found in ReactionNetwork.graph. By solving all PRs, it gives
            information on whether 1. if a path exist from any of the starts to all other molecule nodes, 2. if so what
            is the min cost to reach that node from any of the start, 3. if there is no path from any of the starts to a
            any of the molecule node, 4. for molecule nodes where the path exist, characterize the in the form of ReactionPath
        :param starts: List(molecular nodes), list of molecular nodes of type int found in the ReactionNetwork.graph
        :param target: a single molecular node of type int found in the ReactionNetwork.graph
        :param weight: "softplus" or "exponent", type of cost function to use when calculating edge weights
        :param max_iter: maximum number of iterations to try to solve all the PRs
        :return: PRs: PR_paths: dict that defines a path from each node to a start,
                of the form {int(node1): {int(start1}: {ReactionPath object}, int(start2): {ReactionPath object}}, int(node2):...}
        :return: min_cost: dict with minimum cost from path start to a node, of from {node: float},
                if no path exist, value is "no_path", if path is unsolved yet, value is "unsolved_path"
        :return: graph: ReactionNetwork.graph of type nx.DiGraph with updated edge weights based on solved PRs
        """
        PRs = {}
        old_solved_PRs = []
        new_solved_PRs = ["placeholder"]
        old_attrs = {}
        new_attrs = {}
        self.weight = weight
        self.num_starts = len(starts)

        if len(self.graph.nodes) == 0:
            self.build()
        orig_graph = copy.deepcopy(self.graph)

        for start in starts:
            PRs[start] = {}

        for PR in PRs:
            for start in starts:
                if start == PR:
                    PRs[PR][start] = ReactionPath.characterize_path([start], weight, self.min_cost, self.graph)
                else:
                    PRs[PR][start] = ReactionPath(None)

            old_solved_PRs.append(PR)
            self.min_cost[PR] = PRs[PR][PR].cost
        for node in self.graph.nodes():
            if self.graph.nodes[node]["bipartite"] == 0 and node != target:
                if node not in PRs:
                    PRs[node] = {}

        ii = 0

        while (len(new_solved_PRs) > 0 or old_attrs != new_attrs) and ii < max_iter:
            min_cost = {}
            cost_from_start = {}
            self.unsolved_PRs = {} # modified by XX. Added a unsolved_PR dict to keep track of unsolved.
            for PR in PRs:
                cost_from_start[PR] = {}
                min_cost[PR] = 10000000000000000.0
                for start in PRs[PR]:
                    if PRs[PR][start].path == None:
                        cost_from_start[PR][start] = "no_path"
                    else:
                        cost_from_start[PR][start] = PRs[PR][start].cost
                        if PRs[PR][start].cost < min_cost[PR]:
                            min_cost[PR] = PRs[PR][start].cost
                for start in starts:
                    if start not in cost_from_start[PR]:
                        cost_from_start[PR][start] = "unsolved"

            PRs, cost_from_start, min_cost = self.find_path_cost(starts, target, weight, old_solved_PRs,
                                                                 cost_from_start, min_cost, PRs)
            solved_PRs = copy.deepcopy(old_solved_PRs)
            solved_PRs, new_solved_PRs, cost_from_start = self.identify_solved_PRs(PRs, solved_PRs, cost_from_start)

            print(ii, len(solved_PRs), len(new_solved_PRs), len(self.unsolved_PRs))
            # modified by XX. Printing (1) iteration index,
            # (2) number of total solved entries till this iteration, (3) number of newly solved entries at this iteration,
            # (4) number of unsolved entries. (2) and (4) should add up to the number of total unique entries.

            attrs = self.update_edge_weights(min_cost, orig_graph)

            self.min_cost = copy.deepcopy(min_cost)
            old_solved_PRs = copy.deepcopy(solved_PRs)
            old_attrs = copy.deepcopy(new_attrs)
            new_attrs = copy.deepcopy(attrs)

            ii += 1

        self.final_PR_check(PRs)
        if save:
            if filename is None:
                print("Provide filename to save the PRs, for now saving as PRs.json")
                filename = "PRs.json"
            dumpfn(PRs, filename, default=lambda o: o.as_dict)
            dumpfn(self.unsolved_PRs, 'unsolved_PRs.json',default=lambda o: o.as_dict)
            dumpfn(json_graph.adjacency_data(self.graph),'RN_graph.json')
            dumpfn(self.min_cost, 'min_cost.json', default=lambda o: o.as_dict)
        return PRs

    def solve_prerequisites_no_target(self, starts: List[int], weight: str, max_iter=20, save=False,
                            filename=None):  # -> Tuple[Union[Dict[Union[int, Any], dict], Any], Any]:
        """
            A method to solve the all the prerequisites found in ReactionNetwork.graph. By solving all PRs, it gives
            information on whether 1. if a path exist from any of the starts to all other molecule nodes, 2. if so what
            is the min cost to reach that node from any of the start, 3. if there is no path from any of the starts to a
            any of the molecule node, 4. for molecule nodes where the path exist, characterize the in the form of ReactionPath
        :param starts: List(molecular nodes), list of molecular nodes of type int found in the ReactionNetwork.graph
        :param weight: "softplus" or "exponent", type of cost function to use when calculating edge weights
        :param max_iter: maximum number of iterations to try to solve all the PRs
        :return: PRs: PR_paths: dict that defines a path from each node to a start,
                of the form {int(node1): {int(start1}: {ReactionPath object}, int(start2): {ReactionPath object}}, int(node2):...}
        :return: min_cost: dict with minimum cost from path start to a node, of from {node: float},
                if no path exist, value is "no_path", if path is unsolved yet, value is "unsolved_path"
        :return: graph: ReactionNetwork.graph of type nx.DiGraph with updated edge weights based on solved PRs
        """
        PRs = {}
        old_solved_PRs = []
        new_solved_PRs = ["placeholder"]
        old_attrs = {}
        new_attrs = {}
        self.weight = weight
        self.num_starts = len(starts)

        if len(self.graph.nodes) == 0:
            self.build()
        orig_graph = copy.deepcopy(self.graph)

        for start in starts:
            PRs[start] = {}

        for PR in PRs:
            for start in starts:
                if start == PR:
                    PRs[PR][start] = ReactionPath.characterize_path([start], weight, self.min_cost, self.graph)
                else:
                    PRs[PR][start] = ReactionPath(None)

            old_solved_PRs.append(PR)
            self.min_cost[PR] = PRs[PR][PR].cost
        for node in self.graph.nodes():
            if self.graph.nodes[node]["bipartite"] == 0: #and node != target:
                if node not in PRs:
                    PRs[node] = {}

        ii = 0

        while (len(new_solved_PRs) > 0 or old_attrs != new_attrs) and ii < max_iter:
            min_cost = {}
            cost_from_start = {}
            self.unsolved_PRs = {} # modified by XX. Added a unsolved_PR dict to keep track of unsolved.
            for PR in PRs:
                cost_from_start[PR] = {}
                min_cost[PR] = 10000000000000000.0
                for start in PRs[PR]:
                    if PRs[PR][start].path == None:
                        cost_from_start[PR][start] = "no_path"
                    else:
                        cost_from_start[PR][start] = PRs[PR][start].cost
                        if PRs[PR][start].cost < min_cost[PR]:
                            min_cost[PR] = PRs[PR][start].cost
                for start in starts:
                    if start not in cost_from_start[PR]:
                        cost_from_start[PR][start] = "unsolved"

            PRs, cost_from_start, min_cost = self.find_path_cost_no_target(starts, weight, old_solved_PRs,
                                                                 cost_from_start, min_cost, PRs)
            solved_PRs = copy.deepcopy(old_solved_PRs)
            solved_PRs, new_solved_PRs, cost_from_start = self.identify_solved_PRs(PRs, solved_PRs, cost_from_start)

            print(ii, len(solved_PRs), len(new_solved_PRs), len(self.unsolved_PRs))
            # modified by XX. Printing (1) iteration index,
            # (2) number of total solved entries till this iteration, (3) number of newly solved entries at this iteration,
            # (4) number of unsolved entries. (2) and (4) should add up to the number of total unique entries.

            attrs = self.update_edge_weights(min_cost, orig_graph)

            self.min_cost = copy.deepcopy(min_cost)
            old_solved_PRs = copy.deepcopy(solved_PRs)
            old_attrs = copy.deepcopy(new_attrs)
            new_attrs = copy.deepcopy(attrs)

            ii += 1

        self.final_PR_check(PRs)
        if save:
            if filename is None:
                print("Provide filename to save the PRs, for now saving as PRs.json")
                filename = "PRs.json"
            dumpfn(PRs, filename, default=lambda o: o.as_dict)
            dumpfn(self.unsolved_PRs, 'unsolved_PRs.json',default=lambda o: o.as_dict)
            dumpfn(json_graph.adjacency_data(self.graph),'RN_graph.json')
            dumpfn(self.min_cost, 'min_cost.json', default=lambda o: o.as_dict)
        return PRs

    def find_path_cost(self, starts, target, weight, old_solved_PRs, cost_from_start, min_cost, PRs):
        """
            A method to characterize the path to all the PRs. Characterize by determining if the path exist or not, and
            if so, is it a minimum cost path, and if so set PRs[node][start] = ReactionPath(path)
        :param starts: List(molecular nodes), list of molecular nodes of type int found in the ReactionNetwork.graph
        :param target: a single molecular node of type int found in the ReactionNetwork.graph
        :param weight: "softplus" or "exponent", type of cost function to use when calculating edge weights
        :param old_solved_PRs: list of PRs (molecular nodes of type int) that are already solved
        :param cost_from_start: dict of type {node1: {start1: float, start2: float}, node2: {...}}
        :param min_cost: dict with minimum cost from path start to a node, of from {node: float},
                if no path exist, value is "no_path", if path is unsolved yet, value is "unsolved_path"
        :param PRs: dict that defines a path from each node to a start, of the form {int(node1):
                {int(start1}: {ReactionPath object}, int(start2): {ReactionPath object}}, int(node2):...}
        :return: PRs: updated PRs based on new PRs solved
        :return: cost_from_start: updated cost_from_start based on new PRs solved
        :return: min_cost: updated min_cost based on new PRs solved
        """
        ## Below modified by XX. Keep a record of entries cannot be reached from starting materials.
        # When doing dijkstra algorithm, we need to remove those nodes.
        self.not_reachable_nodes = []
        for PR in PRs:
            reachable = False
            if all(start in PRs[PR].keys() for start in starts):
                for start in starts:
                    if PRs[PR][start].path != None:
                        reachable = True
            else:
                reachable = True
            if not reachable:
                self.not_reachable_nodes.append(PR)
        print('not reachable nodes:', self.not_reachable_nodes)
        ## XX modification ends
        self.num_starts = len(starts)
        for node in self.graph.nodes():
            if self.graph.nodes[node]["bipartite"] == 0 and node not in old_solved_PRs and node != target:
                self.unsolved_PRs[node] = {}  # modified by XX. Added a dict to keep track of unsolved cases.
                for start in starts:
                    if start not in PRs[node]:
                        path_exists = True
                        try:
                            length, dij_path = nx.algorithms.simple_paths._bidirectional_dijkstra(
                                self.graph,
                                source=hash(start),
                                target=hash(node),
                                ignore_nodes=self.find_or_remove_bad_nodes([node, target]+self.not_reachable_nodes),
                                weight=self.weight)
                        except nx.exception.NetworkXNoPath:
                            PRs[node][start] = ReactionPath(None)
                            path_exists = False
                            cost_from_start[node][start] = "no_path"
                        if path_exists:
                            path_class = ReactionPath.characterize_path(dij_path, weight, self.min_cost, self.graph,
                                                                        old_solved_PRs)
                            cost_from_start[node][start] = path_class.cost
                            if len(path_class.unsolved_prereqs) == 0:
                                PRs[node][start] = path_class
                            else:
                                self.unsolved_PRs[node][start] = path_class  # modified by XX. If unsolved, still add the path to the unsolved dict to keep a record.
                            if path_class.cost < min_cost[node]:
                                min_cost[node] = path_class.cost

        return PRs, cost_from_start, min_cost

    def find_path_cost_no_target(self, starts, weight, old_solved_PRs, cost_from_start, min_cost, PRs):
        """
            A method to characterize the path to all the PRs. Characterize by determining if the path exist or not, and
            if so, is it a minimum cost path, and if so set PRs[node][start] = ReactionPath(path)
        :param starts: List(molecular nodes), list of molecular nodes of type int found in the ReactionNetwork.graph
        :param weight: "softplus" or "exponent", type of cost function to use when calculating edge weights
        :param old_solved_PRs: list of PRs (molecular nodes of type int) that are already solved
        :param cost_from_start: dict of type {node1: {start1: float, start2: float}, node2: {...}}
        :param min_cost: dict with minimum cost from path start to a node, of from {node: float},
                if no path exist, value is "no_path", if path is unsolved yet, value is "unsolved_path"
        :param PRs: dict that defines a path from each node to a start, of the form {int(node1):
                {int(start1}: {ReactionPath object}, int(start2): {ReactionPath object}}, int(node2):...}
        :return: PRs: updated PRs based on new PRs solved
        :return: cost_from_start: updated cost_from_start based on new PRs solved
        :return: min_cost: updated min_cost based on new PRs solved
        """
        ## Below modified by XX. Keep a record of entries cannot be reached from starting materials.
        # When doing dijkstra algorithm, we need to remove those nodes.
        self.not_reachable_nodes = []
        for PR in PRs:
            reachable = False
            if all(start in PRs[PR].keys() for start in starts):
                for start in starts:
                    if PRs[PR][start].path != None:
                        reachable = True
            else:
                reachable = True
            if not reachable:
                self.not_reachable_nodes.append(PR)
        print('not reachable nodes:', self.not_reachable_nodes)
        ## XX modification ends
        self.num_starts = len(starts)
        for node in self.graph.nodes():
            if self.graph.nodes[node]["bipartite"] == 0 and node not in old_solved_PRs: #and node != target:
                self.unsolved_PRs[node] = {}  # modified by XX. Added a dict to keep track of unsolved cases.
                for start in starts:
                    if start not in PRs[node]:
                        path_exists = True
                        try:
                            length, dij_path = nx.algorithms.simple_paths._bidirectional_dijkstra(
                                self.graph,
                                source=hash(start),
                                target=hash(node),
                                ignore_nodes=self.find_or_remove_bad_nodes([node]+self.not_reachable_nodes),
                                weight=self.weight)
                        except nx.exception.NetworkXNoPath:
                            PRs[node][start] = ReactionPath(None)
                            path_exists = False
                            cost_from_start[node][start] = "no_path"
                        if path_exists:
                            path_class = ReactionPath.characterize_path(dij_path, weight, self.min_cost, self.graph,
                                                                        old_solved_PRs)
                            cost_from_start[node][start] = path_class.cost
                            if len(path_class.unsolved_prereqs) == 0:
                                PRs[node][start] = path_class
                            else:
                                self.unsolved_PRs[node][start] = path_class  # modified by XX. If unsolved, still add the path to the unsolved dict to keep a record.
                            if path_class.cost < min_cost[node]:
                                min_cost[node] = path_class.cost

        return PRs, cost_from_start, min_cost

    def identify_solved_PRs(self, PRs, solved_PRs, cost_from_start):
        """
            A method to identify new solved PRs after each iteration
        :param PRs: dict that defines a path from each node to a start, of the form {int(node1):
                {int(start1}: {ReactionPath object}, int(start2): {ReactionPath object}}, int(node2):...}
        :param solved_PRs: list of PRs (molecular nodes of type int) that are already solved
        :param cost_from_start: dict of type {node1: {start1: float, start2: float}, node2: {...}}
        :return: solved_PRs: list of all the PRs(molecular nodes of type int) that are already solved plus new PRs solved in the current iteration
        :return: new_solved_PRs: list of just the new PRs(molecular nodes of type int) solved during current iteration
        :return: cost_from_start: updated dict of cost_from_start based on the new PRs solved during current iteration
        """
        new_solved_PRs = []
        for PR in PRs:
            if PR not in solved_PRs:
                if len(PRs[PR].keys()) == self.num_starts:
                    solved_PRs.append(PR)
                    new_solved_PRs.append(PR)
                    self.unsolved_PRs.pop(PR, None)  # modified by XX. Removing key from unsolved dict if it's solved.
                else:
                    best_start_so_far = [None, 10000000000000000.0]
                    for start in PRs[PR]:
                        if PRs[PR][start] is not None:  # ALWAYS TRUE
                            # if PRs[PR][start] == "unsolved": #### DOES THIS EVER HAPPEN ---- NEED TO FIX
                            #     print("ERROR: unsolved should never be encountered here!")
                            if PRs[PR][start].cost < best_start_so_far[1]:
                                best_start_so_far[0] = start
                                best_start_so_far[1] = PRs[PR][start].cost
                    if best_start_so_far[0] is not None:
                        num_beaten = 0
                        for start in cost_from_start[PR]:
                            if start != best_start_so_far[0]:
                                if cost_from_start[PR][start] == "no_path":
                                    num_beaten += 1
                                elif cost_from_start[PR][start] >= best_start_so_far[1]: ## modified by XX. changed '>' to '>='.
                                    num_beaten += 1
                        if num_beaten == self.num_starts - 1:
                            solved_PRs.append(PR)
                            new_solved_PRs.append(PR)
                            self.unsolved_PRs.pop(PR,None)  # modified by XX. Removing key from unsolved dict if it's solved.

        return solved_PRs, new_solved_PRs, cost_from_start

    def update_edge_weights(self, min_cost: Dict[int, float], orig_graph: nx.DiGraph) -> Dict[Tuple[int, str], Dict[
        str, float]]:  # , solved_PRs: List[int], new_attrs:Dict[Tuple[int, str],Dict[str,float]]):
        """
            A method to update the ReactionNetwork.graph edge weights based on the new cost of solving PRs
        :param min_cost: dict with minimum cost from path start to a node, of from {node: float},
                if no path exist, value is "no_path", if path is unsolved yet, value is "unsolved_path"
        :param orig_graph: ReactionNetwork.graph of type nx.Digraph before the start of current iteration of updates
        :return: attrs: dict of form {(node1, node2), {"softplus": float, "exponent": float, "weight: 1}, (node2, node3): {...}}
                dict of all the edges to update the weights of
        """
        if len(self.graph.nodes) == 0:
            self.graph = self.build()
        if self.PR_record is None:
            self.PR_record = self.build_PR_record()

        attrs = {}
        for PR_ind in min_cost:
            for rxn_node in self.PR_record[PR_ind]:
                non_PR_reactant_node = int(rxn_node.split(",")[0].split("+PR_")[0])
                attrs[(non_PR_reactant_node, rxn_node)] = {
                    self.weight: orig_graph[non_PR_reactant_node][rxn_node][self.weight] + min_cost[PR_ind]}
        nx.set_edge_attributes(self.graph, attrs)

        return attrs

    def final_PR_check(self, PRs: Mapping_PR_Dict):
        """
            A method to check errors in the path attributes of the PRs with a path, if no path then prints no path from any start to a given
        :param PRs: dict that defines a path from each node to a start, of the form {int(node1):
                {int(start1}: {ReactionPath object}, int(start2): {ReactionPath object}}, int(node2):...}
        """
        for PR in PRs:
            print("current PR:",PR)
            path_found = False
            if PRs[PR] != {}:
                for start in PRs[PR]:
                    if PRs[PR][start].path != None:
                        path_found = True
                        path_dict_class = ReactionPath.characterize_path_final(PRs[PR][start].path, self.weight,
                                                                               self.min_cost, self.graph, PRs)
                        if abs(path_dict_class.cost - path_dict_class.pure_cost) > 0.0001:
                            print("WARNING: cost mismatch for PR", PR, path_dict_class.cost, path_dict_class.pure_cost,
                                  path_dict_class.full_path)
                if not path_found:
                    print("No path found from any start to PR", PR)
            else:
                print("Unsolvable path from any start to PR", PR)

    def find_or_remove_bad_nodes(self, nodes: List[str], remove_nodes=False) -> List[str] or nx.DiGraph:
        """
            A method to either create a list of the nodes a path solving method should ignore or generate a graph without
            all the nodes it a path solving method should not use in obtaining a path.
        :param nodes: List(molecular nodes), list of molecular nodes of type int found in the ReactionNetwork.graph
        that should be ignored when solving a path
        :param remove_nodes: if False (default), return list of bad nodes, if True, return a version of
        ReactionNetwork.graph (of type nx.Digraph) from with list of bad nodes are removed
        :return: if remove_nodes = False -> list[node], if remove_nodes = True -> nx.DiGraph
        """
        if len(self.graph.nodes) == 0:
            self.graph = self.build()
        if self.PR_record is None:
            self.PR_record = self.build_PR_record()
        if self.Reactant_record is None:
            self.Reactant_record = self.build_reactant_record()
        bad_nodes = []
        for node in nodes:
            for bad_node in self.PR_record[node]:
                bad_nodes.append(bad_node)
            for bad_nodes2 in self.Reactant_record[node]:
                bad_nodes.append(bad_nodes2)
        if remove_nodes:
            pruned_graph = copy.deepcopy(self.graph)
            pruned_graph.remove_nodes_from(bad_nodes)
            return pruned_graph
        else:
            return bad_nodes

    def valid_shortest_simple_paths(self, start: int, target: int, PRs=[]):  # -> Generator[List[str]]:????
        """
            A method to determine shortest path from start to target
        :param start: molecular node of type int from ReactionNetwork.graph
        :param target: molecular node of type int from ReactionNetwork.graph
        :param PRs: not used currently?
        :return: nx.path_generator of type generator
        """
        bad_nodes = PRs
        bad_nodes.append(target)
        valid_graph = self.find_or_remove_bad_nodes(bad_nodes, remove_nodes=True)
        return nx.shortest_simple_paths(valid_graph, hash(start), hash(target), weight=self.weight)

    def find_paths(self, starts, target, weight, num_paths=10, solved_PRs_path=None):  # -> ??
        """
            A method to find the shorted path from given starts to a target
        :param starts: starts: List(molecular nodes), list of molecular nodes of type int found in the ReactionNetwork.graph
        :param target: a single molecular node of type int found in the ReactionNetwork.graph
        :param weight: "softplus" or "exponent", type of cost function to use when calculating edge weights
        :param num_paths: Number (of type int) of paths to find. Defaults to 10.
        :param solved_PRs_path: dict that defines a path from each node to a start,
                of the form {int(node1): {int(start1}: {ReactionPath object}, int(start2): {ReactionPath object}}, int(node2):...}
                if None, method will solve PRs
        :param solved_min_cost: dict with minimum cost from path start to a node, of from {node: float},
                if no path exist, value is "no_path", if path is unsolved yet, value is "unsolved_path",
                of None, method will solve for min_cost
        :param updated_graph: nx.DiGraph with udpated edge weights based on the solved PRs, if none, method will solve for PRs and update graph accordingly
        :param save: if True method will save PRs paths, min cost and updated graph after all the PRs are solved,
                    if False, method will not save anything (default)
        :return: PR_paths: solved dict of PRs
        :return: paths: list of paths (number of paths based on the value of num_paths)
        """

        self.weight = weight
        self.num_starts = len(starts)
        paths = []
        c = itertools.count()
        my_heapq = []
        print("Solving prerequisites...")
        if solved_PRs_path is None:
            self.min_cost = {}
            #self.graph = self.build() modified by XX
            PR_paths = self.solve_prerequisites(starts, target, weight, save=True)

        else:
            PR_paths = {}
            for key in solved_PRs_path:
                PR_paths[int(key)] = {}
                for start in solved_PRs_path[key]:
                    PR_paths[int(key)][int(start)] = copy.deepcopy(solved_PRs_path[key][start])

            self.min_cost = {}
            for key in PR_paths:
                self.min_cost[int(key)] = None
                for start in PR_paths[key]:
                    if self.min_cost[int(key)] is None:
                        self.min_cost[int(key)] = PR_paths[key][start].cost
                    elif self.min_cost[int(key)] > PR_paths[key][start].cost:
                        self.min_cost[int(key)] = PR_paths[key][start].cost

            #self.build() modified by XX
            #self.build_PR_record() modified by XX
            self.weight = weight
            for PR in self.PR_record:
                for rxn_node in self.PR_record[PR]:
                    non_PR_reactant_node = int(rxn_node.split(",")[0].split("+PR_")[0])
                    self.graph[non_PR_reactant_node][rxn_node][self.weight] = self.graph[non_PR_reactant_node][rxn_node][
                                                                              self.weight] + self.min_cost[PR]
        print("Finding paths...")
        for start in starts:
            ind = 0
            for path in self.valid_shortest_simple_paths(start, target):
                if ind == num_paths:
                    break
                else:
                    ind += 1
                    path_dict_class2 = ReactionPath.characterize_path_final(path, self.weight, self.min_cost,
                                                                            self.graph, PR_paths)
                    heapq.heappush(my_heapq, (path_dict_class2.cost, next(c), path_dict_class2))

        while len(paths) < num_paths and my_heapq:
            # Check if any byproduct could yield a prereq cheaper than from starting molecule(s)?
            (cost_HP, _x, path_dict_HP_class) = heapq.heappop(my_heapq)
            print(len(paths), cost_HP, len(my_heapq), path_dict_HP_class.path_dict)
            paths.append(
                path_dict_HP_class.path_dict)  ### ideally just append the class, but for now dict for easy printing

        print(PR_paths)
        print(paths)

        return PR_paths, paths

    def find_paths_for_all(self, starts, weight, num_paths=10, load_file=True, path=''):  # -> ??
        """
            A method to find the shorted path from given starts to all the nodes in the graph
        :param starts: starts: List(molecular nodes), list of molecular nodes of type int found in the ReactionNetwork.graph
        :param weight: "softplus" or "exponent", type of cost function to use when calculating edge weights
        :param num_paths: Number (of type int) of paths to find. Defaults to 10.
        :param solved_PRs_path: dict that defines a path from each node to a start,
                of the form {int(node1): {int(start1}: {ReactionPath object}, int(start2): {ReactionPath object}}, int(node2):...}
                if None, method will solve PRs
        :param solved_min_cost: dict with minimum cost from path start to a node, of from {node: float},
                if no path exist, value is "no_path", if path is unsolved yet, value is "unsolved_path",
                of None, method will solve for min_cost
        :param updated_graph: nx.DiGraph with udpated edge weights based on the solved PRs, if none, method will solve for PRs and update graph accordingly
        :param save: if True method will save PRs paths, min cost and updated graph after all the PRs are solved,
                    if False, method will not save anything (default)
        :return: PR_paths: solved dict of PRs
        :return: paths: list of paths (number of paths based on the value of num_paths)
        """

        self.weight = weight
        self.num_starts = len(starts)
        paths = []
        c = itertools.count()
        my_heapq = []
        if not load_file:
            self.min_cost = {}
            self.graph = self.build()
            print("Solving prerequisites...")
            PR_paths = self.solve_prerequisites_no_target(starts, weight)

        else:
            print("Loading files...")
            solved_PRs_path = loadfn(path+'PRs.json')
            min_cost = loadfn(path+'min_cost.json')
            self.graph = json_graph.adjacency_graph(loadfn(path+'RN_graph.json'))
            assert len(self.graph.nodes) != 0
            PR_paths = {}

            for key in solved_PRs_path:
                PR_paths[int(key)] = {}
                for start in solved_PRs_path[key]:
                    PR_paths[int(key)][int(start)] = copy.deepcopy(solved_PRs_path[key][start])

            self.min_cost = {}
            for key in min_cost:
                self.min_cost[int(key)] = min_cost[key]

        print("Finding paths...")
        self.all_paths = {}
        for PR in PR_paths:
            print('PR:',PR)
            self.all_paths[PR] = []
            if PR in starts:
                continue
            for start in starts:
                ind = 0
                #try:
                for path in self.valid_shortest_simple_paths(start, PR):
                    if ind == num_paths:
                        break
                    else:
                        ind += 1
                        path_dict_class2 = ReactionPath.characterize_path_final(path, self.weight, self.min_cost,
                                                                                self.graph, PR_paths)
                        heapq.heappush(my_heapq, (path_dict_class2.cost, next(c), path_dict_class2))
                # except:
                #     pass

            while len(paths) < num_paths and my_heapq:
                # Check if any byproduct could yield a prereq cheaper than from starting molecule(s)?
                (cost_HP, _x, path_dict_HP_class) = heapq.heappop(my_heapq)
                print(len(paths), cost_HP, len(my_heapq), path_dict_HP_class.path_dict)
                paths.append(
                    path_dict_HP_class.path_dict)  ### ideally just append the class, but for now dict for easy printing
            self.all_paths[PR].append(paths)
            #print(PR_paths)
            print(paths)
        dumpfn(self.all_paths,'all_path.json')

        return

    def load_files(self,path=''):

        print("Loading files...")
        solved_PRs_path = loadfn(path+'PRs.json')
        min_cost = loadfn(path+'min_cost.json')
        self.graph = json_graph.adjacency_graph(loadfn(path+'RN_graph.json'))
        unsolved_PRs = loadfn(path+'unsolved_PRs.json')
        assert len(self.graph.nodes) != 0
        self.PR_paths = {}
        self.unsolved_PRs = {}

        for key in solved_PRs_path:
            self.PR_paths[int(key)] = {}
            for start in solved_PRs_path[key]:
                self.PR_paths[int(key)][int(start)] = copy.deepcopy(solved_PRs_path[key][start])

        for key in unsolved_PRs:
            self.unsolved_PRs[int(key)] = {}
            for start in unsolved_PRs[key]:
                self.unsolved_PRs[int(key)][int(start)] = copy.deepcopy(unsolved_PRs[key][start])

        self.min_cost = {}
        for key in min_cost:
            self.min_cost[int(key)] = min_cost[key]

        return

    def find_paths_XX(self, starts, target, weight, num_paths=10):  # -> ??
        """
            A method to find the shorted path from given starts to all the nodes in the graph
        :param starts: starts: List(molecular nodes), list of molecular nodes of type int found in the ReactionNetwork.graph
        :param weight: "softplus" or "exponent", type of cost function to use when calculating edge weights
        :param num_paths: Number (of type int) of paths to find. Defaults to 10.
        :param solved_PRs_path: dict that defines a path from each node to a start,
                of the form {int(node1): {int(start1}: {ReactionPath object}, int(start2): {ReactionPath object}}, int(node2):...}
                if None, method will solve PRs
        :param solved_min_cost: dict with minimum cost from path start to a node, of from {node: float},
                if no path exist, value is "no_path", if path is unsolved yet, value is "unsolved_path",
                of None, method will solve for min_cost
        :param updated_graph: nx.DiGraph with udpated edge weights based on the solved PRs, if none, method will solve for PRs and update graph accordingly
        :param save: if True method will save PRs paths, min cost and updated graph after all the PRs are solved,
                    if False, method will not save anything (default)
        :return: PR_paths: solved dict of PRs
        :return: paths: list of paths (number of paths based on the value of num_paths)
        """

        self.weight = weight
        self.num_starts = len(starts)
        paths = []
        c = itertools.count()
        my_heapq = []
        print("Finding paths...")

        for start in starts:
            ind = 0
            # try:
            print('start:',start)
            for path in self.valid_shortest_simple_paths(start, target):
                if ind == num_paths:
                    break
                else:
                    ind += 1
                    path_dict_class2 = ReactionPath.characterize_path_final(path, self.weight, self.min_cost,
                                                                            self.graph, self.PR_paths)
                    heapq.heappush(my_heapq, (path_dict_class2.cost, next(c), path_dict_class2))
            # except:
            #     pass

        while len(paths) < num_paths and my_heapq:
            # Check if any byproduct could yield a prereq cheaper than from starting molecule(s)?
            (cost_HP, _x, path_dict_HP_class) = heapq.heappop(my_heapq)
            print(len(paths), cost_HP, len(my_heapq), path_dict_HP_class.path_dict)
            paths.append(
                path_dict_HP_class.path_dict)  ### ideally just append the class, but for now dict for easy printing
        #print(PR_paths)
        print(paths)

        return paths

    def find_paths_XX_all(self, starts, weight, num_paths=1,path=''):

        self.all_paths = {}
        self.load_files(path)
        for PR in self.PR_paths:
            print('PR:',PR)
            self.all_paths[PR] = []
            if PR in starts:
                continue
            else:
                paths = self.find_paths_XX(starts, PR, weight, num_paths)
                self.all_paths[PR].append(paths)

        dumpfn(self.all_paths, 'all_paths.json')

        return

    def get_species_from_path(self, path='', thresh=0.0):
        '''
        Get all the entries from path finding to all species in the network.
        :param path: path to the 'all_paths.json' file
        :param thresh: A fugde factor for free energy
        :return:
        '''
        filtered_entries_list = []
        filtered_PRs = []
        self.all_paths = loadfn(path+'all_paths.json')

        for PR in self.all_paths:
            if self.all_paths[PR] == []:
                filtered_PRs.append(int(PR))
                filtered_entries_list.append(self.entries_list[int(PR)])
            elif self.all_paths[PR] != [[]]:
                overall_free_energy_change = self.all_paths[PR][0][0]['overall_free_energy_change']
                if overall_free_energy_change > thresh:
                    filtered_PRs.append(int(PR))
                    filtered_entries_list.append(self.entries_list[int(PR)])

        if not os.path.isdir('filtered_mols'):
            os.mkdir('filtered_mols')

        for i, entry in enumerate(filtered_entries_list):
            mol = entry.molecule
            mol.to('xyz','filtered_mols/'+str(i)+'.xyz')
        dumpfn(filtered_entries_list, 'filtered_entries_list.json')
        dumpfn(filtered_PRs, 'filtered_PRs.json')
        print('Number of species remaining:',len(filtered_PRs))

        return

    def get_species_direct_from_PR(self, starts, weight, path='', thresh=0.0):
        '''
        Get all the entries from path finding to all species in the network.
        :param path: path to the 'all_paths.json' file
        :param thresh: A fugde factor for free energy
        :return:
        '''
        filtered_entries_list = []
        filtered_PRs = []
        self.load_files(path)

        for PR in self.PR_paths:
            print('PR:',PR)
            min_free_energy_change = 1e8
            for start in self.PR_paths[PR]:
                rxn_path = self.PR_paths[PR][start]
                new_rxn_path = rxn_path.characterize_path_final(rxn_path.path,weight, self.min_cost, self.graph, self.PR_paths)
                print(new_rxn_path.full_path)
                #if len(list(set(new_rxn_path.solved_prereqs))) != len(list(set(new_rxn_path.all_prereqs))):
                    #print(PR, start)
                overall_free_energy_change = new_rxn_path.overall_free_energy_change
                if overall_free_energy_change <= min_free_energy_change:
                    min_free_energy_change = overall_free_energy_change
            if min_free_energy_change < thresh:
                filtered_PRs.append(int(PR))
                filtered_entries_list.append(self.entries_list[int(PR)])

        for start in starts:
            filtered_entries_list.append(self.entries_list[start])
            filtered_PRs.append(start)

        if not os.path.isdir('filtered_mols'):
            os.mkdir('filtered_mols')

        for i, entry in enumerate(filtered_entries_list):
            mol = entry.molecule
            mol.to('xyz','filtered_mols/'+str(filtered_PRs[i])+'.xyz')
        dumpfn(filtered_entries_list, 'filtered_entries_list.json')
        dumpfn(filtered_PRs, 'filtered_PRs.json')
        print('Number of species remaining:',len(filtered_PRs))

        return

    def get_species_direct_from_PR_w_intermediates(self, starts, weight, path='', thresh=0.0,name=''):
        '''
        Get all the entries from path finding to all species in the network.
        Select the species accessible with deltaG < -thresh pathways and include all intermediates along the way.
        :param path: path to the 'all_paths.json' file
        :param thresh: A fugde factor for free energy
        :return:
        '''
        filtered_entries_list = []
        filtered_species = []
        self.load_files(path)

        for PR in self.PR_paths:
            print('PR:',PR)
            min_free_energy_change = 1e8
            for start in self.PR_paths[PR]:
                rxn_path = self.PR_paths[PR][start]
                new_rxn_path = rxn_path.characterize_path_final(rxn_path.path,weight, self.min_cost, self.graph, self.PR_paths)
                print(new_rxn_path.full_path)
                #if len(list(set(new_rxn_path.solved_prereqs))) != len(list(set(new_rxn_path.all_prereqs))):
                    #print(PR, start)
                overall_free_energy_change = new_rxn_path.overall_free_energy_change
                if overall_free_energy_change <= min_free_energy_change:
                    min_free_energy_change = overall_free_energy_change
                    path_to_parse = new_rxn_path.full_path
                    if path_to_parse != None:
                        relevant_species = []
                        for item in path_to_parse:
                            if isinstance(item, str):
                                reactants, products = item.split(',')[0].split('+'), item.split(',')[1].split('+')
                                reactants = [i.replace('PR_','') for i in reactants]
                                relevant_species += list(set(reactants+products))
                        relevant_species = list(set(relevant_species))
                        print(relevant_species)
                        assert str(PR) in relevant_species

            if min_free_energy_change < thresh and not(any(int(specie) in self.unsolved_PRs for specie in relevant_species)):
                for item in relevant_species:
                    filtered_species.append(int(item))
                    filtered_entries_list.append(self.entries_list[int(item)])

        for start in starts:
            if start not in filtered_species:
                filtered_entries_list.append(self.entries_list[start])
                filtered_species.append(start)

        if not os.path.isdir(name+'filtered_mols'):
            os.mkdir(name+'filtered_mols')

        for i, entry in enumerate(filtered_entries_list):
            mol = entry.molecule
            mol.to('xyz',name+'filtered_mols/'+str(filtered_species[i])+'.xyz')
        dumpfn(filtered_entries_list, name+'filtered_entries_list.json')
        dumpfn(filtered_species, name+'filtered_species.json')
        print('Number of species remaining:',len(filtered_species))

        return

    def get_LEDC_LEMC_cost(self,LEDC_ind, LEMC_ind, weight):
        print("Getting LEDC/LEMC full path!")
        print('LEDC ind:', LEDC_ind)
        for start in self.PR_paths[LEDC_ind]:
            print('start:',start)
            rxn_path = self.PR_paths[LEDC_ind][start]
            new_rxn_path = rxn_path.characterize_path_final(rxn_path.path, weight, self.min_cost, self.graph,
                                                            self.PR_paths)
            print('LEDC free energy change from'+ str(start) +':',new_rxn_path.overall_free_energy_change)
            print('LEDC full path from' + str(start) +':',new_rxn_path.full_path)
        print('LEMC ind:', LEDC_ind)
        for start in self.PR_paths[LEMC_ind]:
            print('start:',start)
            rxn_path = self.PR_paths[LEMC_ind][start]
            new_rxn_path = rxn_path.characterize_path_final(rxn_path.path, weight, self.min_cost, self.graph,
                                                            self.PR_paths)
            print('LEMC full path from' + str(start) +':',new_rxn_path.full_path)
            print('LEMC free energy change from' + str(start) + ':', new_rxn_path.overall_free_energy_change)

        return

    def get_cost_for_inds(self,inds, weight,path=''):
        self.load_files(path)
        overall_free_energy_charges = {}
        full_paths = {}
        for i, ind in enumerate(inds):
            overall_free_energy_charges[ind] = {}
            full_paths[ind] = {}
            if ind == None:
                overall_free_energy_charges[ind] = None
                full_paths[ind] = None
            else:
                for start in self.PR_paths[ind]:
                    print('start:', start)
                    overall_free_energy_charges[ind][start] = {}
                    rxn_path = self.PR_paths[ind][start]
                    new_rxn_path = rxn_path.characterize_path_final(rxn_path.path, weight, self.min_cost, self.graph,
                                                                    self.PR_paths)
                    overall_free_energy_charges[ind][start] = new_rxn_path.overall_free_energy_change
                    full_paths[ind][start] = new_rxn_path.full_path
        dumpfn(overall_free_energy_charges, 'overall_free_energy_changes_for_inds.json')
        dumpfn(full_paths, 'full_paths_for_inds.json')

        return

    def remove_node(self,node_ind):
        '''
        Remove a species from self.graph. Also remove all the reaction nodes with that species. Used for removing Li0.
        :return:
        '''
        self.graph.remove_node(node_ind)
        nodes = list(self.graph.nodes)
        for node in nodes:
            if self.graph.nodes[node]["bipartite"] == 1:
                reactants = node.split(',')[0].split('+')
                reactants = [reac.replace('PR_','') for reac in reactants]
                products =  node.split(',')[1].split('+')
                if str(node_ind) in reactants or str(node_ind) in products:
                    self.graph.remove_node(node)

        return

if __name__ == "__main__":
    prod_entries = []
    entries = loadfn(
        "/Users/xiaoweixie/PycharmProjects/electrolyte/LEMC/smd_production_entries_3LiEC_10_lowest_free_energy_20200424_5_species.json")
    for entry in entries:
        if "optimized_molecule" in entry["output"]:
            molecule = entry["output"]["optimized_molecule"]
        else:
            molecule = entry["output"]["initial_molecule"]
        H = float(entry["output"]["enthalpy"])
        S = float(entry["output"]["entropy"])
        E = float(entry["output"]["final_energy"])
        mol_entry = MoleculeEntry(molecule=molecule, energy=E, enthalpy=H, entropy=S, entry_id=entry["task_id"])
        prod_entries.append(mol_entry)
    RN = ReactionNetwork(input_entries=prod_entries)
    # dumpfn(RN,"/Users/xiaoweixie/pymatgen/pymatgen/analysis/reaction_network/mgcf/LEMC_RN_electronic_4_species/LEMC_small_RN.json")

    EC_mg = MoleculeGraph.with_local_env_strategy(
        Molecule.from_file("/Users/xiaoweixie/Desktop/Sam_production/xyzs/EC.xyz"),
        OpenBabelNN(),
        reorder=False,
        extend_structure=False)
    EC_mg = metal_edge_extender(EC_mg)

    LiEC_mg = MoleculeGraph.with_local_env_strategy(
        Molecule.from_file("/Users/xiaoweixie/Desktop/Sam_production/xyzs/LiEC_bi.xyz"),
        OpenBabelNN(),
        reorder=False,
        extend_structure=False)
    LiEC_mg = metal_edge_extender(LiEC_mg)

    LEMC_mg = MoleculeGraph.with_local_env_strategy(
        Molecule.from_file("/Users/xiaoweixie/Desktop/Sam_production/xyzs/LEMC.xyz"),
        OpenBabelNN(),
        reorder=False,
        extend_structure=False)
    LEMC_mg = metal_edge_extender(LEMC_mg)

    LEDC_mg = MoleculeGraph.with_local_env_strategy(
        Molecule.from_file("/Users/xiaoweixie/Desktop/Sam_production/xyzs/LEDC.xyz"),
        OpenBabelNN(),
        reorder=False,
        extend_structure=False)
    LEDC_mg = metal_edge_extender(LEDC_mg)

    H2O_mg = MoleculeGraph.with_local_env_strategy(
        Molecule.from_file("/Users/xiaoweixie/Desktop/Sam_production/xyzs/water.xyz"),
        OpenBabelNN(),
        reorder=False,
        extend_structure=False)

    LiCO3_minus_mg = MoleculeGraph.with_local_env_strategy(
        Molecule.from_file("/Users/xiaoweixie/Desktop/Sam_production/xyzs/LiCO3.xyz"),
        OpenBabelNN(),
        reorder=False,
        extend_structure=False)
    LiCO3_minus_mg = metal_edge_extender(LiCO3_minus_mg)

    EC_ind = None
    LEDC_ind = None
    LEMC_ind = None
    LiEC_ind = None
    LiCO3_minus_ind = None
    H2O_ind = None

    for entry in RN.entries["C3 H4 O3"][10][0]:
        if EC_mg.isomorphic_to(entry.mol_graph):
            if entry.free_energy == -9317.492754189294:
                EC_ind = entry.parameters["ind"]
                break
    for entry in RN.entries["C4 H4 Li2 O6"][17][0]:
        if LEDC_mg.isomorphic_to(entry.mol_graph):
            if entry.free_energy == -16910.7035955349:
                LEDC_ind = entry.parameters["ind"]
                break
    # for entry in RN.entries["C3 H5 Li1 O4"][13][0]:
    #     if LEMC_mg.isomorphic_to(entry.mol_graph):
    #         if entry.free_energy == -11587.839161760392:
    #             LEMC_ind = entry.parameters["ind"]
    #             break
    for entry in RN.entries["C3 H4 Li1 O3"][12][0]:
        if LiEC_mg.isomorphic_to(entry.mol_graph):
            print('LiEC found')
            if entry.free_energy == -9521.708410009893:
                LiEC_ind = entry.parameters["ind"]
                break
    for entry in RN.entries["H2 O1"][2][0]:
        if H2O_mg.isomorphic_to(entry.mol_graph):
            H2O_ind = entry.parameters["ind"]
            break
    # for entry in RN.entries["C1 Li1 O3"][5][-1]:
    #     if LiCO3_minus_mg.isomorphic_to(entry.mol_graph):
    #         # print('LiCO3 minus found!', entry.free_energy)
    #         # entry.mol_graph.molecule.to('xyz','LiCO3_minus_test.xyz')
    #         if entry.free_energy == -7389.618636590198:
    #             LiCO3_minus_ind = entry.parameters["ind"]
    #             break

    Li1_ind = RN.entries["Li1"][0][1][0].parameters["ind"]

    print("EC_ind", EC_ind)
    print("LEDC_ind", LEDC_ind)
    print("LEMC_ind", LEMC_ind)
    print("Li1_ind", Li1_ind)
    print("LiEC_ind", LiEC_ind)
    print("LiCO3_minus_ind", LiCO3_minus_ind)

    starts = [EC_ind, Li1_ind, H2O_ind]
    target = LEDC_ind
    weight = "softplus"
    max_iter = 100
    RN.num_starts = len(starts)
    RN.build()
    RN.build_concerted_reactions(name="nothing", read_file=False, num_processors=2, reaction_type="break2_form2", allowed_charge_change=0)
    #RN.solve_prerequisites(starts,target,weight,max_iter,save=True)