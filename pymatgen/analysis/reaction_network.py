# coding: utf-8
# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.


import logging
import copy
import itertools
import numpy as np
from monty.json import MSONable
from pymatgen.analysis.graphs import MoleculeGraph, MolGraphSplitError, isomorphic
from pymatgen.analysis.local_env import OpenBabelNN
from pymatgen.io.babel import BabelMolAdaptor
from pymatgen import Molecule
from pymatgen.analysis.fragmenter import metal_edge_extender
import networkx as nx
from networkx.algorithms import bipartite


__author__ = "Samuel Blau"
__copyright__ = "Copyright 2019, The Materials Project"
__version__ = "1.0"
__maintainer__ = "Samuel Blau"
__email__ = "samblau1@gmail.com"
__status__ = "Alpha"
__date__ = "7/30/19"


logger = logging.getLogger(__name__)



class ReactionNetwork(MSONable):
    """
    Class to create a reaction network from entries

    Args:
        entries ([MoleculeEntry]): A list of ReactionNetworkEntry objects.
        

    """

    def __init__(self, entries):

        self.entries = {}
        self.entries_list = []
        
        print(len(entries),"total entries")

        get_formula = lambda x: x.formula
        get_Nbonds = lambda x: x.Nbonds
        get_charge = lambda x: x.charge

        sorted_entries_0 = sorted(entries, key=get_formula)
        for k1, g1 in itertools.groupby(sorted_entries_0, get_formula):
            sorted_entries_1 = sorted(list(g1),key=get_Nbonds)
            self.entries[k1] = {}
            for k2, g2 in itertools.groupby(sorted_entries_1, get_Nbonds):
                sorted_entries_2 = sorted(list(g2),key=get_charge)
                self.entries[k1][k2] = {}
                for k3, g3 in itertools.groupby(sorted_entries_2, get_charge):
                    sorted_entries_3 = list(g3)
                    if len(sorted_entries_3) > 1:
                        unique = []
                        for entry in sorted_entries_3:
                            isomorphic_found = False
                            for ii,Uentry in enumerate(unique):
                                if isomorphic(entry.graph,Uentry.graph):
                                    isomorphic_found = True
                                    # print("Isomorphic entries with equal charges found!")
                                    if entry.free_energy != None and Uentry.free_energy != None:
                                        if entry.free_energy < Uentry.free_energy:
                                            unique[ii] = entry
                                            if entry.energy > Uentry.energy:
                                                print("WARNING: Free energy lower but electronic energy higher!")
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

        print(len(self.entries_list),"unique entries")

        for ii, entry in enumerate(self.entries_list):
            entry.parameters["ind"] = ii
        

        self.graph = nx.Graph()
        self.graph.add_nodes_from(range(len(self.entries_list)),bipartite=0)

        print("Before reactions")
        print(len(self.graph.nodes),"nodes")
        print(len(self.graph.edges),"edges")
        print()

        self.one_electron_redox()

        print("After 1e redox")
        print(len(self.graph.nodes),"nodes")
        print(len(self.graph.edges),"edges")
        print()

        self.intramol_single_bond_change()

        print("After intramol bond change")
        print(len(self.graph.nodes),"nodes")
        print(len(self.graph.edges),"edges")
        print()

        self.single_bond_breakage()

        print("After single bond breakage")
        print(len(self.graph.nodes),"nodes")
        print(len(self.graph.edges),"edges")
        print()

        

        
    def one_electron_redox(self):
        # A^n ±e- <-> A^n±1
        # Two entries with:
        #     identical composition
        #     identical number of edges
        #     a charge difference of 1
        #     isomorphic molecule graphs

        for formula in self.entries:
            for Nbonds in self.entries[formula]:
                charges = list(self.entries[formula][Nbonds].keys())
                if len(charges) > 1:
                    for ii in range(len(charges)-1):
                        charge0 = charges[ii]
                        charge1 = charges[ii+1]
                        if charge1-charge0 == 1:
                            for entry0 in self.entries[formula][Nbonds][charge0]:
                                for entry1 in self.entries[formula][Nbonds][charge1]:
                                    if isomorphic(entry0.graph,entry1.graph):
                                        if entry0.parameters["ind"] <= entry1.parameters["ind"]:
                                            node_name = str(entry0.parameters["ind"])+","+str(entry1.parameters["ind"])
                                        else:
                                            node_name = str(entry1.parameters["ind"])+","+str(entry0.parameters["ind"])
                                        self.graph.add_node(node_name,rxn_type="one_electron_redox",bipartite=1)
                                        self.graph.add_edge(entry0.parameters["ind"],node_name)
                                        self.graph.add_edge(entry1.parameters["ind"],node_name)
                                        break

    def intramol_single_bond_change(self):
        # A^n <-> B^n
        # Two entries with:
        #     identical composition
        #     number of edges differ by 1
        #     identical charge
        #     removing one of the edges in the graph with more edges yields a graph isomorphic to the other entry

        for formula in self.entries:
            Nbonds_list = list(self.entries[formula].keys())
            if len(Nbonds_list) > 1:
                for ii in range(len(Nbonds_list)-1):
                    Nbonds0 = Nbonds_list[ii]
                    Nbonds1 = Nbonds_list[ii+1]
                    if Nbonds1-Nbonds0 == 1:
                        for charge in self.entries[formula][Nbonds0]:
                            if charge in self.entries[formula][Nbonds1]:
                                for entry1 in self.entries[formula][Nbonds1][charge]:
                                    for edge in entry1.edges:
                                        mg = copy.deepcopy(entry1.mol_graph)
                                        mg.break_edge(edge[0],edge[1],allow_reverse=True)
                                        if nx.is_weakly_connected(mg.graph):
                                            for entry0 in self.entries[formula][Nbonds0][charge]:
                                                if isomorphic(entry0.graph,mg.graph):
                                                    if entry0.parameters["ind"] <= entry1.parameters["ind"]:
                                                        node_name = str(entry0.parameters["ind"])+","+str(entry1.parameters["ind"])
                                                    else:
                                                        node_name = str(entry1.parameters["ind"])+","+str(entry0.parameters["ind"])
                                                    self.graph.add_node(node_name,rxn_type="intramol_single_bond_change",bipartite=1)
                                                    self.graph.add_edge(entry0.parameters["ind"],node_name)
                                                    self.graph.add_edge(entry1.parameters["ind"],node_name)
                                                    break


    def single_bond_breakage(self):
        # A <-> B + C
        # Three entries with:
        #     comp(A) = comp(B) + comp(C)
        #     charge(A) = charge(B) + charge(C)
        #     removing one of the edges in A yields two disconnected subgraphs that are isomorphic to B and C
        #        what about for Li complexes that have two O-Li bonds?

        for formula in self.entries:
            for Nbonds in self.entries[formula]:
                if Nbonds > 0:
                    for charge in self.entries[formula][Nbonds]:
                        for entry in self.entries[formula][Nbonds][charge]:
                            for edge in entry.edges:
                                bond = [(edge[0],edge[1])]
                                try:
                                    frags = entry.mol_graph.split_molecule_subgraphs(bond, allow_reverse=True)
                                    graph0 = frags[0].graph
                                    formula0 = frags[0].molecule.composition.alphabetical_formula
                                    Nbonds0 = len(frags[0].graph.edges())
                                    graph1 = frags[1].graph
                                    formula1 = frags[1].molecule.composition.alphabetical_formula
                                    Nbonds1 = len(frags[1].graph.edges())
                                    if formula0 in self.entries and formula1 in self.entries:
                                        if Nbonds0 in self.entries[formula0] and Nbonds1 in self.entries[formula1]:
                                            for charge0 in self.entries[formula0][Nbonds0]:
                                                for entry0 in self.entries[formula0][Nbonds0][charge0]:
                                                    if isomorphic(graph0,entry0.graph):
                                                        charge1 = charge - charge0
                                                        if charge1 in self.entries[formula1][Nbonds1]:
                                                            for entry1 in self.entries[formula1][Nbonds1][charge1]:
                                                                if isomorphic(graph1,entry1.graph):
                                                                    if entry0.parameters["ind"] <= entry1.parameters["ind"]:
                                                                        node_name = str(entry.parameters["ind"])+","+str(entry0.parameters["ind"])+"+"+str(entry1.parameters["ind"])
                                                                    else:
                                                                        node_name = str(entry.parameters["ind"])+","+str(entry1.parameters["ind"])+"+"+str(entry0.parameters["ind"])
                                                                    self.graph.add_node(node_name,rxn_type="single_bond_breakage",bipartite=1)
                                                                    self.graph.add_edge(entry.parameters["ind"],node_name)
                                                                    self.graph.add_edge(entry0.parameters["ind"],node_name)
                                                                    self.graph.add_edge(entry1.parameters["ind"],node_name)
                                                                    break
                                                        break
                                except MolGraphSplitError:
                                    pass






