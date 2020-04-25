# coding: utf-8
# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.


import logging
import copy
import itertools
import heapq
import numpy as np
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
from pymatgen.analysis.reaction_network.extract_reactions import *
from monty.serialization import loadfn, dumpfn
from multiprocessing import cpu_count
from pathos.multiprocessing import ProcessingPool as Pool

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
        input_entries ([MoleculeEntry]): A list of MoleculeEntry objects.
        electron_free_energy (float): The Gibbs free energy of an electron.
            Defaults to -2.15 eV, the value at which the LiEC SEI forms.
    """

    def __init__(self, input_entries, electron_free_energy=-2.15):

        self.input_entries = input_entries
        self.electron_free_energy = electron_free_energy
        self.entries = {}
        self.entries_list = []

        print(len(self.input_entries),"input entries")

        connected_entries = []
        for entry in self.input_entries:
            # print(len(entry.molecule))
            if len(entry.molecule) > 1:
                if nx.is_weakly_connected(entry.graph):
                    connected_entries.append(entry)
            else:
                connected_entries.append(entry)
        print(len(connected_entries),"connected entries")

        get_formula = lambda x: x.formula
        get_Nbonds = lambda x: x.Nbonds
        get_charge = lambda x: x.charge

        sorted_entries_0 = sorted(connected_entries, key=get_formula)
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

        print(len(self.entries_list),"unique entries")

        for ii, entry in enumerate(self.entries_list):
            entry.parameters["ind"] = ii

        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(range(len(self.entries_list)),bipartite=0)

        self.num_reactions = 0
        self.one_electron_redox()
        self.intramol_single_bond_change()
        self.intermol_single_bond_change()
        self.coordination_bond_change()
        #self.add_water_reactions()
        #self.concerted_2_steps()
        #self.add_LEDC_concerted_reactions()
        #self.add_water_lithium_reaction()
        self.PR_record = self.build_PR_record()
        self.min_cost = {}
        self.num_starts = None

    def one_electron_redox(self):
        # One electron oxidation / reduction without change to bonding
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
                                    if entry0.mol_graph.isomorphic_to(entry1.mol_graph):
                                        self.add_reaction([entry0],[entry1],"one_electron_redox")
                                        break

    def intramol_single_bond_change(self):
        # Intramolecular formation / breakage of one bond
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
                                                if entry0.mol_graph.isomorphic_to(mg):
                                                    self.add_reaction([entry0],[entry1],"intramol_single_bond_change")
                                                    break


    def intermol_single_bond_change(self):
        # Intermolecular formation / breakage of one bond
        # A <-> B + C aka B + C <-> A
        # Three entries with:
        #     comp(A) = comp(B) + comp(C)
        #     charge(A) = charge(B) + charge(C)
        #     removing one of the edges in A yields two disconnected subgraphs that are isomorphic to B and C

        for formula in self.entries:
            for Nbonds in self.entries[formula]:
                if Nbonds > 0:
                    for charge in self.entries[formula][Nbonds]:
                        for entry in self.entries[formula][Nbonds][charge]:
                            for edge in entry.edges:
                                bond = [(edge[0],edge[1])]
                                try:
                                    frags = entry.mol_graph.split_molecule_subgraphs(bond, allow_reverse=True)
                                    formula0 = frags[0].molecule.composition.alphabetical_formula
                                    Nbonds0 = len(frags[0].graph.edges())
                                    formula1 = frags[1].molecule.composition.alphabetical_formula
                                    Nbonds1 = len(frags[1].graph.edges())
                                    if formula0 in self.entries and formula1 in self.entries:
                                        if Nbonds0 in self.entries[formula0] and Nbonds1 in self.entries[formula1]:
                                            for charge0 in self.entries[formula0][Nbonds0]:
                                                for entry0 in self.entries[formula0][Nbonds0][charge0]:
                                                    if frags[0].isomorphic_to(entry0.mol_graph):
                                                        charge1 = charge - charge0
                                                        if charge1 in self.entries[formula1][Nbonds1]:
                                                            for entry1 in self.entries[formula1][Nbonds1][charge1]:
                                                                if frags[1].isomorphic_to(entry1.mol_graph):
                                                                    self.add_reaction([entry],[entry0,entry1],"intermol_single_bond_change")
                                                                    break
                                                        break
                                except MolGraphSplitError:
                                    pass

    def coordination_bond_change(self):
        # Simultaneous formation / breakage of multiple coordination bonds
        # A + M <-> AM aka AM <-> A + M
        # Three entries with:
        #     M = Li or Mg
        #     comp(AM) = comp(A) + comp(M)
        #     charge(AM) = charge(A) + charge(M)
        #     removing two M-containing edges in AM yields two disconnected subgraphs that are isomorphic to B and C
        M_entries = {}
        for formula in self.entries:
            if formula == "Li1" or formula == "Mg1":
                if formula not in M_entries:
                    M_entries[formula] = {}
                for charge in self.entries[formula][0]:
                    assert(len(self.entries[formula][0][charge])==1)
                    M_entries[formula][charge] = self.entries[formula][0][charge][0]
        if M_entries != {}:
            for formula in self.entries:
                if "Li" in formula or "Mg" in formula:
                    for Nbonds in self.entries[formula]:
                        if Nbonds > 2:
                            for charge in self.entries[formula][Nbonds]:
                                for entry in self.entries[formula][Nbonds][charge]:
                                    nosplit_M_bonds = []
                                    for edge in entry.edges:
                                        if str(entry.molecule.sites[edge[0]].species) in M_entries or str(entry.molecule.sites[edge[1]].species) in M_entries:
                                            M_bond = (edge[0],edge[1])
                                            try:
                                                frags = entry.mol_graph.split_molecule_subgraphs([M_bond], allow_reverse=True)
                                            except MolGraphSplitError:
                                                nosplit_M_bonds.append(M_bond)
                                    bond_pairs = itertools.combinations(nosplit_M_bonds, 2)
                                    for bond_pair in bond_pairs:
                                        try:
                                            frags = entry.mol_graph.split_molecule_subgraphs(bond_pair, allow_reverse=True)
                                            M_ind = None
                                            M_formula = None
                                            for ii,frag in enumerate(frags):
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
                                                        if nonM_formula in self.entries:
                                                            if nonM_Nbonds in self.entries[nonM_formula]:
                                                                for nonM_charge in self.entries[nonM_formula][nonM_Nbonds]:
                                                                    M_charge = entry.charge - nonM_charge
                                                                    if M_charge in M_entries[M_formula]:
                                                                        for nonM_entry in self.entries[nonM_formula][nonM_Nbonds][nonM_charge]:
                                                                            if frag.isomorphic_to(nonM_entry.mol_graph):
                                                                                self.add_reaction([entry],[nonM_entry,M_entries[M_formula][M_charge]],"coordination_bond_change")
                                                                                break
                                        except MolGraphSplitError:
                                            pass

    def concerted_2_steps(self):
        reactions_to_add = []
        for node0 in self.graph.nodes():
            if self.graph.node[node0]["bipartite"] == 0:
                node0_rxns = list(self.graph.neighbors(node0))
                for rxn0 in node0_rxns:
                    if self.graph.node[rxn0]["free_energy"] > 0:
                        rxn0_products = list(self.graph.neighbors(rxn0))
                        if len(rxn0_products) == 2: # This must be an A -> B+C bond breaking reaction
                            for node1 in rxn0_products:
                                node1_rxns = list(self.graph.neighbors(node1))
                                for rxn1 in node1_rxns:
                                    if self.graph.node[rxn0]["free_energy"] + self.graph.node[rxn1]["free_energy"] < -1e-8 and "PR" in rxn1: # This must be an A+B -> C bond forming reaction"
                                        reactant_nodes = [node0]
                                        product_nodes = list(self.graph.neighbors(rxn1))
                                        if "PR" in rxn0:
                                            reactant_nodes.append(int(rxn0.split(",")[0].split("+PR_")[1]))
                                        if "PR" in rxn1:
                                            reactant_nodes.append(int(rxn1.split(",")[0].split("+PR_")[1]))
                                        if len(reactant_nodes) > 2:
                                            print("WARNING: More than two reactants! Ignoring...")
                                        for prod in rxn0_products:
                                            if prod != node1:
                                                product_nodes.append(prod)
                                        if len(product_nodes) > 2:
                                            print("WARNING: More than two products! Ignoring...")
                                        if len(reactant_nodes) <= 2 and len(product_nodes) <= 2:
                                            entries0 = []
                                            for ind in reactant_nodes:
                                                entries0.append(self.entries_list[ind])
                                            entries1 = []
                                            for ind in product_nodes:
                                                entries1.append(self.entries_list[ind])
                                            reactions_to_add.append([entries0,entries1])
        for to_add in reactions_to_add:
            print(len(to_add[0]),len(to_add[1]))
            self.add_reaction(to_add[0],to_add[1],"concerted")


    def find_concerted_candidates(self,name):
        self.unique_mol_graphs = []
        for entry in self.entries_list:
            mol_graph = entry.mol_graph
            self.unique_mol_graphs.append(mol_graph)

        self.unique_mol_graphs_new = []
        # For duplicate mol graphs, create a map between later species with former ones
        self.unique_mol_graph_dict = {}

        for i in range(len(self.unique_mol_graphs)):
            mol_graph = self.unique_mol_graphs[i]
            found = False
            for j in range(len(self.unique_mol_graphs_new)):
                new_mol_graph = self.unique_mol_graphs_new[j]
                if mol_graph.isomorphic_to(new_mol_graph):
                    found = True
                    self.unique_mol_graph_dict[i] = j
                    continue
            if not found:
                self.unique_mol_graph_dict[i] = len(self.unique_mol_graphs_new)
                self.unique_mol_graphs_new.append(mol_graph)
        dumpfn(self.unique_mol_graph_dict, name + "_unique_mol_graph_map.json")
        # find all molecule pairs that satisfy the stoichiometry constraint
        self.stoi_list, self.species_same_stoi_dict = identify_same_stoi_mol_pairs(self.unique_mol_graphs_new)
        self.reac_prod_dict = {}
        for i, key in enumerate(self.species_same_stoi_dict.keys()):
            species_list = self.species_same_stoi_dict[key]
            new_species_list_reactant = []
            new_species_list_product = []
            for species in species_list:
                new_species_list_reactant.append(species)
                new_species_list_product.append(species)
            if new_species_list_reactant != [] and new_species_list_product != []:
                self.reac_prod_dict[key] = {'reactants': new_species_list_reactant, 'products': new_species_list_product}
        return

    def find_concerted_candidates_equal(self,name):
        self.unique_mol_graphs = []
        for entry in self.entries_list:
            mol_graph = entry.mol_graph
            self.unique_mol_graphs.append(mol_graph)

        self.unique_mol_graphs_new = []
        # For duplicate mol graphs, create a map between later species with former ones
        self.unique_mol_graph_dict = {}

        for i in range(len(self.unique_mol_graphs)):
            mol_graph = self.unique_mol_graphs[i]
            found = False
            for j in range(len(self.unique_mol_graphs_new)):
                new_mol_graph = self.unique_mol_graphs_new[j]
                if mol_graph.isomorphic_to(new_mol_graph):
                    found = True
                    self.unique_mol_graph_dict[i] = j
                    continue
            if not found:
                self.unique_mol_graph_dict[i] = len(self.unique_mol_graphs_new)
                self.unique_mol_graphs_new.append(mol_graph)
        dumpfn(self.unique_mol_graph_dict, name + "_unique_mol_graph_map.json")
        # find all molecule pairs that satisfy the stoichiometry constraint
        self.stoi_list, self.species_same_stoi_dict = identify_same_stoi_mol_pairs(self.unique_mol_graphs_new)
        self.reac_prod_dict = {}
        for i, key in enumerate(self.species_same_stoi_dict.keys()):
            species_list = self.species_same_stoi_dict[key]
            new_species_list_reactant = []
            new_species_list_product = []
            for species in species_list:
                new_species_list_reactant.append(species)
                new_species_list_product.append(species)
            if new_species_list_reactant != [] and new_species_list_product != []:
                self.reac_prod_dict[key] = {'reactants': new_species_list_reactant, 'products': new_species_list_product}
        self.concerted_rxns_to_determine = []
        for key in self.reac_prod_dict.keys():
            reactants = self.reac_prod_dict[key]['reactants']
            products = self.reac_prod_dict[key]['products']
            for j in range(len(reactants)):
                reac = reactants[j]
                for k in range(len(products)):
                    prod = products[k]
                    if k <= j:
                        continue
                    else:
                        self.concerted_rxns_to_determine.append([reac,prod])
        return

    def find_concerted_general_multiprocess(self, args):
        key, name = args[0], args[1]
        valid_reactions_dict = {}
        print(key)
        valid_reactions_dict[key] = []
        reactants = self.reac_prod_dict[key]['reactants']
        products = self.reac_prod_dict[key]['products']
        for j in range(len(reactants)):
            reac = reactants[j]
            for k in range(len(products)):
                prod = products[k]
                if k <= j:
                    continue
                else:
                    print('reactant:', reac)
                    print('product:', prod)
                    split_reac = reac.split('_')
                    split_prod = prod.split('_')
                    if (len(split_reac) == 1 and len(split_prod) == 1):
                        mol_graph1 = self.unique_mol_graphs_new[int(split_reac[0])]
                        mol_graph2 = self.unique_mol_graphs_new[int(split_prod[0])]
                        if identify_self_reactions(mol_graph1, mol_graph2):
                            if [reac, prod] not in valid_reactions_dict[key]:
                                valid_reactions_dict[key].append([reac, prod])
                    elif (len(split_reac) == 2 and len(split_prod) == 1):
                        assert split_prod[0] not in split_reac
                        mol_graphs1 = [self.unique_mol_graphs_new[int(split_reac[0])],
                                       self.unique_mol_graphs_new[int(split_reac[1])]]
                        mol_graphs2 = [self.unique_mol_graphs_new[int(split_prod[0])]]
                        if identify_reactions_AB_C(mol_graphs1, mol_graphs2):
                            if [reac, prod] not in valid_reactions_dict[key]:
                                valid_reactions_dict[key].append([reac, prod])
                    elif (len(split_reac) == 1 and len(split_prod) == 2):
                        mol_graphs1 = [self.unique_mol_graphs_new[int(split_prod[0])],
                                       self.unique_mol_graphs_new[int(split_prod[1])]]
                        mol_graphs2 = [self.unique_mol_graphs_new[int(split_reac[0])]]
                        if identify_reactions_AB_C(mol_graphs1, mol_graphs2):
                            if [reac, prod] not in valid_reactions_dict[key]:
                                valid_reactions_dict[key].append([reac, prod])
                    elif (len(split_reac) == 2 and len(split_prod) == 2):
                        # self reaction
                        if (split_reac[0] in split_prod) or (split_reac[1] in split_prod):
                            new_split_reac = None
                            new_split_prod = None
                            if (split_reac[0] in split_prod):
                                prod_index = split_prod.index(split_reac[0])
                                new_split_reac = split_reac[1]
                                if prod_index == 0:
                                    new_split_prod = split_prod[1]
                                elif prod_index == 1:
                                    new_split_prod = split_prod[0]
                            elif (split_reac[1] in split_prod):
                                prod_index = split_prod.index(split_reac[1])
                                new_split_reac = split_reac[0]
                                if prod_index == 0:
                                    new_split_prod = split_prod[1]
                                elif prod_index == 1:
                                    new_split_prod = split_prod[0]
                            mol_graph1 = self.unique_mol_graphs_new[int(new_split_reac)]
                            mol_graph2 = self.unique_mol_graphs_new[int(new_split_prod)]
                            if identify_self_reactions(mol_graph1, mol_graph2):
                                if [new_split_reac, new_split_prod] not in valid_reactions_dict[key]:
                                    valid_reactions_dict[key].append([new_split_reac, new_split_prod])
                        # A + B -> C + D
                        else:
                            mol_graphs1 = [self.unique_mol_graphs_new[int(split_reac[0])],
                                           self.unique_mol_graphs_new[int(split_reac[1])]]
                            mol_graphs2 = [self.unique_mol_graphs_new[int(split_prod[0])],
                                           self.unique_mol_graphs_new[int(split_prod[1])]]
                            if identify_reactions_AB_CD(mol_graphs1, mol_graphs2):
                                if [reac, prod] not in valid_reactions_dict[key]:
                                    valid_reactions_dict[key].append([reac, prod])
        dumpfn(valid_reactions_dict, name+ "_valid_concerted_rxns_" + str(key) + ".json")
        return valid_reactions_dict

    def find_concerted_general_multiprocess_equal(self, args):
        i, name = args[0], args[1]
        valid_reactions = []

        reac = self.concerted_rxns_to_determine[i][0]
        prod = self.concerted_rxns_to_determine[i][1]

        print('reactant:', reac)
        print('product:', prod)
        split_reac = reac.split('_')
        split_prod = prod.split('_')
        if (len(split_reac) == 1 and len(split_prod) == 1):
            mol_graph1 = self.unique_mol_graphs_new[int(split_reac[0])]
            mol_graph2 = self.unique_mol_graphs_new[int(split_prod[0])]
            if identify_self_reactions(mol_graph1, mol_graph2):
                if [reac, prod] not in valid_reactions:
                    valid_reactions.append([reac, prod])
        elif (len(split_reac) == 2 and len(split_prod) == 1):
            assert split_prod[0] not in split_reac
            mol_graphs1 = [self.unique_mol_graphs_new[int(split_reac[0])],
                           self.unique_mol_graphs_new[int(split_reac[1])]]
            mol_graphs2 = [self.unique_mol_graphs_new[int(split_prod[0])]]
            if identify_reactions_AB_C(mol_graphs1, mol_graphs2):
                if [reac, prod] not in valid_reactions:
                    valid_reactions.append([reac, prod])
        elif (len(split_reac) == 1 and len(split_prod) == 2):
            mol_graphs1 = [self.unique_mol_graphs_new[int(split_prod[0])],
                           self.unique_mol_graphs_new[int(split_prod[1])]]
            mol_graphs2 = [self.unique_mol_graphs_new[int(split_reac[0])]]
            if identify_reactions_AB_C(mol_graphs1, mol_graphs2):
                if [reac, prod] not in valid_reactions:
                    valid_reactions.append([reac, prod])
        elif (len(split_reac) == 2 and len(split_prod) == 2):
            # self reaction
            if (split_reac[0] in split_prod) or (split_reac[1] in split_prod):
                new_split_reac = None
                new_split_prod = None
                if (split_reac[0] in split_prod):
                    prod_index = split_prod.index(split_reac[0])
                    new_split_reac = split_reac[1]
                    if prod_index == 0:
                        new_split_prod = split_prod[1]
                    elif prod_index == 1:
                        new_split_prod = split_prod[0]
                elif (split_reac[1] in split_prod):
                    prod_index = split_prod.index(split_reac[1])
                    new_split_reac = split_reac[0]
                    if prod_index == 0:
                        new_split_prod = split_prod[1]
                    elif prod_index == 1:
                        new_split_prod = split_prod[0]
                mol_graph1 = self.unique_mol_graphs_new[int(new_split_reac)]
                mol_graph2 = self.unique_mol_graphs_new[int(new_split_prod)]
                if identify_self_reactions(mol_graph1, mol_graph2):
                    if [new_split_reac, new_split_prod] not in valid_reactions:
                        valid_reactions.append([new_split_reac, new_split_prod])
            # A + B -> C + D
            else:
                mol_graphs1 = [self.unique_mol_graphs_new[int(split_reac[0])],
                               self.unique_mol_graphs_new[int(split_reac[1])]]
                mol_graphs2 = [self.unique_mol_graphs_new[int(split_prod[0])],
                               self.unique_mol_graphs_new[int(split_prod[1])]]
                if identify_reactions_AB_CD(mol_graphs1, mol_graphs2):
                    if [reac, prod] not in valid_reactions:
                        valid_reactions.append([reac, prod])
        return valid_reactions

    def find_concerted_break1_form1_multiprocess_equal(self, args):
        i, name = args[0], args[1]
        valid_reactions = []

        reac = self.concerted_rxns_to_determine[i][0]
        prod = self.concerted_rxns_to_determine[i][1]

        print('reactant:', reac)
        print('product:', prod)
        split_reac = reac.split('_')
        split_prod = prod.split('_')
        if (len(split_reac) == 1 and len(split_prod) == 1):
            mol_graph1 = self.unique_mol_graphs_new[int(split_reac[0])]
            mol_graph2 = self.unique_mol_graphs_new[int(split_prod[0])]
            if identify_self_reactions(mol_graph1, mol_graph2):
                if [reac, prod] not in valid_reactions:
                    valid_reactions.append([reac, prod])
        elif (len(split_reac) == 2 and len(split_prod) == 1):
            assert split_prod[0] not in split_reac
            mol_graphs1 = [self.unique_mol_graphs_new[int(split_reac[0])],
                           self.unique_mol_graphs_new[int(split_reac[1])]]
            mol_graphs2 = [self.unique_mol_graphs_new[int(split_prod[0])]]
            if identify_reactions_AB_C_break1_form1(mol_graphs1, mol_graphs2):
                if [reac, prod] not in valid_reactions:
                    valid_reactions.append([reac, prod])
        elif (len(split_reac) == 1 and len(split_prod) == 2):
            mol_graphs1 = [self.unique_mol_graphs_new[int(split_prod[0])],
                           self.unique_mol_graphs_new[int(split_prod[1])]]
            mol_graphs2 = [self.unique_mol_graphs_new[int(split_reac[0])]]
            if identify_reactions_AB_C_break1_form1(mol_graphs1, mol_graphs2):
                if [reac, prod] not in valid_reactions:
                    valid_reactions.append([reac, prod])
        elif (len(split_reac) == 2 and len(split_prod) == 2):
            # self reaction
            if (split_reac[0] in split_prod) or (split_reac[1] in split_prod):
                new_split_reac = None
                new_split_prod = None
                if (split_reac[0] in split_prod):
                    prod_index = split_prod.index(split_reac[0])
                    new_split_reac = split_reac[1]
                    if prod_index == 0:
                        new_split_prod = split_prod[1]
                    elif prod_index == 1:
                        new_split_prod = split_prod[0]
                elif (split_reac[1] in split_prod):
                    prod_index = split_prod.index(split_reac[1])
                    new_split_reac = split_reac[0]
                    if prod_index == 0:
                        new_split_prod = split_prod[1]
                    elif prod_index == 1:
                        new_split_prod = split_prod[0]
                mol_graph1 = self.unique_mol_graphs_new[int(new_split_reac)]
                mol_graph2 = self.unique_mol_graphs_new[int(new_split_prod)]
                if identify_self_reactions(mol_graph1, mol_graph2):
                    if [new_split_reac, new_split_prod] not in valid_reactions:
                        valid_reactions.append([new_split_reac, new_split_prod])
            # A + B -> C + D
            else:
                mol_graphs1 = [self.unique_mol_graphs_new[int(split_reac[0])],
                               self.unique_mol_graphs_new[int(split_reac[1])]]
                mol_graphs2 = [self.unique_mol_graphs_new[int(split_prod[0])],
                               self.unique_mol_graphs_new[int(split_prod[1])]]
                if identify_reactions_AB_CD_break1_form1(mol_graphs1, mol_graphs2):
                    if [reac, prod] not in valid_reactions:
                        valid_reactions.append([reac, prod])
        return valid_reactions

    def multiprocess(self,name, num_processors):
        keys = self.reac_prod_dict.keys()
        #keys = [83, 79, 77]
        args = [(key, name) for key in keys]
        pool = Pool(num_processors)
        results = pool.map(self.find_concerted_general_multiprocess,args)
        self.valid_reactions_dict = {}
        for i in range(len(results)):
            valid_reactions_dict = results[i]
            key = list(valid_reactions_dict.keys())[0]
            self.valid_reactions_dict[key] = valid_reactions_dict[key]
        dumpfn(self.valid_reactions_dict, name + "_valid_concerted_rxns_all.json")
        return

    def multiprocess_equal(self,name, num_processors):
        nums = list(np.arange(len(self.concerted_rxns_to_determine)))
        #nums = [0,1,2,3,4,5,6]
        #keys = [83, 79, 77]
        args = [(i, name) for i in nums]
        pool = Pool(num_processors)
        results = pool.map(self.find_concerted_general_multiprocess_equal,args)
        self.valid_reactions = []
        for i in range(len(results)):
            valid_reactions = results[i]
            self.valid_reactions += valid_reactions
        dumpfn(self.valid_reactions, name + "_valid_concerted_rxns_all.json")
        return

    def multiprocess_break1_form1_equal(self,name, num_processors):
        nums = list(np.arange(len(self.concerted_rxns_to_determine)))
        #nums = [0,1,2,3,4,5,6]
        #keys = [83, 79, 77]
        args = [(i, name) for i in nums]
        pool = Pool(num_processors)
        results = pool.map(self.find_concerted_break1_form1_multiprocess_equal,args)
        self.valid_reactions = []
        for i in range(len(results)):
            valid_reactions = results[i]
            self.valid_reactions += valid_reactions
        dumpfn(self.valid_reactions, name + "_valid_concerted_rxns_all_break1_form1.json")
        return

    def find_concerted_general(self,name):
        # Add general concerted reactions (max break 2 form 2 bonds)
        self.unique_mol_graphs = []
        for entry in self.entries_list:
            mol_graph = entry.mol_graph
            self.unique_mol_graphs.append(mol_graph)
        # find all molecule pairs that satisfy the stoichiometry constraint
        self.stoi_list, self.species_same_stoi_dict = identify_same_stoi_mol_pairs(self.unique_mol_graphs)
        self.reac_prod_dict = {}
        for i, key in enumerate(self.species_same_stoi_dict.keys()):
            species_list = self.species_same_stoi_dict[key]
            new_species_list_reactant = []
            new_species_list_product = []
            for species in species_list:
                new_species_list_reactant.append(species)
                new_species_list_product.append(species)
            if new_species_list_reactant != [] and new_species_list_product != []:
                self.reac_prod_dict[key] = {'reactants': new_species_list_reactant, 'products': new_species_list_product}

        self.valid_reactions_dict = {}
        for i, key in enumerate(list(self.reac_prod_dict.keys())):
            print(key)
            self.valid_reactions_dict[key] = []
            reactants = self.reac_prod_dict[key]['reactants']
            products = self.reac_prod_dict[key]['products']
            for j in range(len(reactants)):
                reac = reactants[j]
                for k in range(len(products)):
                    prod = products[k]
                    if k <= j:
                        continue
                    else:
                        print('reactant:', reac)
                        print('product:', prod)
                        split_reac = reac.split('_')
                        split_prod = prod.split('_')
                        if (len(split_reac) == 1 and len(split_prod) == 1):
                            mol_graph1 = self.unique_mol_graphs[int(split_reac[0])]
                            mol_graph2 = self.unique_mol_graphs[int(split_prod[0])]
                            if identify_self_reactions(mol_graph1, mol_graph2):
                                if [reac, prod] not in self.valid_reactions_dict[key]:
                                    self.valid_reactions_dict[key].append([reac, prod])
                        elif (len(split_reac) == 2 and len(split_prod) == 1):
                            assert split_prod[0] not in split_reac
                            mol_graphs1 = [self.unique_mol_graphs[int(split_reac[0])],
                                           self.unique_mol_graphs[int(split_reac[1])]]
                            mol_graphs2 = [self.unique_mol_graphs[int(split_prod[0])]]
                            if identify_reactions_AB_C(mol_graphs1, mol_graphs2):
                                if [reac, prod] not in self.valid_reactions_dict[key]:
                                    self.valid_reactions_dict[key].append([reac, prod])
                        elif (len(split_reac) == 1 and len(split_prod) == 2):
                            mol_graphs1 = [self.unique_mol_graphs[int(split_prod[0])],
                                           self.unique_mol_graphs[int(split_prod[1])]]
                            mol_graphs2 = [self.unique_mol_graphs[int(split_reac[0])]]
                            if identify_reactions_AB_C(mol_graphs1, mol_graphs2):
                                if [reac, prod] not in self.valid_reactions_dict[key]:
                                    self.valid_reactions_dict[key].append([reac, prod])
                        elif (len(split_reac) == 2 and len(split_prod) == 2):
                            # self reaction
                            if (split_reac[0] in split_prod) or (split_reac[1] in split_prod):
                                new_split_reac = None
                                new_split_prod = None
                                if (split_reac[0] in split_prod):
                                    prod_index = split_prod.index(split_reac[0])
                                    new_split_reac = split_reac[1]
                                    if prod_index == 0:
                                        new_split_prod = split_prod[1]
                                    elif prod_index == 1:
                                        new_split_prod = split_prod[0]
                                elif (split_reac[1] in split_prod):
                                    prod_index = split_prod.index(split_reac[1])
                                    new_split_reac = split_reac[0]
                                    if prod_index == 0:
                                        new_split_prod = split_prod[1]
                                    elif prod_index == 1:
                                        new_split_prod = split_prod[0]
                                mol_graph1 = self.unique_mol_graphs[int(new_split_reac)]
                                mol_graph2 = self.unique_mol_graphs[int(new_split_prod)]
                                if identify_self_reactions(mol_graph1, mol_graph2):
                                    if [new_split_reac, new_split_prod] not in self.valid_reactions_dict[key]:
                                        self.valid_reactions_dict[key].append([new_split_reac, new_split_prod])
                            # A + B -> C + D
                            else:
                                mol_graphs1 = [self.unique_mol_graphs[int(split_reac[0])],
                                               self.unique_mol_graphs[int(split_reac[1])]]
                                mol_graphs2 = [self.unique_mol_graphs[int(split_prod[0])],
                                               self.unique_mol_graphs[int(split_prod[1])]]
                                if identify_reactions_AB_CD(mol_graphs1, mol_graphs2):
                                    if [reac, prod] not in self.valid_reactions_dict[key]:
                                        self.valid_reactions_dict[key].append([reac, prod])
            dumpfn(self.valid_reactions_dict,name+"_valid_concerted_rxns_"+str(key)+".json")

    def find_concerted_general_2(self,name):
        # Add general concerted reactions (max break 2 form 2 bonds)
        # Only consider unique mol graphs to identify concerted rxns and then broadcast back
        self.unique_mol_graphs = []
        for entry in self.entries_list:
            mol_graph = entry.mol_graph
            self.unique_mol_graphs.append(mol_graph)

        self.unique_mol_graphs_new = []
        # For duplicate mol graphs, create a map between later species with former ones
        self.unique_mol_graph_dict = {}

        for i in range(len(self.unique_mol_graphs)):
            mol_graph = self.unique_mol_graphs[i]
            found = False
            for j in range(len(self.unique_mol_graphs_new)):
                new_mol_graph = self.unique_mol_graphs_new[j]
                if mol_graph.isomorphic_to(new_mol_graph):
                    found = True
                    self.unique_mol_graph_dict[i] = j
                    continue
            if not found:
                self.unique_mol_graph_dict[i] = len(self.unique_mol_graphs_new)
                self.unique_mol_graphs_new.append(mol_graph)
        dumpfn(self.unique_mol_graph_dict, name + "_unique_mol_graph_map.json")
        # find all molecule pairs that satisfy the stoichiometry constraint
        self.stoi_list, self.species_same_stoi_dict = identify_same_stoi_mol_pairs(self.unique_mol_graphs_new)
        self.reac_prod_dict = {}
        for i, key in enumerate(self.species_same_stoi_dict.keys()):
            species_list = self.species_same_stoi_dict[key]
            new_species_list_reactant = []
            new_species_list_product = []
            for species in species_list:
                new_species_list_reactant.append(species)
                new_species_list_product.append(species)
            if new_species_list_reactant != [] and new_species_list_product != []:
                self.reac_prod_dict[key] = {'reactants': new_species_list_reactant, 'products': new_species_list_product}

        self.valid_reactions_dict = {}
        for i, key in enumerate(list(self.reac_prod_dict.keys())):
            print(key)
            self.valid_reactions_dict[key] = []
            reactants = self.reac_prod_dict[key]['reactants']
            products = self.reac_prod_dict[key]['products']
            for j in range(len(reactants)):
                reac = reactants[j]
                for k in range(len(products)):
                    prod = products[k]
                    if k <= j:
                        continue
                    else:
                        print('reactant:', reac)
                        print('product:', prod)
                        split_reac = reac.split('_')
                        split_prod = prod.split('_')
                        if (len(split_reac) == 1 and len(split_prod) == 1):
                            mol_graph1 = self.unique_mol_graphs_new[int(split_reac[0])]
                            mol_graph2 = self.unique_mol_graphs_new[int(split_prod[0])]
                            if identify_self_reactions(mol_graph1, mol_graph2):
                                if [reac, prod] not in self.valid_reactions_dict[key]:
                                    self.valid_reactions_dict[key].append([reac, prod])
                        elif (len(split_reac) == 2 and len(split_prod) == 1):
                            assert split_prod[0] not in split_reac
                            mol_graphs1 = [self.unique_mol_graphs_new[int(split_reac[0])],
                                           self.unique_mol_graphs_new[int(split_reac[1])]]
                            mol_graphs2 = [self.unique_mol_graphs_new[int(split_prod[0])]]
                            if identify_reactions_AB_C(mol_graphs1, mol_graphs2):
                                if [reac, prod] not in self.valid_reactions_dict[key]:
                                    self.valid_reactions_dict[key].append([reac, prod])
                        elif (len(split_reac) == 1 and len(split_prod) == 2):
                            mol_graphs1 = [self.unique_mol_graphs_new[int(split_prod[0])],
                                           self.unique_mol_graphs_new[int(split_prod[1])]]
                            mol_graphs2 = [self.unique_mol_graphs_new[int(split_reac[0])]]
                            if identify_reactions_AB_C(mol_graphs1, mol_graphs2):
                                if [reac, prod] not in self.valid_reactions_dict[key]:
                                    self.valid_reactions_dict[key].append([reac, prod])
                        elif (len(split_reac) == 2 and len(split_prod) == 2):
                            # self reaction
                            if (split_reac[0] in split_prod) or (split_reac[1] in split_prod):
                                new_split_reac = None
                                new_split_prod = None
                                if (split_reac[0] in split_prod):
                                    prod_index = split_prod.index(split_reac[0])
                                    new_split_reac = split_reac[1]
                                    if prod_index == 0:
                                        new_split_prod = split_prod[1]
                                    elif prod_index == 1:
                                        new_split_prod = split_prod[0]
                                elif (split_reac[1] in split_prod):
                                    prod_index = split_prod.index(split_reac[1])
                                    new_split_reac = split_reac[0]
                                    if prod_index == 0:
                                        new_split_prod = split_prod[1]
                                    elif prod_index == 1:
                                        new_split_prod = split_prod[0]
                                mol_graph1 = self.unique_mol_graphs_new[int(new_split_reac)]
                                mol_graph2 = self.unique_mol_graphs_new[int(new_split_prod)]
                                if identify_self_reactions(mol_graph1, mol_graph2):
                                    if [new_split_reac, new_split_prod] not in self.valid_reactions_dict[key]:
                                        self.valid_reactions_dict[key].append([new_split_reac, new_split_prod])
                            # A + B -> C + D
                            else:
                                mol_graphs1 = [self.unique_mol_graphs_new[int(split_reac[0])],
                                               self.unique_mol_graphs_new[int(split_reac[1])]]
                                mol_graphs2 = [self.unique_mol_graphs_new[int(split_prod[0])],
                                               self.unique_mol_graphs_new[int(split_prod[1])]]
                                if identify_reactions_AB_CD(mol_graphs1, mol_graphs2):
                                    if [reac, prod] not in self.valid_reactions_dict[key]:
                                        self.valid_reactions_dict[key].append([reac, prod])
            dumpfn(self.valid_reactions_dict, name+ "_valid_concerted_rxns_" + str(key) + ".json")

    def add_concerted_reactions(self):
        reactions_to_add = []
        for key in self.valid_reactions_dict.keys():
            for i in range(len(self.valid_reactions_dict[key])):
                rxn = self.valid_reactions_dict[key][i]
                reactant_nodes = rxn[0].split('_')
                product_nodes = rxn[1].split('_')
                entries0 = []
                entries1 = []
                for ind in reactant_nodes:
                    entries0.append(self.entries_list[int(ind)])
                for ind in product_nodes:
                    entries1.append(self.entries_list[int(ind)])
                if len(entries0) == 1 and len(entries1) == 1:
                    if rxn[0]+','+rxn[1] not in self.graph.nodes:
                        reactions_to_add.append([entries0, entries1])
                elif len(entries0) == 2 and len(entries1) == 1:
                    if int(reactant_nodes[0]) <= int(reactant_nodes[1]):
                        reactant_name = rxn[0].replace('_','+')
                    else:
                        reactant_name = rxn[0].split('_')[1] + '+' + rxn[0].split('_')[0]
                    if rxn[1]+','+reactant_name not in self.graph.nodes:
                        reactions_to_add.append([entries0, entries1])
                elif len(entries0) == 1 and len(entries1) == 2:
                    if int(product_nodes[0]) <= int(product_nodes[1]):
                        product_name = rxn[1].replace('_','+')
                    else:
                        product_name = rxn[1].split('_')[1] + '+' + rxn[1].split('_')[0]
                    if rxn[0]+','+product_name not in self.graph.nodes:
                        reactions_to_add.append([entries0, entries1])
                elif len(entries0) == 2 and len(entries1) == 2:
                    if int(product_nodes[0]) <= int(product_nodes[1]):
                        product_name = rxn[1].replace('_','+')
                    else:
                        product_name = rxn[1].split('_')[1] + '+' + rxn[1].split('_')[0]
                    reactant_name = reactant_nodes[0] + '+PR_' + reactant_nodes[1]
                    if reactant_name+','+product_name not in self.graph.nodes:
                        reactions_to_add.append([entries0, entries1])
        for to_add in reactions_to_add:
            print(len(to_add[0]),len(to_add[1]))
            self.add_reaction(to_add[0],to_add[1],"concerted")

    def add_concerted_reactions_2(self, read_file=False, file_name=None, num=None):
        # Add concerted reactions from find_concerted_general_2 function
        reactions_to_add = []
        if read_file:
            self.valid_reactions_dict = loadfn(file_name+"_valid_concerted_rxns_" + str(num) + ".json")
            self.unique_mol_graph_dict = loadfn(file_name+"_unique_mol_graph_map.json")
        for key in self.valid_reactions_dict.keys():
            for i in range(len(self.valid_reactions_dict[key])):
                rxn = self.valid_reactions_dict[key][i]
                reactant_nodes = rxn[0].split('_')
                product_nodes = rxn[1].split('_')
                reactant_candidates = []
                product_candidates = []
                for reac in reactant_nodes:
                    reac_cands = []
                    for map_key in self.unique_mol_graph_dict.keys():
                        if self.unique_mol_graph_dict[map_key] == int(reac):
                            reac_cands.append(map_key)
                    reactant_candidates.append(reac_cands)
                for prod in product_nodes:
                    prod_cands = []
                    for map_key in self.unique_mol_graph_dict.keys():
                        if self.unique_mol_graph_dict[map_key] == int(prod):
                            prod_cands.append(map_key)
                    product_candidates.append(prod_cands)
                print('reactant candidates:',reactant_candidates)
                print('product candidates:',product_candidates)

                if len(reactant_candidates) == 1 and len(product_candidates) == 1:
                    for j in reactant_candidates[0]:
                        for k in product_candidates[0]:
                            entries0 = []
                            entries1 = []
                            entries0.append(self.entries_list[int(j)])
                            entries1.append(self.entries_list[int(k)])
                            if str(j) + ',' + str(k) not in self.graph.nodes:
                                reactions_to_add.append([entries0, entries1])

                elif len(reactant_candidates) == 2 and len(product_candidates) == 1:
                    for j in reactant_candidates[0]:
                        for k in reactant_candidates[1]:
                            for m in product_candidates[0]:
                                entries0 = []
                                entries1 = []
                                entries0.append(self.entries_list[int(j)])
                                entries0.append(self.entries_list[int(k)])
                                entries1.append(self.entries_list[int(m)])
                                if int(j) <= int(k):
                                    reactant_name = str(j) + '+' + str(k)
                                else:
                                    reactant_name = str(k) + '+' + str(j)
                                if str(m) + ',' + reactant_name not in self.graph.nodes:
                                    reactions_to_add.append([entries0, entries1])

                elif len(reactant_candidates) == 1 and len(product_candidates) == 2:
                    for j in reactant_candidates[0]:
                        for m in product_candidates[0]:
                            for n in product_candidates[1]:
                                entries0 = []
                                entries1 = []
                                entries0.append(self.entries_list[int(j)])
                                entries1.append(self.entries_list[int(m)])
                                entries1.append(self.entries_list[int(n)])
                                if int(m) <= int(n):
                                    product_name = str(m) + '+' + str(n)
                                else:
                                    product_name = str(n) + '+' + str(m)
                                if str(j) + ',' + product_name not in self.graph.nodes:
                                    reactions_to_add.append([entries0, entries1])

                elif len(reactant_candidates) == 2 and len(product_candidates) == 2:
                    for j in reactant_candidates[0]:
                        for k in reactant_candidates[1]:
                            for m in product_candidates[0]:
                                for n in product_candidates[1]:
                                    entries0 = []
                                    entries1 = []
                                    entries0.append(self.entries_list[int(j)])
                                    entries0.append(self.entries_list[int(k)])
                                    entries1.append(self.entries_list[int(m)])
                                    entries1.append(self.entries_list[int(n)])
                                    if int(m) <= int(n):
                                        product_name = str(m) + '+' + str(n)
                                    else:
                                        product_name = str(n) + '+' + str(m)
                                    reactant_name = str(j) + '+PR_' + str(k)
                                    if reactant_name + ',' + product_name not in self.graph.nodes:
                                        reactions_to_add.append([entries0, entries1])

        for to_add in reactions_to_add:
            print(len(to_add[0]),len(to_add[1]))
            self.add_reaction(to_add[0],to_add[1],"concerted")

    def add_concerted_reactions_from_list(self, read_file=False, file_name=None, break1_form1=False):
        # Add concerted reactions from self.multiprocess_equal function
        reactions_to_add = []
        if read_file:
            if break1_form1:
                self.valid_reactions = loadfn(file_name + "_valid_concerted_rxns_all_break1_form1.json")
            else:
                self.valid_reactions = loadfn(file_name+"_valid_concerted_rxns_all.json")
            self.unique_mol_graph_dict = loadfn(file_name+"_unique_mol_graph_map.json")
        for i in range(len(self.valid_reactions)):
            rxn = self.valid_reactions[i]
            reactant_nodes = rxn[0].split('_')
            product_nodes = rxn[1].split('_')
            reactant_candidates = []
            product_candidates = []
            for reac in reactant_nodes:
                reac_cands = []
                for map_key in self.unique_mol_graph_dict.keys():
                    if self.unique_mol_graph_dict[map_key] == int(reac):
                        reac_cands.append(map_key)
                reactant_candidates.append(reac_cands)
            for prod in product_nodes:
                prod_cands = []
                for map_key in self.unique_mol_graph_dict.keys():
                    if self.unique_mol_graph_dict[map_key] == int(prod):
                        prod_cands.append(map_key)
                product_candidates.append(prod_cands)
            print('reactant candidates:',reactant_candidates)
            print('product candidates:',product_candidates)

            if len(reactant_candidates) == 1 and len(product_candidates) == 1:
                for j in reactant_candidates[0]:
                    for k in product_candidates[0]:
                        entries0 = []
                        entries1 = []
                        entries0.append(self.entries_list[int(j)])
                        entries1.append(self.entries_list[int(k)])
                        if str(j) + ',' + str(k) not in self.graph.nodes:
                            reactions_to_add.append([entries0, entries1])

            elif len(reactant_candidates) == 2 and len(product_candidates) == 1:
                for j in reactant_candidates[0]:
                    for k in reactant_candidates[1]:
                        for m in product_candidates[0]:
                            entries0 = []
                            entries1 = []
                            entries0.append(self.entries_list[int(j)])
                            entries0.append(self.entries_list[int(k)])
                            entries1.append(self.entries_list[int(m)])
                            if int(j) <= int(k):
                                reactant_name = str(j) + '+' + str(k)
                            else:
                                reactant_name = str(k) + '+' + str(j)
                            if str(m) + ',' + reactant_name not in self.graph.nodes:
                                reactions_to_add.append([entries0, entries1])

            elif len(reactant_candidates) == 1 and len(product_candidates) == 2:
                for j in reactant_candidates[0]:
                    for m in product_candidates[0]:
                        for n in product_candidates[1]:
                            entries0 = []
                            entries1 = []
                            entries0.append(self.entries_list[int(j)])
                            entries1.append(self.entries_list[int(m)])
                            entries1.append(self.entries_list[int(n)])
                            if int(m) <= int(n):
                                product_name = str(m) + '+' + str(n)
                            else:
                                product_name = str(n) + '+' + str(m)
                            if str(j) + ',' + product_name not in self.graph.nodes:
                                reactions_to_add.append([entries0, entries1])

            elif len(reactant_candidates) == 2 and len(product_candidates) == 2:
                for j in reactant_candidates[0]:
                    for k in reactant_candidates[1]:
                        for m in product_candidates[0]:
                            for n in product_candidates[1]:
                                entries0 = []
                                entries1 = []
                                entries0.append(self.entries_list[int(j)])
                                entries0.append(self.entries_list[int(k)])
                                entries1.append(self.entries_list[int(m)])
                                entries1.append(self.entries_list[int(n)])
                                if int(m) <= int(n):
                                    product_name = str(m) + '+' + str(n)
                                else:
                                    product_name = str(n) + '+' + str(m)
                                reactant_name = str(j) + '+PR_' + str(k)
                                if reactant_name + ',' + product_name not in self.graph.nodes:
                                    reactions_to_add.append([entries0, entries1])

        for to_add in reactions_to_add:
            print(len(to_add[0]),len(to_add[1]))
            self.add_reaction(to_add[0],to_add[1],"concerted")

    def add_water_reactions(self):
        # Since concerted reactions remain intractable, this function adds two specific concerted
        # reactions involving water so that realisic paths to OH- are possible:
        # 2H2O -> OH- + H3O+
        # 2H2O + 2e- -> H2 + 2OH-
        # Note that in the 2nd reaction, H2 should be in gas phase, but all calcs are currently in SMD.
        H2O_found = False
        OHminus_found = False
        H3Oplus_found = False
        H2_found = False
        try:
            H2O_entry = self.entries["H2 O1"][2][0][0]
            # print("H2O_entry",H2O_entry)
            H2O_found = True
        except KeyError:
            print("Missing H2O, will not add either concerted water splitting reaction")

        try:
            OHminus_entry = self.entries["H1 O1"][1][-1][0]
            # print("OHminus_entry",OHminus_entry)
            OHminus_found = True
        except KeyError:
            print("Missing OH-, will not add either concerted water splitting reaction")

        if H2O_found and OHminus_found:
            try:
                H3Oplus_entry = self.entries["H3 O1"][3][1][0]
                # print("H3Oplus_entry",H3Oplus_entry)
                H3Oplus_found = True
            except KeyError:
                print("Missing H3O+, will not add concerted water splitting rxn1")

            try:
                H2_entry = self.entries["H2"][1][0][0]
                # print("H2_entry",H2_entry)
                H2_found = True
            except KeyError:
                print("Missing H2, will not add concerted water splitting rxn2")

            if H3Oplus_found or H2_found:
                H2O_PR_H2O_name = str(H2O_entry.parameters["ind"])+"+PR_"+str(H2O_entry.parameters["ind"])

                if H3Oplus_found:
                    print("Adding concerted water splitting rxn1: 2H2O -> OH- + H3O+")

                    if OHminus_entry.parameters["ind"] <= H3Oplus_entry.parameters["ind"]:
                        OHminus_H3Oplus_name = str(OHminus_entry.parameters["ind"])+"+"+str(H3Oplus_entry.parameters["ind"])
                    else:
                        OHminus_H3Oplus_name = str(H3Oplus_entry.parameters["ind"])+"+"+str(OHminus_entry.parameters["ind"])

                    rxn_node_1 = H2O_PR_H2O_name+","+OHminus_H3Oplus_name
                    rxn1_energy = OHminus_entry.energy + H3Oplus_entry.energy - 2*H2O_entry.energy
                    rxn1_free_energy = OHminus_entry.free_energy + H3Oplus_entry.free_energy - 2*H2O_entry.free_energy
                    print("Rxn1 free energy =",rxn1_free_energy)

                    self.graph.add_node(rxn_node_1,rxn_type="water_dissociation",bipartite=1,energy=rxn1_energy,free_energy=rxn1_free_energy)
                    self.graph.add_edge(H2O_entry.parameters["ind"],
                                        rxn_node_1,
                                        softplus=self.softplus(rxn1_free_energy),
                                        exponent=self.exponent(rxn1_free_energy),
                                        weight=1.0
                                        )
                    self.graph.add_edge(rxn_node_1,
                                        OHminus_entry.parameters["ind"],
                                        softplus=0.0,
                                        exponent=0.0,
                                        weight=1.0
                                        )
                    self.graph.add_edge(rxn_node_1,
                                        H3Oplus_entry.parameters["ind"],
                                        softplus=0.0,
                                        exponent=0.0,
                                        weight=1.0
                                        )

                if H2_found:
                    print("Adding concerted water splitting rxn2: 2H2O + 2e- -> H2 + 2OH-")

                    OHminus2_H2_name = str(OHminus_entry.parameters["ind"])+"+"+str(OHminus_entry.parameters["ind"])+"+"+str(H2_entry.parameters["ind"])
                    rxn_node_2 = H2O_PR_H2O_name+","+OHminus2_H2_name
                    rxn2_energy = 2*OHminus_entry.energy + H2_entry.energy - 2*H2O_entry.energy
                    rxn2_free_energy = 2*OHminus_entry.free_energy + H2_entry.free_energy - 2*H2O_entry.free_energy - 2*self.electron_free_energy
                    print("Water rxn2 free energy =",rxn2_free_energy)

                    self.graph.add_node(rxn_node_2,rxn_type="water_2e_redox",bipartite=1,energy=rxn2_energy,free_energy=rxn2_free_energy)
                    self.graph.add_edge(H2O_entry.parameters["ind"],
                                        rxn_node_2,
                                        softplus=self.softplus(rxn2_free_energy),
                                        exponent=self.exponent(rxn2_free_energy),
                                        weight=1.0
                                        )
                    self.graph.add_edge(rxn_node_2,
                                        OHminus_entry.parameters["ind"],
                                        softplus=0.0,
                                        exponent=0.0,
                                        weight=1.0
                                        )
                    self.graph.add_edge(rxn_node_2,
                                        H2_entry.parameters["ind"],
                                        softplus=0.0,
                                        exponent=0.0,
                                        weight=1.0
                                        )
    def add_water_lithium_reaction(self):
        # 2Li+ + 2H2O + 2e- -> 2LiOH +H2
        H2O_found = False
        Li_plus_found = False
        LiOH_found = False
        H2_found = False
        try:
            H2O_entry = self.entries["H2 O1"][2][0][0]
            # print("H2O_entry",H2O_entry)
            H2O_found = True
        except KeyError:
            print("Missing H2O, will not add 2Li+ + 2H2O + 2e- -> 2LiOH +H2 reaction")

        try:
            Li_plus_entry = self.entries["Li1"][0][1][0]
            # print("OHminus_entry",OHminus_entry)
            Li_plus_found = True
        except KeyError:
            print("Missing OH-, will not add 2Li+ + 2H2O + 2e- -> 2LiOH +H2 reaction")

        try:
            LiOH_entry = self.entries['H1 Li1 O1'][2][0][0]
            # print("H3Oplus_entry",H3Oplus_entry)
            LiOH_found = True
        except KeyError:
            print("Missing H3O+, will not add 2Li+ + 2H2O + 2e- -> 2LiOH +H2 reaction")

        try:
            H2_entry = self.entries["H2"][1][0][0]
            # print("H2_entry",H2_entry)
            H2_found = True
        except KeyError:
            print("Missing H2, will not add 2Li+ + 2H2O + 2e- -> 2LiOH +H2 reaction")

        if H2O_found and Li_plus_found and LiOH_found and H2_found:
            print("Adding concerted water splitting rxn1: 2H2O -> OH- + H3O+")
            if H2O_entry.parameters["ind"] <= Li_plus_entry.parameters["ind"]:
                H2O_Li_name = str(H2O_entry.parameters["ind"])+ "+" +str(Li_plus_entry.parameters["ind"])
            else:
                H2O_Li_name = str(Li_plus_entry.parameters["ind"])+ "+" +str(H2O_entry.parameters["ind"])

            if LiOH_entry.parameters["ind"] <= H2_entry.parameters["ind"]:
                LiOH_H2_name = str(LiOH_entry.parameters["ind"]) + "+" + str(H2_entry.parameters["ind"])
            else:
                LiOH_H2_name = str(H2_entry.parameters["ind"]) + "+" + str(LiOH_entry.parameters["ind"])

            rxn_node = H2O_Li_name + "," + LiOH_H2_name
            rxn_energy = 2 * LiOH_entry.energy + H2_entry.energy - 2 * H2O_entry.energy - 2 * Li_plus_entry.energy
            rxn_free_energy = 2 * LiOH_entry.free_energy + H2_entry.free_energy - 2 * H2O_entry.free_energy - \
                              2 * Li_plus_entry.free_energy - 2 * self.electron_free_energy
            print("Rxn free energy =", rxn_free_energy)

            self.graph.add_node(rxn_node,rxn_type="water_lithium_reaction",bipartite=1,energy=rxn_energy,free_energy=rxn_free_energy)
            self.graph.add_edge(H2O_entry.parameters["ind"],
                                rxn_node,
                                softplus=self.softplus(rxn_free_energy),
                                exponent=self.exponent(rxn_free_energy),
                                weight=1.0
                                )
            self.graph.add_edge(Li_plus_entry.parameters["ind"],
                                rxn_node,
                                softplus=self.softplus(rxn_free_energy),
                                exponent=self.exponent(rxn_free_energy),
                                weight=1.0
                                )
            self.graph.add_edge(rxn_node,
                                LiOH_entry.parameters["ind"],
                                softplus=0.0,
                                exponent=0.0,
                                weight=1.0
                                )
            self.graph.add_edge(rxn_node,
                                H2_entry.parameters["ind"],
                                softplus=0.0,
                                exponent=0.0,
                                weight=1.0
                                )

    def add_LEDC_concerted_reactions(self):
        # 2LiEC-RO -> LEDC + C2H4
        # LiCO3 -1 + LiEC 1 -> LEDC
        # LiEC-RO: 'C3 H4 Li1 O3'

        LEDC_found = False
        C2H4_found = False
        LiEC_RO_found = False
        LiCO3_minus_found = False
        LiEC_plus_found = False
        '''
        try:
            LEDC_entry = self.entries['C4 H4 Li2 O6'][17][0][0]
            LEDC_found = True
        except KeyError:
            print("Missing LEDC, will not add either 2LiEC-RO -> LEDC + C2H4 reaction or LiCO3 -1 + LiEC 1 -> LEDC reactions")

        try:
            LiEC_RO_entry = self.entries['C3 H4 Li1 O3'][11][0][0]
            LiEC_RO_found = True
        except KeyError:
            print("Missing LiEC-RO, will not add 2LiEC-RO -> LEDC + C2H4 reaction")

        try:
            C2H4_entry = self.entries['C2 H4'][5][0][0]
            C2H4_found = True
        except KeyError:
            print("Missing C2H4, will not add 2LiEC-RO -> LEDC + C2H4 reaction")

        try:
            LiCO3_minus_entry = self.entries['C1 Li1 O3'][5][-1][0]
            LiCO3_minus_found = True
        except KeyError:
            print("Missing LiCO3-, will not add LiCO3 -1 + LiEC 1 -> LEDC reaction")

        try:
            LiEC_plus_entry = self.entries['C3 H4 Li1 O3'][12][1][0]
            LiEC_plus_found = True
        except KeyError:
            print("Missing LiEC+, will not add LiCO3 -1 + LiEC 1 -> LEDC reaction")
        
        '''

        try:
            LEDC_entry = self.entries['C4 H4 Li2 O6'][15][0][0]
            LEDC_found = True
        except KeyError:
            print("Missing LEDC, will not add either 2LiEC-RO -> LEDC + C2H4 reaction or LiCO3 -1 + LiEC 1 -> LEDC reactions")

        try:
            LiEC_RO_entry = self.entries['C3 H4 Li1 O3'][10][0][1]
            LiEC_RO_found = True
        except KeyError:
            print("Missing LiEC-RO, will not add 2LiEC-RO -> LEDC + C2H4 reaction")

        try:
            C2H4_entry = self.entries['C2 H4'][5][0][0]
            C2H4_found = True
        except KeyError:
            print("Missing C2H4, will not add 2LiEC-RO -> LEDC + C2H4 reaction")

        try:
            LiCO3_minus_entry = self.entries['C1 Li1 O3'][4][-1][0]
            LiCO3_minus_found = True
        except KeyError:
            print("Missing LiCO3-, will not add LiCO3 -1 + LiEC 1 -> LEDC reaction")

        try:
            LiEC_plus_entry = self.entries['C3 H4 Li1 O3'][11][1][0]
            LiEC_plus_found = True
        except KeyError:
            print("Missing LiEC+, will not add LiCO3 -1 + LiEC 1 -> LEDC reaction")



        if LiEC_RO_found and C2H4_found and LEDC_found:
            print("Adding concerted reaction 2LiEC-RO -> LEDC + C2H4")
            LiEC_RO_PR_LiEC_RO_name = str(LiEC_RO_entry.parameters["ind"]) + "+PR_" + str(LiEC_RO_entry.parameters["ind"])

            if LEDC_entry.parameters["ind"] <= C2H4_entry.parameters["ind"]:
                LEDC_C2H4_name = str(LEDC_entry.parameters["ind"]) + "+" + str(C2H4_entry.parameters["ind"])
            else:
                LEDC_C2H4_name = str(C2H4_entry.parameters["ind"]) + "+" + str(LEDC_entry.parameters["ind"])

            rxn_node_1 = LiEC_RO_PR_LiEC_RO_name + "," + LEDC_C2H4_name
            rxn1_energy = LEDC_entry.energy + C2H4_entry.energy - 2 * LiEC_RO_entry.energy
            rxn1_free_energy = LEDC_entry.free_energy + C2H4_entry.free_energy - 2 * LiEC_RO_entry.free_energy
            print("2LiEC-RO -> LEDC + C2H4 free energy =", rxn1_free_energy)

            self.graph.add_node(rxn_node_1, rxn_type="2LiEC-RO -> LEDC + C2H4", bipartite=1, energy=rxn1_energy,
                                free_energy=rxn1_free_energy)
            self.graph.add_edge(LiEC_RO_entry.parameters["ind"],
                                rxn_node_1,
                                softplus=self.softplus(rxn1_free_energy),
                                exponent=self.exponent(rxn1_free_energy),
                                weight=1.0
                                )
            self.graph.add_edge(rxn_node_1,
                                LEDC_entry.parameters["ind"],
                                softplus=0.0,
                                exponent=0.0,
                                weight=1.0
                                )
            self.graph.add_edge(rxn_node_1,
                                C2H4_entry.parameters["ind"],
                                softplus=0.0,
                                exponent=0.0,
                                weight=1.0
                                )

        if LiEC_plus_found and LiCO3_minus_found and LEDC_found:
            print("LiCO3 -1 + LiEC 1 -> LEDC")
            if LiEC_plus_entry.parameters["ind"] <= LiCO3_minus_entry.parameters["ind"]:
                LiEC_plus_LiCO3_minus_name = str(LiEC_plus_entry.parameters["ind"]) + "+" + str(LiCO3_minus_entry.parameters["ind"])
            else:
                LiEC_plus_LiCO3_minus_name = str(LiCO3_minus_entry.parameters["ind"]) + "+" + str(LiEC_plus_entry.parameters["ind"])
            LEDC_name = str(LEDC_entry.parameters["ind"])

            rxn_node_2 = LiEC_plus_LiCO3_minus_name+","+LEDC_name
            rxn2_energy =LEDC_entry.energy - LiEC_plus_entry.energy - LiCO3_minus_entry.energy
            rxn2_free_energy = LEDC_entry.free_energy - LiEC_plus_entry.free_energy - LiCO3_minus_entry.free_energy
            print("LiCO3 -1 + LiEC 1 -> LEDC free energy =",rxn2_free_energy)

            self.graph.add_node(rxn_node_2,rxn_type="LiCO3 -1 + LiEC 1 -> LEDC",bipartite=1,energy=rxn2_energy,free_energy=rxn2_free_energy)
            self.graph.add_edge(LiEC_plus_entry.parameters["ind"],
                                rxn_node_2,
                                softplus=self.softplus(rxn2_free_energy),
                                exponent=self.exponent(rxn2_free_energy),
                                weight=1.0
                                )
            self.graph.add_edge(LiCO3_minus_entry.parameters["ind"],
                                rxn_node_2,
                                softplus=0.0,
                                exponent=0.0,
                                weight=1.0
                                )
            self.graph.add_edge(rxn_node_2,
                                LEDC_entry.parameters["ind"],
                                softplus=0.0,
                                exponent=0.0,
                                weight=1.0
                                )


    def add_reaction(self,entries0,entries1,rxn_type):
        """
        Args:
            entries0 ([MoleculeEntry]): list of MoleculeEntry objects on one side of the reaction
            entries1 ([MoleculeEntry]): list of MoleculeEntry objects on the other side of the reaction
            rxn_type (string): general reaction category. At present, must be one_electron_redox or 
                              intramol_single_bond_change or intermol_single_bond_change.
        """
        self.num_reactions += 1
        if rxn_type == "one_electron_redox":
            if len(entries0) != 1 or len(entries1) != 1:
                raise RuntimeError("One electron redox requires two lists that each contain one entry!")
        elif rxn_type == "intramol_single_bond_change":
            if len(entries0) != 1 or len(entries1) != 1:
                raise RuntimeError("Intramolecular single bond change requires two lists that each contain one entry!")
        elif rxn_type == "intermol_single_bond_change":
            if len(entries0) != 1 or len(entries1) != 2:
                raise RuntimeError("Intermolecular single bond change requires two lists that contain one entry and two entries, respectively!")
        elif rxn_type == "coordination_bond_change":
            if len(entries0) != 1 or len(entries1) != 2:
                raise RuntimeError("Coordination bond change requires two lists that contain one entry and two entries, respectively!")
        elif rxn_type == "concerted":
            if len(entries0) > 2 or len(entries1) > 2:
                raise RuntimeError("Concerted reactions require two lists that each contain two or fewer entries!")
        else:
            raise RuntimeError("Reaction type "+rxn_type+" is not supported!")
        if rxn_type == "one_electron_redox" or rxn_type == "intramol_single_bond_change":# or (rxn_type == "concerted" and len(entries0) == 1 and len(entries1) == 1):
            entry0 = entries0[0]
            entry1 = entries1[0]
            if rxn_type == "one_electron_redox":
                val0 = entry0.charge
                val1 = entry1.charge
                if val1<val0:
                    rxn_type_A = "One electron reduction"
                    rxn_type_B = "One electron oxidation"
                else:
                    rxn_type_A = "One electron oxidation"
                    rxn_type_B = "One electron reduction"
            elif rxn_type == "intramol_single_bond_change":
                val0 = entry0.Nbonds
                val1 = entry1.Nbonds
                if val1<val0:
                    rxn_type_A = "Intramolecular single bond breakage"
                    rxn_type_B = "Intramolecular single bond formation"
                else:
                    rxn_type_A = "Intramolecular single bond formation"
                    rxn_type_B = "Intramolecular single bond breakage"
            elif rxn_type == "concerted":
                rxn_type_A = "Concerted"
                rxn_type_B = "Concerted"
            node_name_A = str(entry0.parameters["ind"])+","+str(entry1.parameters["ind"])
            node_name_B = str(entry1.parameters["ind"])+","+str(entry0.parameters["ind"])
            energy_A = entry1.energy-entry0.energy
            energy_B = entry0.energy-entry1.energy
            if entry1.free_energy != None and entry0.free_energy != None:
                free_energy_A = entry1.free_energy-entry0.free_energy
                free_energy_B = entry0.free_energy-entry1.free_energy
                if rxn_type == "one_electron_redox":
                    if rxn_type_A == "One electron reduction":
                        free_energy_A += -self.electron_free_energy
                        free_energy_B += self.electron_free_energy
                    else:
                        free_energy_A += self.electron_free_energy
                        free_energy_B += -self.electron_free_energy
                elif rxn_type == "concerted":
                    if entry0.charge - entry1.charge == 1:
                        free_energy_A += -self.electron_free_energy
                        free_energy_B += self.electron_free_energy
                    elif entry0.charge - entry1.charge == -1:
                        free_energy_A += self.electron_free_energy
                        free_energy_B += -self.electron_free_energy
                    elif entry0.charge != entry1.charge:
                        raise RuntimeError("Concerted charge difference of "+str(abs(entry0.charge - entry1.charge))+" detected!")
            else:
                free_energy_A = None
                free_energy_B = None

            self.graph.add_node(node_name_A,rxn_type=rxn_type_A,bipartite=1,energy=energy_A,free_energy=free_energy_A)
            self.graph.add_edge(entry0.parameters["ind"],
                                node_name_A,
                                softplus=self.softplus(free_energy_A),
                                exponent=self.exponent(free_energy_A),
                                weight=1.0)
            self.graph.add_edge(node_name_A,
                                entry1.parameters["ind"],
                                softplus=0.0,
                                exponent=0.0,
                                weight=1.0)
            self.graph.add_node(node_name_B,rxn_type=rxn_type_B,bipartite=1,energy=energy_B,free_energy=free_energy_B)
            self.graph.add_edge(entry1.parameters["ind"],
                                node_name_B,
                                softplus=self.softplus(free_energy_B),
                                exponent=self.exponent(free_energy_B),
                                weight=1.0)
            self.graph.add_edge(node_name_B,
                                entry0.parameters["ind"],
                                softplus=0.0,
                                exponent=0.0,
                                weight=1.0)

        elif rxn_type == "intermol_single_bond_change" or rxn_type == "coordination_bond_change":  # or (rxn_type == "concerted" and len(entries0) == 1 and len(entries1) == 2):
            entry = entries0[0]
            entry0 = entries1[0]
            entry1 = entries1[1]
            if entry0.parameters["ind"] <= entry1.parameters["ind"]:
                two_mol_name = str(entry0.parameters["ind"]) + "+" + str(entry1.parameters["ind"])
            else:
                two_mol_name = str(entry1.parameters["ind"]) + "+" + str(entry0.parameters["ind"])
            two_mol_name0 = str(entry0.parameters["ind"]) + "+PR_" + str(entry1.parameters["ind"])
            two_mol_name1 = str(entry1.parameters["ind"]) + "+PR_" + str(entry0.parameters["ind"])
            node_name_A = str(entry.parameters["ind"]) + "," + two_mol_name
            node_name_B0 = two_mol_name0 + "," + str(entry.parameters["ind"])
            node_name_B1 = two_mol_name1 + "," + str(entry.parameters["ind"])

            if rxn_type == "intermol_single_bond_change":
                rxn_type_A = "Molecular decomposition breaking one bond A -> B+C"
                rxn_type_B = "Molecular formation from one new bond A+B -> C"
            elif rxn_type == "coordination_bond_change":
                rxn_type_A = "Coordination bond breaking AM -> A+M"
                rxn_type_B = "Coordination bond forming A+M -> AM"
            elif rxn_type == "concerted":
                rxn_type_A = "Concerted"
                rxn_type_B = "Concerted"

            energy_A = entry0.energy + entry1.energy - entry.energy
            energy_B = entry.energy - entry0.energy - entry1.energy
            if entry1.free_energy != None and entry0.free_energy != None and entry.free_energy != None:
                free_energy_A = entry0.free_energy + entry1.free_energy - entry.free_energy
                free_energy_B = entry.free_energy - entry0.free_energy - entry1.free_energy

                if rxn_type == "concerted":
                    if entry.charge - (entry0.charge + entry1.charge) == 1:
                        free_energy_A += -self.electron_free_energy
                        free_energy_B += self.electron_free_energy
                    elif entry.charge - (entry0.charge + entry1.charge) == -1:
                        free_energy_A += self.electron_free_energy
                        free_energy_B += -self.electron_free_energy
                    elif entry.charge != (entry0.charge + entry1.charge):
                        raise RuntimeError("Concerted charge difference of "+str(abs(entry.charge - (entry0.charge + entry1.charge)))+" detected!")

            self.graph.add_node(node_name_A, rxn_type=rxn_type_A, bipartite=1, energy=energy_A, free_energy=free_energy_A)
            self.graph.add_edge(entry.parameters["ind"],
                                node_name_A,
                                softplus=self.softplus(free_energy_A),
                                exponent=self.exponent(free_energy_A),
                                weight=1.0
                                )
            self.graph.add_edge(node_name_A,
                                entry0.parameters["ind"],
                                softplus=0.0,
                                exponent=0.0,
                                weight=1.0
                                )
            self.graph.add_edge(node_name_A,
                                entry1.parameters["ind"],
                                softplus=0.0,
                                exponent=0.0,
                                weight=1.0
                                )
            self.graph.add_node(node_name_B0, rxn_type=rxn_type_B, bipartite=1, energy=energy_B, free_energy=free_energy_B)
            self.graph.add_node(node_name_B1, rxn_type=rxn_type_B, bipartite=1, energy=energy_B, free_energy=free_energy_B)
            self.graph.add_edge(node_name_B0,
                                entry.parameters["ind"],
                                softplus=0.0,
                                exponent=0.0,
                                weight=1.0
                                )
            self.graph.add_edge(node_name_B1,
                                entry.parameters["ind"],
                                softplus=0.0,
                                exponent=0.0,
                                weight=1.0
                                )
            self.graph.add_edge(entry0.parameters["ind"],
                                node_name_B0,
                                softplus=self.softplus(free_energy_B),
                                exponent=self.exponent(free_energy_B),
                                weight=1.0
                                )
            self.graph.add_edge(entry1.parameters["ind"],
                                node_name_B1,
                                softplus=self.softplus(free_energy_B),
                                exponent=self.exponent(free_energy_B),
                                weight=1.0)

        elif rxn_type == "concerted":
            reactant_total_charge = np.sum([item.charge for item in entries0])
            product_total_charge = np.sum([item.charge for item in entries1])
            total_charge_change = product_total_charge - reactant_total_charge
            if abs(total_charge_change) == 0:
                #raise RuntimeError("Concerted charge difference of " + str(abs(total_charge_change)) + " detected!")

                if len(entries0) == 1 and len(entries1) == 1:
                    # modified by XX
                    entryA = entries0[0]
                    A_name = str(entryA.parameters["ind"])

                    entryC = entries1[0]
                    C_name = str(entryC.parameters["ind"])

                    node_name_1 = A_name + "," + C_name
                    node_name_2 = C_name + "," + A_name
                    A_C_energy = entryC.energy - entryA.energy
                    C_A_energy = entryA.energy - entryC.energy
                    if entryA.free_energy != None and entryC.free_energy != None:
                        A_C_free_energy = entryC.free_energy - entryA.free_energy + total_charge_change * self.electron_free_energy
                        C_A_free_energy = entryA.free_energy - entryC.free_energy - total_charge_change * self.electron_free_energy

                    self.graph.add_node(node_name_1, rxn_type=rxn_type, bipartite=1, energy=A_C_energy,
                                        free_energy=A_C_free_energy)
                    self.graph.add_edge(entryA.parameters["ind"],
                                        node_name_1,
                                        softplus=self.softplus(A_C_free_energy),
                                        exponent=self.exponent(A_C_free_energy),
                                        weight=1.0
                                        )
                    self.graph.add_edge(node_name_1,
                                        entryC.parameters["ind"],
                                        softplus=0.0,
                                        exponent=0.0,
                                        weight=1.0
                                        )

                    self.graph.add_node(node_name_2, rxn_type=rxn_type, bipartite=1, energy=C_A_energy,
                                        free_energy=C_A_free_energy)
                    self.graph.add_edge(entryC.parameters["ind"],
                                        node_name_2,
                                        softplus=self.softplus(C_A_free_energy),
                                        exponent=self.exponent(C_A_free_energy),
                                        weight=1.0
                                        )
                    self.graph.add_edge(node_name_2,
                                        entryA.parameters["ind"],
                                        softplus=0.0,
                                        exponent=0.0,
                                        weight=1.0
                                        )

                elif len(entries0) == 2 and len(entries1) == 1:
                    #modified by XX
                    entryA = entries0[0]
                    entryB = entries0[1]
                    if entryA.parameters["ind"] <= entryB.parameters["ind"]:
                        AB_name = str(entryA.parameters["ind"]) + "+" + str(entryB.parameters["ind"])
                    else:
                        AB_name = str(entryB.parameters["ind"]) + "+" + str(entryA.parameters["ind"])
                    A_PR_B_name = str(entryA.parameters["ind"]) + "+PR_" + str(entryB.parameters["ind"])
                    B_PR_A_name = str(entryB.parameters["ind"]) + "+PR_" + str(entryA.parameters["ind"])

                    entryC = entries1[0]
                    C_name = str(entryC.parameters["ind"])

                    node_name_1 = A_PR_B_name + "," + C_name
                    node_name_2 = B_PR_A_name + "," + C_name
                    node_name_3 = C_name + "," + AB_name
                    AB_C_energy = entryC.energy - entryA.energy - entryB.energy
                    C_AB_energy = entryA.energy + entryB.energy - entryC.energy
                    if entryA.free_energy != None and entryB.free_energy != None and entryC.free_energy != None:
                        AB_C_free_energy = entryC.free_energy  - entryA.free_energy - entryB.free_energy + total_charge_change * self.electron_free_energy
                        C_AB_free_energy = entryA.free_energy + entryB.free_energy - entryC.free_energy - total_charge_change * self.electron_free_energy

                    self.graph.add_node(node_name_1, rxn_type=rxn_type, bipartite=1, energy=AB_C_energy,
                                        free_energy=AB_C_free_energy)
                    self.graph.add_edge(entryA.parameters["ind"],
                                        node_name_1,
                                        softplus=self.softplus(AB_C_free_energy),
                                        exponent=self.exponent(AB_C_free_energy),
                                        weight=1.0
                                        )
                    self.graph.add_edge(node_name_1,
                                        entryC.parameters["ind"],
                                        softplus=0.0,
                                        exponent=0.0,
                                        weight=1.0
                                        )

                    self.graph.add_node(node_name_2, rxn_type=rxn_type, bipartite=1, energy=AB_C_energy,
                                        free_energy=AB_C_free_energy)
                    self.graph.add_edge(entryB.parameters["ind"],
                                        node_name_2,
                                        softplus=self.softplus(AB_C_free_energy),
                                        exponent=self.exponent(AB_C_free_energy),
                                        weight=1.0
                                        )
                    self.graph.add_edge(node_name_2,
                                        entryC.parameters["ind"],
                                        softplus=0.0,
                                        exponent=0.0,
                                        weight=1.0
                                        )

                    self.graph.add_node(node_name_3, rxn_type=rxn_type, bipartite=1, energy=C_AB_energy,
                                        free_energy=C_AB_free_energy)
                    self.graph.add_edge(entryC.parameters["ind"],
                                        node_name_3,
                                        softplus=self.softplus(C_AB_free_energy),
                                        exponent=self.exponent(C_AB_free_energy),
                                        weight=1.0
                                        )
                    self.graph.add_edge(node_name_3,
                                        entryA.parameters["ind"],
                                        softplus=0.0,
                                        exponent=0.0,
                                        weight=1.0
                                        )
                    self.graph.add_edge(node_name_3,
                                        entryB.parameters["ind"],
                                        softplus=0.0,
                                        exponent=0.0,
                                        weight=1.0
                                        )

                elif len(entries0) == 1 and len(entries1) == 2:
                    entryA = entries0[0]
                    A_name = str(entryA.parameters["ind"])

                    entryC = entries1[0]
                    entryD = entries1[1]
                    if entryC.parameters["ind"] <= entryD.parameters["ind"]:
                        CD_name = str(entryC.parameters["ind"]) + "+" + str(entryD.parameters["ind"])
                    else:
                        CD_name = str(entryD.parameters["ind"]) + "+" + str(entryC.parameters["ind"])

                    C_PR_D_name = str(entryC.parameters["ind"]) + "+PR_" + str(entryD.parameters["ind"])
                    D_PR_C_name = str(entryD.parameters["ind"]) + "+PR_" + str(entryC.parameters["ind"])
                    node_name_1 = A_name + "," + CD_name
                    node_name_3 = C_PR_D_name + "," + A_name
                    node_name_4 = D_PR_C_name + "," + A_name
                    A_CD_energy = entryC.energy + entryD.energy - entryA.energy
                    CD_A_energy = entryA.energy  - entryC.energy - entryD.energy
                    if entryA.free_energy != None and entryC.free_energy != None and entryD.free_energy != None:
                        A_CD_free_energy = entryC.free_energy + entryD.free_energy - entryA.free_energy + total_charge_change * self.electron_free_energy
                        CD_A_free_energy = entryA.free_energy - entryC.free_energy - entryD.free_energy - total_charge_change * self.electron_free_energy

                    self.graph.add_node(node_name_1, rxn_type=rxn_type, bipartite=1, energy=A_CD_energy,
                                        free_energy=A_CD_free_energy)
                    self.graph.add_edge(entryA.parameters["ind"],
                                        node_name_1,
                                        softplus=self.softplus(A_CD_free_energy),
                                        exponent=self.exponent(A_CD_free_energy),
                                        weight=1.0
                                        )
                    self.graph.add_edge(node_name_1,
                                        entryC.parameters["ind"],
                                        softplus=0.0,
                                        exponent=0.0,
                                        weight=1.0
                                        )
                    self.graph.add_edge(node_name_1,
                                        entryD.parameters["ind"],
                                        softplus=0.0,
                                        exponent=0.0,
                                        weight=1.0
                                        )

                    self.graph.add_node(node_name_3, rxn_type=rxn_type, bipartite=1, energy=CD_A_energy,
                                        free_energy=CD_A_free_energy)
                    self.graph.add_edge(entryC.parameters["ind"],
                                        node_name_3,
                                        softplus=self.softplus(CD_A_free_energy),
                                        exponent=self.exponent(CD_A_free_energy),
                                        weight=1.0
                                        )
                    self.graph.add_edge(node_name_3,
                                        entryA.parameters["ind"],
                                        softplus=0.0,
                                        exponent=0.0,
                                        weight=1.0
                                        )

                    self.graph.add_node(node_name_4, rxn_type=rxn_type, bipartite=1, energy=CD_A_energy,
                                        free_energy=CD_A_free_energy)
                    self.graph.add_edge(entryD.parameters["ind"],
                                        node_name_4,
                                        softplus=self.softplus(CD_A_free_energy),
                                        exponent=self.exponent(CD_A_free_energy),
                                        weight=1.0
                                        )
                    self.graph.add_edge(node_name_4,
                                        entryA.parameters["ind"],
                                        softplus=0.0,
                                        exponent=0.0,
                                        weight=1.0
                                        )

                elif len(entries0) == 2 and len(entries1) == 2:
                    entryA = entries0[0]
                    entryB = entries0[1]
                    if entryA.parameters["ind"] <= entryB.parameters["ind"]:
                        AB_name = str(entryA.parameters["ind"])+"+"+str(entryB.parameters["ind"])
                    else:
                        AB_name = str(entryB.parameters["ind"])+"+"+str(entryA.parameters["ind"])
                    A_PR_B_name = str(entryA.parameters["ind"])+"+PR_"+str(entryB.parameters["ind"])
                    B_PR_A_name = str(entryB.parameters["ind"])+"+PR_"+str(entryA.parameters["ind"])

                    entryC = entries1[0]
                    entryD = entries1[1]
                    if entryC.parameters["ind"] <= entryD.parameters["ind"]:
                        CD_name = str(entryC.parameters["ind"])+"+"+str(entryD.parameters["ind"])
                    else:
                        CD_name = str(entryD.parameters["ind"])+"+"+str(entryC.parameters["ind"])

                    C_PR_D_name = str(entryC.parameters["ind"])+"+PR_"+str(entryD.parameters["ind"])
                    D_PR_C_name = str(entryD.parameters["ind"])+"+PR_"+str(entryC.parameters["ind"])
                    node_name_1 = A_PR_B_name+","+CD_name
                    node_name_2 = B_PR_A_name+","+CD_name
                    node_name_3 = C_PR_D_name+","+AB_name
                    node_name_4 = D_PR_C_name+","+AB_name
                    AB_CD_energy = entryC.energy + entryD.energy - entryA.energy - entryB.energy
                    CD_AB_energy = entryA.energy + entryB.energy - entryC.energy - entryD.energy
                    if entryA.free_energy != None and entryB.free_energy != None and entryC.free_energy != None and entryD.free_energy != None:
                        AB_CD_free_energy = entryC.free_energy + entryD.free_energy - entryA.free_energy - entryB.free_energy + total_charge_change * self.electron_free_energy
                        CD_AB_free_energy = entryA.free_energy + entryB.free_energy - entryC.free_energy - entryD.free_energy - total_charge_change * self.electron_free_energy

                    self.graph.add_node(node_name_1,rxn_type=rxn_type,bipartite=1,energy=AB_CD_energy,free_energy=AB_CD_free_energy)
                    self.graph.add_edge(entryA.parameters["ind"],
                                        node_name_1,
                                        softplus=self.softplus(AB_CD_free_energy),
                                        exponent=self.exponent(AB_CD_free_energy),
                                        weight=1.0
                                        )
                    self.graph.add_edge(node_name_1,
                                        entryC.parameters["ind"],
                                        softplus=0.0,
                                        exponent=0.0,
                                        weight=1.0
                                        )
                    self.graph.add_edge(node_name_1,
                                        entryD.parameters["ind"],
                                        softplus=0.0,
                                        exponent=0.0,
                                        weight=1.0
                                        )

                    self.graph.add_node(node_name_2,rxn_type=rxn_type,bipartite=1,energy=AB_CD_energy,free_energy=AB_CD_free_energy)
                    self.graph.add_edge(entryB.parameters["ind"],
                                        node_name_2,
                                        softplus=self.softplus(AB_CD_free_energy),
                                        exponent=self.exponent(AB_CD_free_energy),
                                        weight=1.0
                                        )
                    self.graph.add_edge(node_name_2,
                                        entryC.parameters["ind"],
                                        softplus=0.0,
                                        exponent=0.0,
                                        weight=1.0
                                        )
                    self.graph.add_edge(node_name_2,
                                        entryD.parameters["ind"],
                                        softplus=0.0,
                                        exponent=0.0,
                                        weight=1.0
                                        )

                    self.graph.add_node(node_name_3,rxn_type=rxn_type,bipartite=1,energy=CD_AB_energy,free_energy=CD_AB_free_energy)
                    self.graph.add_edge(entryC.parameters["ind"],
                                        node_name_3,
                                        softplus=self.softplus(CD_AB_free_energy),
                                        exponent=self.exponent(CD_AB_free_energy),
                                        weight=1.0
                                        )
                    self.graph.add_edge(node_name_3,
                                        entryA.parameters["ind"],
                                        softplus=0.0,
                                        exponent=0.0,
                                        weight=1.0
                                        )
                    self.graph.add_edge(node_name_3,
                                        entryB.parameters["ind"],
                                        softplus=0.0,
                                        exponent=0.0,
                                        weight=1.0
                                        )

                    self.graph.add_node(node_name_4,rxn_type=rxn_type,bipartite=1,energy=CD_AB_energy,free_energy=CD_AB_free_energy)
                    self.graph.add_edge(entryD.parameters["ind"],
                                        node_name_4,
                                        softplus=self.softplus(CD_AB_free_energy),
                                        exponent=self.exponent(CD_AB_free_energy),
                                        weight=1.0
                                        )
                    self.graph.add_edge(node_name_4,
                                        entryA.parameters["ind"],
                                        softplus=0.0,
                                        exponent=0.0,
                                        weight=1.0
                                        )
                    self.graph.add_edge(node_name_4,
                                        entryB.parameters["ind"],
                                        softplus=0.0,
                                        exponent=0.0,
                                        weight=1.0
                                        )
                else:
                    print("Concerted "+str(len(entries0))+","+str(len(entries1))+" found! Ignoring for now...")
            else:
                print("Concerted charge difference of " + str(abs(total_charge_change)) + " detected! Ignoring for now...")

    def softplus(self,free_energy):
        return np.log(1 + (273.0 / 500.0) * np.exp(free_energy))

    def exponent(self,free_energy):
        return np.exp(free_energy)

    def build_PR_record(self):
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

    def characterize_path(self,path,weight,PR_paths={},final=False):
        path_dict = {}
        path_dict["byproducts"] = []
        path_dict["unsolved_prereqs"] = []
        path_dict["solved_prereqs"] = []
        path_dict["all_prereqs"] = []
        path_dict["cost"] = 0.0
        path_dict["path"] = path

        for ii,step in enumerate(path):
            if ii != len(path)-1:
                path_dict["cost"] += self.graph[step][path[ii+1]][weight]
                if ii%2 == 1:
                    rxn = step.split(",")
                    if "+PR_" in rxn[0]:
                        PR = int(rxn[0].split("+PR_")[1])
                        path_dict["all_prereqs"].append(PR)
                    if "+" in rxn[1]:
                        desired_prod_satisfied = False
                        prods = rxn[1].split("+")
                        # if prods[0] == prods[1]:
                        #     path_dict["byproducts"].append(int(prods[0]))
                        # else:
                        #     for prod in prods:
                        #         if int(prod) != path[ii+1]:
                        #             path_dict["byproducts"].append(int(prod))
                        for prod in prods:
                            if int(prod) != path[ii+1]:
                                path_dict["byproducts"].append(int(prod))
                            elif desired_prod_satisfied:
                                path_dict["byproducts"].append(int(prod))
                            else:
                                desired_prod_satisfied = True
        for PR in path_dict["all_prereqs"]:
            if PR in path_dict["byproducts"]:
                # Note that we're ignoring the order in which BPs are made vs they come up as PRs...
                path_dict["all_prereqs"].remove(PR)
                path_dict["byproducts"].remove(PR)
                if PR in self.min_cost:
                    path_dict["cost"] -= self.min_cost[PR]
                else:
                    print("Missing PR cost to remove:",PR)
        for PR in path_dict["all_prereqs"]:
            # if len(PR_paths[PR].keys()) == self.num_starts:
            if PR in PR_paths:
                path_dict["solved_prereqs"].append(PR)
            else:
                path_dict["unsolved_prereqs"].append(PR)

        if final:
            path_dict["overall_free_energy_change"] = 0.0
            path_dict["hardest_step"] = None
            path_dict["description"] = ""
            path_dict["pure_cost"] = 0.0

            assert(len(path_dict["solved_prereqs"])==len(path_dict["all_prereqs"]))
            assert(len(path_dict["unsolved_prereqs"])==0)
            del path_dict["solved_prereqs"]
            del path_dict["unsolved_prereqs"]

            PRs_to_join = copy.deepcopy(path_dict["all_prereqs"])
            full_path = copy.deepcopy(path)
            while len(PRs_to_join) > 0:
                new_PRs = []
                for PR in PRs_to_join:
                    PR_path = None
                    PR_min_cost = 1000000000000000.0
                    for start in PR_paths[PR]:
                        if PR_paths[PR][start] != "no_path":
                            if PR_paths[PR][start]["cost"] < PR_min_cost:
                                PR_min_cost = PR_paths[PR][start]["cost"]
                                PR_path = PR_paths[PR][start]
                    assert(len(PR_path["solved_prereqs"])==len(PR_path["all_prereqs"]))
                    for new_PR in PR_path["all_prereqs"]:
                        new_PRs.append(new_PR)
                        path_dict["all_prereqs"].append(new_PR)
                    for new_BP in PR_path["byproducts"]:
                        path_dict["byproducts"].append(new_BP)
                    full_path = PR_path["path"] + full_path
                PRs_to_join = copy.deepcopy(new_PRs)

            for PR in path_dict["all_prereqs"]:
                if PR in path_dict["byproducts"]:
                    print("WARNING: Matching prereq and byproduct found!",PR)

            for ii,step in enumerate(full_path):
                if self.graph.nodes[step]["bipartite"] == 1:
                    if weight == "softplus":
                        path_dict["pure_cost"] += self.softplus(self.graph.nodes[step]["free_energy"])
                    elif weight == "exponent":
                        path_dict["pure_cost"] += self.exponent(self.graph.nodes[step]["free_energy"])
                    path_dict["overall_free_energy_change"] += self.graph.nodes[step]["free_energy"]
                    if path_dict["description"] == "":
                        path_dict["description"] += self.graph.nodes[step]["rxn_type"]
                    else:
                        path_dict["description"] += ", " + self.graph.nodes[step]["rxn_type"]
                    if path_dict["hardest_step"] == None:
                        path_dict["hardest_step"] = step
                    elif self.graph.nodes[step]["free_energy"] > self.graph.nodes[path_dict["hardest_step"]]["free_energy"]:
                        path_dict["hardest_step"] = step
            del path_dict["path"]
            path_dict["full_path"] = full_path
            if path_dict["hardest_step"] == None:
                path_dict["hardest_step_deltaG"] = None
            else:
                path_dict["hardest_step_deltaG"] = self.graph.nodes[path_dict["hardest_step"]]["free_energy"]
        return path_dict

    def solve_prerequisites(self,starts,target,weight,max_iter=100):
        PRs = {}
        self.old_solved_PRs = []
        new_solved_PRs = ["placeholder"]
        orig_graph = copy.deepcopy(self.graph)
        old_attrs = {}
        new_attrs = {}

        for start in starts:
            PRs[start] = {}
        for PR in PRs:
            for start in starts:
                if start == PR:
                    PRs[PR][start] = self.characterize_path([start],weight)
                else:
                    PRs[PR][start] = "no_path"
            self.old_solved_PRs.append(PR)
            self.min_cost[PR] = PRs[PR][PR]["cost"]
        for node in self.graph.nodes():
            if self.graph.nodes[node]["bipartite"] == 0 and node != target:
                if node not in PRs:
                    PRs[node] = {}

        ii = 0
        while (len(new_solved_PRs) > 0 or old_attrs != new_attrs) and ii < max_iter:
            min_cost = {}
            cost_from_start = {}
            for PR in PRs:
                cost_from_start[PR] = {}
                min_cost[PR] = 10000000000000000.0
                for start in PRs[PR]:
                    if PRs[PR][start] == "no_path":
                        cost_from_start[PR][start] = "no_path"
                    else:
                        cost_from_start[PR][start] = PRs[PR][start]["cost"]
                        if PRs[PR][start]["cost"] < min_cost[PR]:
                            min_cost[PR] = PRs[PR][start]["cost"]
                for start in starts:
                    if start not in cost_from_start[PR]:
                        cost_from_start[PR][start] = "unsolved"
            for node in self.graph.nodes():
                if self.graph.nodes[node]["bipartite"] == 0 and node not in self.old_solved_PRs and node != target:
                    for start in starts:
                        if start not in PRs[node]:
                            path_exists = True
                            try:
                                length,dij_path = nx.algorithms.simple_paths._bidirectional_dijkstra(
                                    self.graph,
                                    source=hash(start),
                                    target=hash(node),
                                    ignore_nodes=self.find_or_remove_bad_nodes([target,node]),
                                    weight=weight)
                            except nx.exception.NetworkXNoPath:
                                PRs[node][start] = "no_path"
                                path_exists = False
                                cost_from_start[node][start] = "no_path"
                            if path_exists:
                                if len(dij_path) > 1 and len(dij_path)%2 == 1:
                                    path = self.characterize_path(dij_path,weight,self.old_solved_PRs)
                                    # if node == 8:
                                    #     print('node:',node)
                                    #     print(path)
                                    cost_from_start[node][start] = path["cost"]
                                    if len(path["unsolved_prereqs"]) == 0:
                                        PRs[node][start] = path
                                        # print("Solved PR",node,PRs[node])
                                    if path["cost"] < min_cost[node]:
                                        min_cost[node] = path["cost"]
                                else:
                                    print("Does this ever happen?")

            solved_PRs = copy.deepcopy(self.old_solved_PRs)
            new_solved_PRs = []
            self.unsolved_PRs = []
            for PR in PRs:
                if PR not in solved_PRs:
                    if len(PRs[PR].keys()) == self.num_starts:
                        solved_PRs.append(PR)
                        new_solved_PRs.append(PR)
                    else:
                        best_start_so_far = [None,10000000000000000.0]
                        for start in PRs[PR]:
                            if PRs[PR][start] != "no_path":
                                if PRs[PR][start] == "unsolved":
                                    print("ERROR: unsolved should never be encountered here!")
                                if PRs[PR][start]["cost"] < best_start_so_far[1]:
                                    best_start_so_far[0] = start
                                    best_start_so_far[1] = PRs[PR][start]["cost"]
                        if best_start_so_far[0] != None:
                            num_beaten = 0
                            for start in cost_from_start[PR]:
                                if start != best_start_so_far[0]:
                                    if cost_from_start[PR][start] == "no_path":
                                        num_beaten += 1
                                    elif cost_from_start[PR][start] >= best_start_so_far[1]:
                                        num_beaten += 1
                            if num_beaten == self.num_starts - 1:
                                solved_PRs.append(PR)
                                new_solved_PRs.append(PR)
                            else:
                                self.unsolved_PRs.append(PR)

            # new_solved_PRs = []
            # for PR in solved_PRs:
            #     if PR not in old_solved_PRs:
            #         new_solved_PRs.append(PR)

            print(ii,len(self.old_solved_PRs),len(new_solved_PRs))
            attrs = {}

            for PR_ind in min_cost:
                for rxn_node in self.PR_record[PR_ind]:
                    non_PR_reactant_node = int(rxn_node.split(",")[0].split("+PR_")[0])
                    PR_node = int(rxn_node.split(",")[0].split("+PR_")[1])
                    assert(int(PR_node)==PR_ind)
                    attrs[(non_PR_reactant_node,rxn_node)] = {weight:orig_graph[non_PR_reactant_node][rxn_node][weight]+min_cost[PR_ind]}
                    # prod_nodes = []
                    # if "+" in split_node[1]:
                    #     tmp = split_node[1].split("+")
                    #     for prod_ind in tmp:
                    #         prod_nodes.append(int(prod_ind))
                    # else:
                    #     prod_nodes.append(int(split_node[1]))
                    # for prod_node in prod_nodes:
                    #     attrs[(node,prod_node)] = {weight:orig_graph[node][prod_node][weight]+min_cost[PR_ind]}
            nx.set_edge_attributes(self.graph,attrs)
            self.min_cost = copy.deepcopy(min_cost)
            self.old_solved_PRs = copy.deepcopy(solved_PRs)
            ii += 1
            old_attrs = copy.deepcopy(new_attrs)
            new_attrs = copy.deepcopy(attrs)

        # for PR in PRs:
        #     path_found = False
        #     if PRs[PR] != {}:
        #         for start in PRs[PR]:
        #             if PRs[PR][start] != "no_path":
        #                 path_found = True
        #                 path_dict = self.characterize_path(PRs[PR][start]["path"],weight,PRs,True)
        #                 if abs(path_dict["cost"]-path_dict["pure_cost"])>0.0001:
        #                     print("WARNING: cost mismatch for PR",PR,path_dict["cost"],path_dict["pure_cost"],path_dict["full_path"])
        #         if not path_found:
        #             print("No path found from any start to PR",PR)
        #     else:
        #         print("Unsolvable path from any start to PR",PR)
        #print(self.min_cost)
        # print(len(self.min_cost.keys()))
        # for i in range(75):
        #     if i not in self.min_cost.keys():
        #         print('not solved:', i)
            #print(self.min_cost[i])
        return PRs

    def solve_prerequisites_wo_target(self,starts,weight,max_iter=100, save=False, name='default'):
        PRs = {}
        self.old_solved_PRs = []
        new_solved_PRs = ["placeholder"]
        orig_graph = copy.deepcopy(self.graph)
        old_attrs = {}
        new_attrs = {}

        for start in starts:
            PRs[start] = {}
        for PR in PRs:
            for start in starts:
                if start == PR:
                    PRs[PR][start] = self.characterize_path([start],weight)
                else:
                    PRs[PR][start] = "no_path"
            self.old_solved_PRs.append(PR)
            self.min_cost[PR] = PRs[PR][PR]["cost"]
        for node in self.graph.nodes():
            if self.graph.nodes[node]["bipartite"] == 0:
                if node not in PRs:
                    PRs[node] = {}

        ii = 0
        while (len(new_solved_PRs) > 0 or old_attrs != new_attrs) and ii < max_iter:
            min_cost = {}
            cost_from_start = {}
            for PR in PRs:
                cost_from_start[PR] = {}
                min_cost[PR] = 10000000000000000.0
                for start in PRs[PR]:
                    if PRs[PR][start] == "no_path":
                        cost_from_start[PR][start] = "no_path"
                    else:
                        cost_from_start[PR][start] = PRs[PR][start]["cost"]
                        if PRs[PR][start]["cost"] < min_cost[PR]:
                            min_cost[PR] = PRs[PR][start]["cost"]
                for start in starts:
                    if start not in cost_from_start[PR]:
                        cost_from_start[PR][start] = "unsolved"
            for node in self.graph.nodes():
                if self.graph.nodes[node]["bipartite"] == 0 and node not in self.old_solved_PRs:
                    for start in starts:
                        if start not in PRs[node]:
                            path_exists = True
                            try:
                                length,dij_path = nx.algorithms.simple_paths._bidirectional_dijkstra(
                                    self.graph,
                                    source=hash(start),
                                    target=hash(node),
                                    weight=weight)
                            except nx.exception.NetworkXNoPath:
                                PRs[node][start] = "no_path"
                                path_exists = False
                                cost_from_start[node][start] = "no_path"
                            if path_exists:
                                if len(dij_path) > 1 and len(dij_path)%2 == 1:
                                    path = self.characterize_path(dij_path,weight,self.old_solved_PRs)
                                    # if node == 8:
                                    #     print('node:',node)
                                    #     print(path)
                                    cost_from_start[node][start] = path["cost"]
                                    if len(path["unsolved_prereqs"]) == 0:
                                        PRs[node][start] = path
                                        # print("Solved PR",node,PRs[node])
                                    if path["cost"] < min_cost[node]:
                                        min_cost[node] = path["cost"]
                                else:
                                    print("Does this ever happen?")

            solved_PRs = copy.deepcopy(self.old_solved_PRs)
            new_solved_PRs = []
            self.unsolved_PRs = []
            for PR in PRs:
                if PR not in solved_PRs:
                    if len(PRs[PR].keys()) == self.num_starts:
                        solved_PRs.append(PR)
                        new_solved_PRs.append(PR)
                    else:
                        best_start_so_far = [None,10000000000000000.0]
                        for start in PRs[PR]:
                            if PRs[PR][start] != "no_path":
                                if PRs[PR][start] == "unsolved":
                                    print("ERROR: unsolved should never be encountered here!")
                                if PRs[PR][start]["cost"] < best_start_so_far[1]:
                                    best_start_so_far[0] = start
                                    best_start_so_far[1] = PRs[PR][start]["cost"]
                        if best_start_so_far[0] != None:
                            num_beaten = 0
                            for start in cost_from_start[PR]:
                                if start != best_start_so_far[0]:
                                    if cost_from_start[PR][start] == "no_path":
                                        num_beaten += 1
                                    elif cost_from_start[PR][start] >= best_start_so_far[1]:
                                        num_beaten += 1
                            if num_beaten == self.num_starts - 1:
                                solved_PRs.append(PR)
                                new_solved_PRs.append(PR)
                            else:
                                self.unsolved_PRs.append(PR)

            # new_solved_PRs = []
            # for PR in solved_PRs:
            #     if PR not in old_solved_PRs:
            #         new_solved_PRs.append(PR)

            print(ii,len(self.old_solved_PRs),len(new_solved_PRs))
            attrs = {}

            for PR_ind in min_cost:
                for rxn_node in self.PR_record[PR_ind]:
                    non_PR_reactant_node = int(rxn_node.split(",")[0].split("+PR_")[0])
                    PR_node = int(rxn_node.split(",")[0].split("+PR_")[1])
                    assert(int(PR_node)==PR_ind)
                    attrs[(non_PR_reactant_node,rxn_node)] = {weight:orig_graph[non_PR_reactant_node][rxn_node][weight]+min_cost[PR_ind]}
                    # prod_nodes = []
                    # if "+" in split_node[1]:
                    #     tmp = split_node[1].split("+")
                    #     for prod_ind in tmp:
                    #         prod_nodes.append(int(prod_ind))
                    # else:
                    #     prod_nodes.append(int(split_node[1]))
                    # for prod_node in prod_nodes:
                    #     attrs[(node,prod_node)] = {weight:orig_graph[node][prod_node][weight]+min_cost[PR_ind]}
            nx.set_edge_attributes(self.graph,attrs)
            self.min_cost = copy.deepcopy(min_cost)
            self.old_solved_PRs = copy.deepcopy(solved_PRs)
            ii += 1
            old_attrs = copy.deepcopy(new_attrs)
            new_attrs = copy.deepcopy(attrs)

        # for PR in PRs:
        #     path_found = False
        #     if PRs[PR] != {}:
        #         for start in PRs[PR]:
        #             if PRs[PR][start] != "no_path":
        #                 path_found = True
        #                 path_dict = self.characterize_path(PRs[PR][start]["path"],weight,PRs,True)
        #                 if abs(path_dict["cost"]-path_dict["pure_cost"])>0.0001:
        #                     print("WARNING: cost mismatch for PR",PR,path_dict["cost"],path_dict["pure_cost"],path_dict["full_path"])
        #         if not path_found:
        #             print("No path found from any start to PR",PR)
        #     else:
        #         print("Unsolvable path from any start to PR",PR)
        #print(self.min_cost)
        # print(len(self.min_cost.keys()))
        # for i in range(75):
        #     if i not in self.min_cost.keys():
        #         print('not solved:', i)
            #print(self.min_cost[i])
        if save:
            dumpfn(PRs, name+'_PR_paths.json')
            dumpfn(self.min_cost, name+'_min_cost.json')
        return PRs


    def find_or_remove_bad_nodes(self,nodes,remove_nodes=False):
        bad_nodes = []
        for node in nodes:
            for bad_node in self.PR_record[node]:
                bad_nodes.append(bad_node)
        if remove_nodes:
            pruned_graph = copy.deepcopy(self.graph)
            pruned_graph.remove_nodes_from(bad_nodes)
            return pruned_graph
        else:
            return bad_nodes

    def valid_shortest_simple_paths(self,start,target,weight,PRs=[]):
        bad_nodes = PRs
        bad_nodes.append(target)
        valid_graph = self.find_or_remove_bad_nodes(bad_nodes,remove_nodes=True)
        return nx.shortest_simple_paths(valid_graph,hash(start),hash(target),weight=weight)

    def find_paths(self,starts,target,weight,num_paths=10):
        """
        Args:
            starts ([int]): List of starting node IDs (ints). 
            target (int): Target node ID.
            weight (str): String identifying what edge weight to use for path finding.
            num_paths (int): Number of paths to find. Defaults to 10.
        """
        paths = []
        c = itertools.count()
        my_heapq = []

        print("Solving prerequisites...")
        self.num_starts = len(starts)
        PR_paths = self.solve_prerequisites(starts,target,weight)

        print("Finding paths...")
        for start in starts:
            ind = 0
            for path in self.valid_shortest_simple_paths(start,target,weight):
                if ind == num_paths:
                    break
                else:
                    ind += 1
                    path_dict = self.characterize_path(path,weight,PR_paths,final=True)
                    heapq.heappush(my_heapq, (path_dict["cost"],next(c),path_dict))

        while len(paths) < num_paths and my_heapq:
            # Check if any byproduct could yield a prereq cheaper than from starting molecule(s)?
            (cost, _, path_dict) = heapq.heappop(my_heapq)
            print(len(paths),cost,len(my_heapq),path_dict["all_prereqs"])
            paths.append(path_dict)

        return PR_paths, paths

    def identify_sinks(self):
        sinks = []
        for node in self.graph.nodes():
            self.graph.nodes[node]["local_sink"] = 0
        for node in self.graph.nodes():
            if self.graph.nodes[node]["bipartite"] == 0:
                neighbor_list = list(self.graph.neighbors(node))
                if len(neighbor_list) > 0:
                    neg_found = False
                    for neighbor in neighbor_list:
                        if self.graph.nodes[neighbor]["free_energy"] < 0:
                            neg_found = True
                            break
                    if not neg_found:
                        self.graph.nodes[node]["local_sink"] = 1
                        sinks.append(node)
                        for neighbor in neighbor_list:
                            self.graph.nodes[neighbor]["local_sink"] = 1
                            second_neighbor_list = list(self.graph.neighbors(neighbor))
                            for second_neighbor in second_neighbor_list:
                                self.graph.nodes[second_neighbor]["local_sink"] = 2

        return sinks

