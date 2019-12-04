import numpy as np
from pymatgen.analysis.graphs import MoleculeGraph, MolGraphSplitError
from pymatgen.analysis.local_env import OpenBabelNN
from pymatgen.io.babel import BabelMolAdaptor
from pymatgen import Molecule
from pymatgen.analysis.fragmenter import metal_edge_extender
from monty.serialization import dumpfn, loadfn
import os
import matplotlib.pyplot as plt
from monty.json import MSONable
import copy
import networkx as nx
import itertools
import heapq
from ase.units import eV, J, mol
import random


class MoleculeEntry(MSONable):
    def __init__(self, molecule, energy, correction=0.0, enthalpy=None, entropy=None,
                 parameters=None, entry_id=None, attribute=None):
        self.molecule = molecule
        self.uncorrected_energy = energy
        self.enthalpy = enthalpy
        self.entropy = entropy
        self.composition = molecule.composition
        self.correction = correction
        self.parameters = parameters if parameters else {}
        self.entry_id = entry_id
        self.attribute = attribute
        self.mol_graph = MoleculeGraph.with_local_env_strategy(self.molecule,
                                                          OpenBabelNN(),
                                                          reorder=False,
                                                          extend_structure=False)
        self.mol_graph = metal_edge_extender(self.mol_graph)
    @property
    def graph(self):
        return self.mol_graph.graph
    @property
    def edges(self):
        return self.graph.edges()
    @property
    def energy(self):
        """
        Returns the *corrected* energy of the entry.
        """
        return self.uncorrected_energy + self.correction
    @property
    def free_energy(self, temp=298.0):
        if self.enthalpy != None and self.entropy != None:
            return self.energy*27.21139+0.0433641*self.enthalpy-temp*self.entropy*0.0000433641
        else:
            return None
    @property
    def formula(self):
        return self.composition.alphabetical_formula
    @property
    def charge(self):
        return self.molecule.charge
    @property
    def Nbonds(self):
        return len(self.edges)
    def __repr__(self):
        output = ["MoleculeEntry {} - {} - E{} - C{}".format(self.entry_id,
                                                      self.formula,
                                                      self.Nbonds,
                                                      self.charge),
                  "Energy = {:.4f} Hartree".format(self.uncorrected_energy),
                  "Correction = {:.4f} Hartree".format(self.correction),
                  "Enthalpy = {:.4f} kcal/mol".format(self.enthalpy),
                  "Entropy = {:.4f} cal/mol.K".format(self.entropy),
                  "Free Energy = {:.4f} eV".format(self.free_energy),
                  "Parameters:"]
        for k, v in self.parameters.items():
            output.append("{} = {}".format(k, v))
        return "\n".join(output)
    def __str__(self):
        return self.__repr__()


class ReactionNetwork(MSONable):
    def __init__(self, input_entries, electron_free_energy=-2.15):
        self.input_entries = input_entries
        self.electron_free_energy = electron_free_energy
        self.entries = {}
        self.entries_list = []
        print(len(self.input_entries), "input entries")
        connected_entries = []
        for entry in self.input_entries:
            # print(len(entry.molecule))
            if len(entry.molecule) > 1:
                if nx.is_weakly_connected(entry.graph):
                    connected_entries.append(entry)
            else:
                connected_entries.append(entry)
        print(len(connected_entries), "connected entries")
        get_formula = lambda x: x.formula
        get_Nbonds = lambda x: x.Nbonds
        get_charge = lambda x: x.charge
        sorted_entries_0 = sorted(connected_entries, key=get_formula)
        for k1, g1 in itertools.groupby(sorted_entries_0, get_formula):
            sorted_entries_1 = sorted(list(g1), key=get_Nbonds)
            self.entries[k1] = {}
            for k2, g2 in itertools.groupby(sorted_entries_1, get_Nbonds):
                sorted_entries_2 = sorted(list(g2), key=get_charge)
                self.entries[k1][k2] = {}
                for k3, g3 in itertools.groupby(sorted_entries_2, get_charge):
                    sorted_entries_3 = list(g3)
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
            entry.parameters["ind"] = ii
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(range(len(self.entries_list)), bipartite=0)
        self.num_reactions = 0
        self.one_electron_redox()
        self.intramol_single_bond_change()
        self.intermol_single_bond_change()
        self.coordination_bond_change()
        # self.add_water_reactions()
        # self.concerted_2_steps()
        self.add_LEDC_concerted_reactions()
        self.add_water_lithium_reaction()
        self.PR_record = self.build_PR_record()
        self.min_cost = {}
        self.num_starts = None
    def one_electron_redox(self):
        for formula in self.entries:
            for Nbonds in self.entries[formula]:
                charges = list(self.entries[formula][Nbonds].keys())
                if len(charges) > 1:
                    for ii in range(len(charges) - 1):
                        charge0 = charges[ii]
                        charge1 = charges[ii + 1]
                        if charge1 - charge0 == 1:
                            for entry0 in self.entries[formula][Nbonds][charge0]:
                                for entry1 in self.entries[formula][Nbonds][charge1]:
                                    if entry0.mol_graph.isomorphic_to(entry1.mol_graph):
                                        self.add_reaction([entry0], [entry1], "one_electron_redox")
                                        break
    def intramol_single_bond_change(self):
        for formula in self.entries:
            Nbonds_list = list(self.entries[formula].keys())
            if len(Nbonds_list) > 1:
                for ii in range(len(Nbonds_list) - 1):
                    Nbonds0 = Nbonds_list[ii]
                    Nbonds1 = Nbonds_list[ii + 1]
                    if Nbonds1 - Nbonds0 == 1:
                        for charge in self.entries[formula][Nbonds0]:
                            if charge in self.entries[formula][Nbonds1]:
                                for entry1 in self.entries[formula][Nbonds1][charge]:
                                    for edge in entry1.edges:
                                        mg = copy.deepcopy(entry1.mol_graph)
                                        mg.break_edge(edge[0], edge[1], allow_reverse=True)
                                        if nx.is_weakly_connected(mg.graph):
                                            for entry0 in self.entries[formula][Nbonds0][charge]:
                                                if entry0.mol_graph.isomorphic_to(mg):
                                                    self.add_reaction([entry0], [entry1], "intramol_single_bond_change")
                                                    break
    def intermol_single_bond_change(self):
        for formula in self.entries:
            for Nbonds in self.entries[formula]:
                if Nbonds > 0:
                    for charge in self.entries[formula][Nbonds]:
                        for entry in self.entries[formula][Nbonds][charge]:
                            for edge in entry.edges:
                                bond = [(edge[0], edge[1])]
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
                                                                    self.add_reaction([entry], [entry0, entry1],
                                                                                      "intermol_single_bond_change")
                                                                    break
                                                        break
                                except MolGraphSplitError:
                                    pass
    def coordination_bond_change(self):
        M_entries = {}
        for formula in self.entries:
            if formula == "Li1" or formula == "Mg1":
                if formula not in M_entries:
                    M_entries[formula] = {}
                for charge in self.entries[formula][0]:
                    assert (len(self.entries[formula][0][charge]) == 1)
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
                                                        if nonM_formula in self.entries:
                                                            if nonM_Nbonds in self.entries[nonM_formula]:
                                                                for nonM_charge in self.entries[nonM_formula][
                                                                    nonM_Nbonds]:
                                                                    M_charge = entry.charge - nonM_charge
                                                                    if M_charge in M_entries[M_formula]:
                                                                        for nonM_entry in \
                                                                        self.entries[nonM_formula][nonM_Nbonds][
                                                                            nonM_charge]:
                                                                            if frag.isomorphic_to(nonM_entry.mol_graph):
                                                                                self.add_reaction([entry], [nonM_entry,
                                                                                                            M_entries[
                                                                                                                M_formula][
                                                                                                                M_charge]],
                                                                                                  "coordination_bond_change")
                                                                                break
                                        except MolGraphSplitError:
                                            pass
    def add_water_reactions(self):
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
                H2O_PR_H2O_name = str(H2O_entry.parameters["ind"]) + "+PR_" + str(H2O_entry.parameters["ind"])
                if H3Oplus_found:
                    print("Adding concerted water splitting rxn1: 2H2O -> OH- + H3O+")
                    if OHminus_entry.parameters["ind"] <= H3Oplus_entry.parameters["ind"]:
                        OHminus_H3Oplus_name = str(OHminus_entry.parameters["ind"]) + "+" + str(
                            H3Oplus_entry.parameters["ind"])
                    else:
                        OHminus_H3Oplus_name = str(H3Oplus_entry.parameters["ind"]) + "+" + str(
                            OHminus_entry.parameters["ind"])
                    rxn_node_1 = H2O_PR_H2O_name + "," + OHminus_H3Oplus_name
                    rxn1_energy = OHminus_entry.energy + H3Oplus_entry.energy - 2 * H2O_entry.energy
                    rxn1_free_energy = OHminus_entry.free_energy + H3Oplus_entry.free_energy - 2 * H2O_entry.free_energy
                    print("Rxn1 free energy =", rxn1_free_energy)
                    self.graph.add_node(rxn_node_1, rxn_type="water_dissociation", bipartite=1, energy=rxn1_energy,
                                        free_energy=rxn1_free_energy)
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
                    OHminus2_H2_name = str(OHminus_entry.parameters["ind"]) + "+" + str(
                        OHminus_entry.parameters["ind"]) + "+" + str(H2_entry.parameters["ind"])
                    rxn_node_2 = H2O_PR_H2O_name + "," + OHminus2_H2_name
                    rxn2_energy = 2 * OHminus_entry.energy + H2_entry.energy - 2 * H2O_entry.energy
                    rxn2_free_energy = 2 * OHminus_entry.free_energy + H2_entry.free_energy - 2 * H2O_entry.free_energy - 2 * self.electron_free_energy
                    print("Water rxn2 free energy =", rxn2_free_energy)
                    self.graph.add_node(rxn_node_2, rxn_type="water_2e_redox", bipartite=1, energy=rxn2_energy,
                                        free_energy=rxn2_free_energy)
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
                H2O_Li_name = str(H2O_entry.parameters["ind"]) + "+" + str(Li_plus_entry.parameters["ind"])
            else:
                H2O_Li_name = str(Li_plus_entry.parameters["ind"]) + "+" + str(H2O_entry.parameters["ind"])

            if LiOH_entry.parameters["ind"] <= H2_entry.parameters["ind"]:
                LiOH_H2_name = str(LiOH_entry.parameters["ind"]) + "+" + str(H2_entry.parameters["ind"])
            else:
                LiOH_H2_name = str(H2_entry.parameters["ind"]) + "+" + str(LiOH_entry.parameters["ind"])
            rxn_node = H2O_Li_name + "," + LiOH_H2_name
            rxn_energy = 2 * LiOH_entry.energy + H2_entry.energy - 2 * H2O_entry.energy - 2 * Li_plus_entry.energy
            rxn_free_energy = 2 * LiOH_entry.free_energy + H2_entry.free_energy - 2 * H2O_entry.free_energy - \
                              2 * Li_plus_entry.free_energy - 2 * self.electron_free_energy
            print("Rxn free energy =", rxn_free_energy)
            self.graph.add_node(rxn_node, rxn_type="water_lithium_reaction", bipartite=1, energy=rxn_energy,
                                free_energy=rxn_free_energy)
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
        LEDC_found = False
        C2H4_found = False
        LiEC_RO_found = False
        LiCO3_minus_found = False
        LiEC_plus_found = False
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
        if LiEC_RO_found and C2H4_found and LEDC_found:
            print("Adding concerted reaction 2LiEC-RO -> LEDC + C2H4")
            LiEC_RO_PR_LiEC_RO_name = str(LiEC_RO_entry.parameters["ind"]) + "+PR_" + str(
                LiEC_RO_entry.parameters["ind"])
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
                LiEC_plus_LiCO3_minus_name = str(LiEC_plus_entry.parameters["ind"]) + "+" + str(
                    LiCO3_minus_entry.parameters["ind"])
            else:
                LiEC_plus_LiCO3_minus_name = str(LiCO3_minus_entry.parameters["ind"]) + "+" + str(
                    LiEC_plus_entry.parameters["ind"])
            LEDC_name = str(LEDC_entry.parameters["ind"])
            rxn_node_2 = LiEC_plus_LiCO3_minus_name + "," + LEDC_name
            rxn2_energy = LEDC_entry.energy - LiEC_plus_entry.energy - LiCO3_minus_entry.energy
            rxn2_free_energy = LEDC_entry.free_energy - LiEC_plus_entry.free_energy - LiCO3_minus_entry.free_energy
            print("LiCO3 -1 + LiEC 1 -> LEDC free energy =", rxn2_free_energy)
            self.graph.add_node(rxn_node_2, rxn_type="LiCO3 -1 + LiEC 1 -> LEDC", bipartite=1, energy=rxn2_energy,
                                free_energy=rxn2_free_energy)
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
    def add_reaction(self, entries0, entries1, rxn_type):
        self.num_reactions += 1
        if rxn_type == "one_electron_redox":
            if len(entries0) != 1 or len(entries1) != 1:
                raise RuntimeError("One electron redox requires two lists that each contain one entry!")
        elif rxn_type == "intramol_single_bond_change":
            if len(entries0) != 1 or len(entries1) != 1:
                raise RuntimeError("Intramolecular single bond change requires two lists that each contain one entry!")
        elif rxn_type == "intermol_single_bond_change":
            if len(entries0) != 1 or len(entries1) != 2:
                raise RuntimeError(
                    "Intermolecular single bond change requires two lists that contain one entry and two entries, respectively!")
        elif rxn_type == "coordination_bond_change":
            if len(entries0) != 1 or len(entries1) != 2:
                raise RuntimeError(
                    "Coordination bond change requires two lists that contain one entry and two entries, respectively!")
        else:
            raise RuntimeError("Reaction type " + rxn_type + " is not supported!")
        if rxn_type == "one_electron_redox" or rxn_type == "intramol_single_bond_change":
            entry0 = entries0[0]
            entry1 = entries1[0]
            if rxn_type == "one_electron_redox":
                val0 = entry0.charge
                val1 = entry1.charge
                if val1 < val0:
                    rxn_type_A = "One electron reduction"
                    rxn_type_B = "One electron oxidation"
                else:
                    rxn_type_A = "One electron oxidation"
                    rxn_type_B = "One electron reduction"
            elif rxn_type == "intramol_single_bond_change":
                val0 = entry0.Nbonds
                val1 = entry1.Nbonds
                if val1 < val0:
                    rxn_type_A = "Intramolecular single bond breakage"
                    rxn_type_B = "Intramolecular single bond formation"
                else:
                    rxn_type_A = "Intramolecular single bond formation"
                    rxn_type_B = "Intramolecular single bond breakage"
            node_name_A = str(entry0.parameters["ind"]) + "," + str(entry1.parameters["ind"])
            node_name_B = str(entry1.parameters["ind"]) + "," + str(entry0.parameters["ind"])
            energy_A = entry1.energy - entry0.energy
            energy_B = entry0.energy - entry1.energy
            if entry1.free_energy != None and entry0.free_energy != None:
                free_energy_A = entry1.free_energy - entry0.free_energy
                free_energy_B = entry0.free_energy - entry1.free_energy
                if rxn_type == "one_electron_redox":
                    if rxn_type_A == "One electron reduction":
                        free_energy_A += -self.electron_free_energy
                        free_energy_B += self.electron_free_energy
                    else:
                        free_energy_A += self.electron_free_energy
                        free_energy_B += -self.electron_free_energy
            else:
                free_energy_A = None
                free_energy_B = None
            self.graph.add_node(node_name_A, rxn_type=rxn_type_A, bipartite=1, energy=energy_A,
                                free_energy=free_energy_A)
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
            self.graph.add_node(node_name_B, rxn_type=rxn_type_B, bipartite=1, energy=energy_B,
                                free_energy=free_energy_B)
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
            energy_A = entry0.energy + entry1.energy - entry.energy
            energy_B = entry.energy - entry0.energy - entry1.energy
            if entry1.free_energy != None and entry0.free_energy != None and entry.free_energy != None:
                free_energy_A = entry0.free_energy + entry1.free_energy - entry.free_energy
                free_energy_B = entry.free_energy - entry0.free_energy - entry1.free_energy
            self.graph.add_node(node_name_A, rxn_type=rxn_type_A, bipartite=1, energy=energy_A,
                                free_energy=free_energy_A)

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
            self.graph.add_node(node_name_B0, rxn_type=rxn_type_B, bipartite=1, energy=energy_B,
                                free_energy=free_energy_B)
            self.graph.add_node(node_name_B1, rxn_type=rxn_type_B, bipartite=1, energy=energy_B,
                                free_energy=free_energy_B)
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
    def softplus(self, free_energy):
        return np.log(1 + (273.0 / 500.0) * np.exp(free_energy))
    def exponent(self, free_energy):
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
    def characterize_path(self, path, weight, PR_paths={}, final=False):
        path_dict = {}
        path_dict["byproducts"] = []
        path_dict["unsolved_prereqs"] = []
        path_dict["solved_prereqs"] = []
        path_dict["all_prereqs"] = []
        path_dict["cost"] = 0.0
        path_dict["path"] = path
        for ii, step in enumerate(path):
            if ii != len(path) - 1:
                path_dict["cost"] += self.graph[step][path[ii + 1]][weight]
                if ii % 2 == 1:
                    rxn = step.split(",")
                    if "+PR_" in rxn[0]:
                        PR = int(rxn[0].split("+PR_")[1])
                        path_dict["all_prereqs"].append(PR)
                    if "+" in rxn[1]:
                        desired_prod_satisfied = False
                        prods = rxn[1].split("+")
                        for prod in prods:
                            if int(prod) != path[ii + 1]:
                                path_dict["byproducts"].append(int(prod))
                            elif desired_prod_satisfied:
                                path_dict["byproducts"].append(int(prod))
                            else:
                                desired_prod_satisfied = True
        for PR in path_dict["all_prereqs"]:
            if PR in path_dict["byproducts"]:
                path_dict["all_prereqs"].remove(PR)
                path_dict["byproducts"].remove(PR)
                if PR in self.min_cost:
                    path_dict["cost"] -= self.min_cost[PR]
                else:
                    print("Missing PR cost to remove:", PR)
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
            assert (len(path_dict["solved_prereqs"]) == len(path_dict["all_prereqs"]))
            assert (len(path_dict["unsolved_prereqs"]) == 0)
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
                    assert (len(PR_path["solved_prereqs"]) == len(PR_path["all_prereqs"]))
                    for new_PR in PR_path["all_prereqs"]:
                        new_PRs.append(new_PR)
                        path_dict["all_prereqs"].append(new_PR)
                    for new_BP in PR_path["byproducts"]:
                        path_dict["byproducts"].append(new_BP)
                    full_path = PR_path["path"] + full_path
                PRs_to_join = copy.deepcopy(new_PRs)
            for PR in path_dict["all_prereqs"]:
                if PR in path_dict["byproducts"]:
                    print("WARNING: Matching prereq and byproduct found!", PR)
            for ii, step in enumerate(full_path):
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
                    elif self.graph.nodes[step]["free_energy"] > self.graph.nodes[path_dict["hardest_step"]][
                        "free_energy"]:
                        path_dict["hardest_step"] = step
            del path_dict["path"]
            path_dict["full_path"] = full_path
            if path_dict["hardest_step"] == None:
                path_dict["hardest_step_deltaG"] = None
            else:
                path_dict["hardest_step_deltaG"] = self.graph.nodes[path_dict["hardest_step"]]["free_energy"]
        return path_dict
    def solve_prerequisites(self, starts, target, weight, max_iter=20):
        PRs = {}
        old_solved_PRs = []
        new_solved_PRs = ["placeholder"]
        orig_graph = copy.deepcopy(self.graph)
        old_attrs = {}
        new_attrs = {}
        for start in starts:
            PRs[start] = {}
        for PR in PRs:
            for start in starts:
                if start == PR:
                    PRs[PR][start] = self.characterize_path([start], weight)
                else:
                    PRs[PR][start] = "no_path"
            old_solved_PRs.append(PR)
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
                if self.graph.nodes[node]["bipartite"] == 0 and node not in old_solved_PRs and node != target:
                    for start in starts:
                        if start not in PRs[node]:
                            path_exists = True
                            try:
                                length, dij_path = nx.algorithms.simple_paths._bidirectional_dijkstra(
                                    self.graph,
                                    source=hash(start),
                                    target=hash(node),
                                    ignore_nodes=self.find_or_remove_bad_nodes([target, node]),
                                    weight=weight)
                            except nx.exception.NetworkXNoPath:
                                PRs[node][start] = "no_path"
                                path_exists = False
                                cost_from_start[node][start] = "no_path"
                            if path_exists:
                                if len(dij_path) > 1 and len(dij_path) % 2 == 1:
                                    path = self.characterize_path(dij_path, weight, old_solved_PRs)
                                    cost_from_start[node][start] = path["cost"]
                                    if len(path["unsolved_prereqs"]) == 0:
                                        PRs[node][start] = path
                                        # print("Solved PR",node,PRs[node])
                                    if path["cost"] < min_cost[node]:
                                        min_cost[node] = path["cost"]
                                else:
                                    print("Does this ever happen?")

            solved_PRs = copy.deepcopy(old_solved_PRs)
            new_solved_PRs = []
            for PR in PRs:
                if PR not in solved_PRs:
                    if len(PRs[PR].keys()) == self.num_starts:
                        solved_PRs.append(PR)
                        new_solved_PRs.append(PR)
                    else:
                        best_start_so_far = [None, 10000000000000000.0]
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
                                    elif cost_from_start[PR][start] > best_start_so_far[1]:
                                        num_beaten += 1
                            if num_beaten == self.num_starts - 1:
                                solved_PRs.append(PR)
                                new_solved_PRs.append(PR)

            # new_solved_PRs = []
            # for PR in solved_PRs:
            #     if PR not in old_solved_PRs:
            #         new_solved_PRs.append(PR)

            print(ii, len(old_solved_PRs), len(new_solved_PRs))
            attrs = {}

            for PR_ind in min_cost:
                for rxn_node in self.PR_record[PR_ind]:
                    non_PR_reactant_node = int(rxn_node.split(",")[0].split("+PR_")[0])
                    PR_node = int(rxn_node.split(",")[0].split("+PR_")[1])
                    assert (int(PR_node) == PR_ind)
                    attrs[(non_PR_reactant_node, rxn_node)] = {
                        weight: orig_graph[non_PR_reactant_node][rxn_node][weight] + min_cost[PR_ind]}
                    # prod_nodes = []
                    # if "+" in split_node[1]:
                    #     tmp = split_node[1].split("+")
                    #     for prod_ind in tmp:
                    #         prod_nodes.append(int(prod_ind))
                    # else:
                    #     prod_nodes.append(int(split_node[1]))
                    # for prod_node in prod_nodes:
                    #     attrs[(node,prod_node)] = {weight:orig_graph[node][prod_node][weight]+min_cost[PR_ind]}
            nx.set_edge_attributes(self.graph, attrs)
            self.min_cost = copy.deepcopy(min_cost)
            old_solved_PRs = copy.deepcopy(solved_PRs)
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

        return PRs

    def find_or_remove_bad_nodes(self, nodes, remove_nodes=False):
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

    def valid_shortest_simple_paths(self, start, target, weight, PRs=[]):
        bad_nodes = PRs
        bad_nodes.append(target)
        valid_graph = self.find_or_remove_bad_nodes(bad_nodes, remove_nodes=True)
        return nx.shortest_simple_paths(valid_graph, hash(start), hash(target), weight=weight)

    def find_paths(self, starts, target, weight, num_paths=10):
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
        PR_paths = self.solve_prerequisites(starts, target, weight)

        print("Finding paths...")
        for start in starts:
            ind = 0
            for path in self.valid_shortest_simple_paths(start, target, weight):
                if ind == num_paths:
                    break
                else:
                    ind += 1
                    path_dict = self.characterize_path(path, weight, PR_paths, final=True)
                    heapq.heappush(my_heapq, (path_dict["cost"], next(c), path_dict))

        while len(paths) < num_paths and my_heapq:
            # Check if any byproduct could yield a prereq cheaper than from starting molecule(s)?
            (cost, _, path_dict) = heapq.heappop(my_heapq)
            print(len(paths), cost, len(my_heapq), path_dict["all_prereqs"])
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

k_b = 1.38064852e-23
T = 298.15
h = 6.62607015e-34
R = 8.3144598

class StochaticSimulation:
    """
        Class for stochastic simulation from ReactionNetwork class

        Args:
            input_entries ([MoleculeEntry]): A list of MoleculeEntry objects.
            electron_free_energy (float): The Gibbs free energy of an electron.
                Defaults to -2.15 eV, the value at which the LiEC SEI forms.
        """
    def __init__(self, reaction_network):
        self.reaction_network = reaction_network
        self.entries_list = self.reaction_network.entries_list
        self.num_species = len(self.entries_list) + 1
        self.num_reactions = self.reaction_network.num_reactions
        self.graph = self.reaction_network.graph
        self.reactions = []
        # This is unique reactions. i.e. "233+PR_5914,2130"  == "5914+PR_2233,2130"
        self.unique_reaction_nodes = []
        for node0 in self.reaction_network.graph.nodes():
             if self.graph.node[node0]["bipartite"] == 1:
                 reactants = node0.split(",")[0].split("+")
                 reactants = [reac.replace("PR_", "") for reac in reactants]
                 products = node0.split(",")[1].split("+")
                 reactants.sort()
                 products.sort()
                 if [reactants, products] not in self.reactions:
                    self.reactions.append([reactants, products])
                    self.unique_reaction_nodes.append(node0)
        # create a key "charge_change" indicating the total charge change in the reaction : total charge of products - total charge of reactants
        for node in self.unique_reaction_nodes:
            if self.graph.node[node]["rxn_type"] == "One electron oxidation":
                self.graph.node[node]["charge_change"] = 1
            elif self.graph.node[node]["rxn_type"] == "One electron reduction":
                self.graph.node[node]["charge_change"] = -1
            elif self.graph.node[node]["rxn_type"] == "water_lithium_reaction":
                self.graph.node[node]["charge_change"] = -2
            elif self.graph.node[node]["rxn_type"] == "water_2e_redox":
                self.graph.node[node]["charge_change"] = -2
            else:
                self.graph.node[node]["charge_change"] = 0
        return

    def get_rates(self, barrier_uni, barrier_bi):
        '''
        Approximate rates for all the reactions in reaction_nodes. All exergonic unimolecular reactions have the same rate;
        all exergonic bmolecular reactions have the same rate.
        Rates of endergonic reactions are computed from delta G and the corresponding reverse exergonic reaction rate.
        :param reaction_energy:
        :param barrier_uni: Barrier in eV for unimolecular exergonic reactions.
        :param barrier_bi: Barrier in eV for bimolecular exergonic reactions.
        :return:
        '''
        self.reaction_rates = []
        for rxn_node in self.unique_reaction_nodes:
            if self.graph.nodes[rxn_node]['rxn_type'] == 'LiCO3 -1 + LiEC 1 -> LEDC':
                barrier = eV/J*mol * barrier_uni
                rate = k_b * T / h * np.exp(-barrier / R / T)
            elif self.graph.nodes[rxn_node]['rxn_type'] == '2LiEC-RO -> LEDC + C2H4':
                barrier = eV/J*mol * barrier_uni
                rate = k_b * T / h * np.exp(-barrier / R / T)
            elif self.graph.nodes[rxn_node]['rxn_type'] == 'water_lithium_reaction':
                barrier = eV/J*mol * barrier_uni
                rate = k_b * T / h * np.exp(-barrier / R / T)
            else:
                reaction_energy = self.graph.node[rxn_node]["free_energy"]
                reactants = rxn_node.split(",")[0].split("+")
                num_reactants = len(reactants)
                if reaction_energy <= 0:
                    if num_reactants == 1:
                        barrier = eV/J*mol * barrier_uni
                        rate = k_b * T / h * np.exp(-barrier / R / T)
                    elif num_reactants == 2:
                        barrier =  eV/J*mol * barrier_bi
                        rate = k_b * T / h * np.exp(-barrier / R / T)
                elif reaction_energy > 0:
                    if num_reactants == 1:
                        barrier = eV/J*mol * (barrier_uni + reaction_energy)
                        rate = k_b * T / h * np.exp(-barrier / R / T)
                    elif num_reactants == 2:
                        barrier = eV/J*mol * (barrier_bi + reaction_energy)
                        rate = k_b * T / h * np.exp(-barrier / R / T)
                self.reaction_rates.append(rate)
        return

    def get_propensities(self, num_of_mols):
        '''
        :param num_of_mols:  Number of mols array for all the species
        :param rates: Rates array. size: num of reactions
        :return: An array of size num of reactions. Each entry correspond to the propersity of the current reaction, given the rate and number of reactants.
        '''
        self.propensities = []
        for i, rxn_node in enumerate(self.unique_reaction_nodes):
            propensity = self.reaction_rates[i]
            reactants = rxn_node.split(",")[0].split("+")
            reactants = [reac.replace("PR_","") for reac in reactants]
            unique_reactants = list(set(reactants))
            for reactant in unique_reactants:
                num_of_reactants = reactants.count(reactant)
                if num_of_reactants == 1:
                    propensity *= num_of_mols[int(reactant)]
                elif num_of_reactants == 2:
                    propensity *= 0.5 * num_of_mols[int(reactant)] * (num_of_mols[int(reactant)] - 1)
                elif num_of_reactants == 3:
                    propensity *= 1/6 * num_of_mols[int(reactant)] * (num_of_mols[int(reactant)] - 1) \
                                  * (num_of_mols[int(reactant)] - 2)
            if self.graph.node[rxn_node]["charge_change"] == -1:
                propensity *= num_of_mols[-1]
            elif self.graph.node[rxn_node]["charge_change"] == -2:
                propensity *= 0.5 * num_of_mols[-1] * (num_of_mols[-1] - 1)
            self.propensities.append(propensity)
        return self.propensities

    def remove_gas_reactions(self,xyz_dir):
        # find the indices of gases
        test_dir = xyz_dir
        C2H4_mg = MoleculeGraph.with_local_env_strategy(
            Molecule.from_file(os.path.join(test_dir, "ethylene.xyz")),
            OpenBabelNN(),
            reorder=False,
            extend_structure=False)
        for entry in RN.entries['C2 H4'][5][0]:
            if C2H4_mg.isomorphic_to(entry.mol_graph):
                C2H4_ind = entry.parameters["ind"]
                break

        CO_mg = MoleculeGraph.with_local_env_strategy(
            Molecule.from_file(os.path.join(test_dir, "CO.xyz")),
            OpenBabelNN(),
            reorder=False,
            extend_structure=False)
        for entry in RN.entries['C1 O1'][1][0]:
            if CO_mg.isomorphic_to(entry.mol_graph):
                CO_ind = entry.parameters["ind"]
                break

        CO2_mg = MoleculeGraph.with_local_env_strategy(
            Molecule.from_file(os.path.join(test_dir, "CO2.xyz")),
            OpenBabelNN(),
            reorder=False,
            extend_structure=False)
        for entry in RN.entries['C1 O2'][2][0]:
            if CO2_mg.isomorphic_to(entry.mol_graph):
                CO2_ind = entry.parameters["ind"]
                break

        H2_mg = MoleculeGraph.with_local_env_strategy(
            Molecule.from_file(os.path.join(test_dir, "H2.xyz")),
            OpenBabelNN(),
            reorder=False,
            extend_structure=False)
        for entry in RN.entries['H2'][1][0]:
            if H2_mg.isomorphic_to(entry.mol_graph):
                H2_ind = entry.parameters["ind"]
                break

        PF5_mg = MoleculeGraph.with_local_env_strategy(
            Molecule.from_file(os.path.join(test_dir, "PF5.xyz")),
            OpenBabelNN(),
            reorder=False,
            extend_structure=False)
        for entry in RN.entries['F5 P1'][5][0]:
            if PF5_mg.isomorphic_to(entry.mol_graph):
                PF5_ind = entry.parameters["ind"]
                break
        gas_indices = [C2H4_ind, CO_ind, CO2_ind, H2_ind, PF5_ind]
        for i,rxn_node in enumerate(self.unique_reaction_nodes):
            reactants = rxn_node.split(",")[0].split("+")
            reactants = [reac.replace("PR_", "") for reac in reactants]
            if any(reac in gas_indices for reac in reactants):
                self.reaction_rates[i] = 0.0
        return

    def add_concerted_reactions_2_step(self, num_of_mols, num_thresh):
        '''
        Add concerted reactions on the fly, only for species that have non-zero concentration.
        This function only adds two-step concerted reactions.
        :param num_of_mols:
        :return:
        '''

        non_zero_indices = [i for i in range(len(num_of_mols)) if num_of_mols[i] > num_thresh]
        for mol_id in non_zero_indices:
            neighbor_rxns = list(self.graph.neighbors(mol_id))
            for rxn0 in neighbor_rxns:
                if self.graph.node[rxn0]["free_energy"] > 0:
                    rxn0_reactants = rxn0.split(",")
                    rxn0_reactants = [reac.replace("PR_","") for reac in rxn0_reactants]
                    rxn0_products = list(self.graph.neighbors(rxn0))
                    for node1 in rxn0_products:
                        #rxn0_products_copy = copy.deepcopy(rxn0_products)
                        rxn0_products.remove(str(node1))
                        middle_products = rxn0_products
                        node1_rxns = list(self.graph.neighbors(node1))
                        for rxn1 in node1_rxns:
                            if self.graph.node[rxn0]["free_energy"] + self.graph.node[rxn1]["free_energy"] < 0:
                                rxn1_reactants = rxn1.split(",")
                                rxn1_reactants = [reac.replace("PR_", "") for reac in rxn1_reactants]
                                #rxn1_reactants_copy = copy.deepcopy(rxn1_reactants)
                                rxn1_reactants.remove(str(node1))
                                middle_reactants = rxn1_reactants
                                rxn1_products = list(self.graph.neighbors(rxn1))
                                total_reactants = rxn0_reactants + middle_reactants
                                total_products = middle_products + rxn1_products
                                total_reactants.sort()
                                total_products.sort()

                                total_species = total_reactants + total_products
                                total_species_set = list(set(total_species))
                                # remove species that appear both in reactants and products
                                for species in total_species_set:
                                    while (species in total_reactants and species in total_products):
                                        total_reactants.remove(species)
                                        total_products.remove(species)
                                # check the reaction is not in the existing reactions, and both the number of reactants and products less than 2
                                if ([total_reactants, total_products] not in self.reactions) and \
                                        (len(total_reactants) <= 2) and (len(total_products) <= 2):
                                    unique_elements = []
                                    for species in total_species_set:
                                        unique_elements += list(self.entries_list[int(species)].molecule.atomic_numbers)
                                    unique_elements = list(set(unique_elements))
                                    reactant_elements, product_elements = [], []
                                    for species in total_reactants:
                                        reactant_elements += list(self.entries_list[int(species)].molecule.atomic_numbers)
                                    for species in total_products:
                                        product_elements += list(self.entries_list[int(species)].molecule.atomic_numbers)
                                    # check stoichiometry
                                    if all(reactant_elements.count(ele) == product_elements.count(ele) for ele in unique_elements):
                                        # check total charge
                                        reactant_total_charge = np.sum([self.entries_list[int(item)].charge for item in total_reactants])
                                        product_total_charge = np.sum([self.entries_list[int(item)].charge for item in total_products])
                                        if abs(reactant_total_charge - product_total_charge) >= 3:
                                            print("WARNING: total charges differ by more than 3! Ignoring...")
                                        else:
                                            reactant_name = "+".join(total_reactants)
                                            product_name = "+".join(total_products)
                                            reaction_forward_name = reactant_name + "," + product_name
                                            reaction_reverse_name = product_name + "," + reactant_name
                                            total_energy_reactant = np.sum([self.entries_list[int(item)].energy for item in total_reactants])
                                            total_energy_product = np.sum([self.entries_list[int(item)].energy for item in total_products])
                                            total_free_energy_reactant = np.sum([self.entries_list[int(item)].free_energy for item in total_reactants])
                                            total_free_energy_product = np.sum([self.entries_list[int(item)].free_energy for item in total_products])
                                            energy_forward = total_energy_product - total_energy_reactant
                                            energy_reverse = - energy_forward
                                            total_charge_change = product_total_charge - reactant_total_charge
                                            free_energy_forward = total_free_energy_product - total_free_energy_reactant + \
                                                                  total_charge_change * self.reaction_network.electron_free_energy
                                            free_energy_reverse = - free_energy_forward
                                            self.graph.add_node(reaction_forward_name, rxn_type="concerted_two_step",
                                                      bipartite=1, energy=energy_forward, free_energy=free_energy_forward, charge_change=total_charge_change)
                                            self.graph.add_node(reaction_reverse_name, rxn_type="concerted_two_step",bipartite=1, energy=energy_reverse,
                                                                free_energy=free_energy_reverse,charge_change=-total_charge_change)
                                            self.unique_reaction_nodes.append(reaction_forward_name)
                                            self.unique_reaction_nodes.append(reaction_reverse_name)
                                            self.reactions.append([total_reactants, total_products])


    def direct_method(self, initial_conc, time_span, max_output_length):
        '''
        :param initial_conc
        :param time_span: int, number of time steps
        :param max_output_length
        Gillespie, D.T. (1977) Exact Stochastic Simulation of Coupled Chemical Reactions. J Phys Chem, 81:25, 2340-2361.
        :return: t: time vector (Nreaction_events x 1); x:  species amounts (Nreaction_events x Nspecies);
                 rxns: rxns that have been fired up (Nreaction_events x 1).
        '''
        records = {}
        records['a'] = []
        records['random_rxn'] = []
        records['a0'] = []
        records['mu'] = []
        t = np.zeros(max_output_length)
        x = np.zeros([max_output_length, self.num_species])
        rxns = np.zeros(max_output_length)
        t[0] = 0
        x[0,:] = initial_conc
        rxn_count = 0

        while t[rxn_count] < time_span:
            a = self.get_propensities(x[rxn_count,:])
            a0 = np.sum(a)
            random_t = random.uniform(0, 1)
            random_rxn = random.uniform(0, 1)
            tau = -np.log(random_t) / a0
            mu = np.where(np.cumsum(a) >= random_rxn * a0)[0][0]
            records['a'].append(a)
            records['random_rxn'].append(random_rxn)
            records['a0'].append(a0)
            records['mu'].append(mu)

            if rxn_count + 1 > max_output_length:
                t = t[:rxn_count]
                x = x[:rxn_count]
                print("WARNING:Number of reaction events exceeded the number pre-allocated. Simulation terminated prematurely.")

            t[rxn_count + 1] = t[rxn_count] + tau
            x[rxn_count + 1] = x[rxn_count]
            current_reaction = self.unique_reaction_nodes[mu]
            reactants = current_reaction.split(",")[0].split("+")
            reactants = [reac.replace("PR_", "") for reac in reactants]
            products = current_reaction.split(",")[1].split("+")
            for reac in reactants:
                x[rxn_count+1, int(reac)] -= 1
            for prod in products:
                x[rxn_count+1, int(prod)] += 1
            if self.graph.node[current_reaction]["charge_change"] != 0:
                x[rxn_count + 1, -1] += self.graph.node[current_reaction]["charge_change"]
            rxns[rxn_count+1] = mu
            rxn_count += 1

        t = t[:rxn_count]
        x = x[:rxn_count,:]
        rxns = rxns[:rxn_count]
        if t[-1] > time_span:
            t[-1] = time_span
            x[-1,:] = x[rxn_count-1,:]
            rxns[-1] = rxns[rxn_count-1]
        return t, x, rxns, records

    def add_concerted_reactions_on_the_fly(self, initial_conc, time_span, max_output_length, barrier_uni, barrier_bi, xyz_dir,iterations=5):
        iter = 0
        t, x, rxns, records = 0,0,0,0
        print("current nums of reactions:", len(self.unique_reaction_nodes))
        while iter < iterations:
            t, x, rxns, records = self.direct_method(initial_conc, time_span, max_output_length)
            self.add_concerted_reactions_2_step(x[-1,:], 10)
            print("current nums of reactions:", len(self.unique_reaction_nodes))
            self.get_rates(barrier_uni, barrier_bi)
            self.remove_gas_reactions(xyz_dir)
        return t,x,rxns,records


prod_entries = []
entries = loadfn("/home/xiaowei/pymatgen/pymatgen/analysis/reaction_network/smd_production_entries.json")
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

RN = ReactionNetwork(
    prod_entries,
    electron_free_energy=-2.15)

SS = StochaticSimulation(RN)

test_dir = '/home/xiaowei/Sam_production/xyzs/'
EC_mg = MoleculeGraph.with_local_env_strategy(
    Molecule.from_file(os.path.join(test_dir, "EC.xyz")),
    OpenBabelNN(),
    reorder=False,
    extend_structure=False)
EC_mg = metal_edge_extender(EC_mg)

LiEC_mg = MoleculeGraph.with_local_env_strategy(
    Molecule.from_file(os.path.join(test_dir, "LiEC.xyz")),
    OpenBabelNN(),
    reorder=False,
    extend_structure=False)
LiEC_mg = metal_edge_extender(LiEC_mg)

LEDC_mg = MoleculeGraph.with_local_env_strategy(
    Molecule.from_file(os.path.join(test_dir, "LEDC.xyz")),
    OpenBabelNN(),
    reorder=False,
    extend_structure=False)
LEDC_mg = metal_edge_extender(LEDC_mg)

LEMC_mg = MoleculeGraph.with_local_env_strategy(
    Molecule.from_file(os.path.join(test_dir, "LEMC.xyz")),
    OpenBabelNN(),
    reorder=False,
    extend_structure=False)
LEMC_mg = metal_edge_extender(LEMC_mg)

H2O_mg = MoleculeGraph.with_local_env_strategy(
    Molecule.from_file(os.path.join(test_dir, "water.xyz")),
    OpenBabelNN(),
    reorder=False,
    extend_structure=False)
H2O_mg = metal_edge_extender(H2O_mg)
EC_ind = None
LEMC_ind = None
H2O_ind = None
for entry in RN.entries["C3 H4 O3"][10][0]:
    if EC_mg.isomorphic_to(entry.mol_graph):
        EC_ind = entry.parameters["ind"]
        break

for entry in RN.entries["H2 O1"][2][0]:
    if H2O_mg.isomorphic_to(entry.mol_graph):
        H2O_ind = entry.parameters["ind"]
        break

Li1_ind = RN.entries["Li1"][0][1][0].parameters["ind"]
OHminus_ind = RN.entries["H1 O1"][1][-1][0].parameters["ind"]

initial_conc = np.zeros(SS.num_species)
initial_conc[EC_ind] = 15000
initial_conc[Li1_ind] = 1000
initial_conc[H2O_ind] = 30
initial_conc[-1] = 1000
SS.get_rates(1.0841025975148306, 1.3009231170177968)
SS.remove_gas_reactions('/home/xiaowei/Sam_production/xyzs')
t, x, rxns, records = SS.direct_method(initial_conc, 1000000, 100000000)

sorted_species_index = np.argsort(x[-1, :])[::-1]
fig, ax = plt.subplots()
for i in range(100):
    species_index = sorted_species_index[i]
    if x[-1, int(species_index)] > 0 and int(species_index) != EC_ind and int(species_index) != Li1_ind and int(
            species_index) != SS.num_species - 1:
        ax.step(t, x[:, int(species_index)], where='mid', label=str(species_index))
        # ax.plot(T,X[:,int(species_index)], 'C0o', alpha=0.5)
plt.title('KMC')
plt.legend(loc='upper left')
plt.savefig('metal_edge_extender_special_concerted.png')

rxns_set = list(set(rxns))
rxns_count = [list(rxns).count(rxn) for rxn in rxns_set]
index = np.argsort(rxns_count)[::-1]
sorted_rxns = np.array(rxns_set)[index]
x0 = np.arange(len(rxns_set))
fig, ax = plt.subplots()
plt.bar(x0, rxns_count)
# plt.xticks(x, ([str(int(rxn)) for rxn in rxns_set]))
plt.title('reaction decomposition')
plt.savefig('reaction_decomposition_metal_edge_extender_special_concerted.png')
for rxn in sorted_rxns:
    rxn = int(rxn)
    print(SS.unique_reaction_nodes[rxn], SS.reaction_rates[rxn])