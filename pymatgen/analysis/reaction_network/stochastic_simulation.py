# coding: utf-8
# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.

import logging
import numpy as np
from pymatgen.analysis.graphs import MoleculeGraph, MolGraphSplitError
from pymatgen.analysis.local_env import OpenBabelNN
from pymatgen.io.babel import BabelMolAdaptor
from pymatgen import Molecule
from pymatgen.analysis.fragmenter import metal_edge_extender
from pymatgen.entries.mol_entry import MoleculeEntry
from pymatgen.analysis.reaction_network.reaction_network import ReactionNetwork
from monty.serialization import dumpfn, loadfn
import random
import os
import matplotlib.pyplot as plt
from ase.units import eV, J, mol
import copy
import pickle


__author__ = "Xiaowei Xie"
__copyright__ = "Copyright 2019, The Materials Project"
__version__ = "1.0"
__maintainer__ = "Xiaowei Xie"
__email__ = "xxw940407@icloud.com"
__status__ = "Alpha"
__date__ = "11/18/19"

logger = logging.getLogger(__name__)

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
        self.num_species = len(self.reaction_network.entries_list) + 1
        # This is the species indices from which concerted reactions have been searched
        self.searched_species_2_step = []
        self.searched_species_3_step = []
        self.searched_species_4_step = []
        self.num_reactions = self.reaction_network.num_reactions
        self.reactions = []
        # This is unique reactions. i.e. "233+PR_5914,2130"  == "5914+PR_2233,2130"
        self.unique_reaction_nodes = []
        for node0 in self.reaction_network.graph.nodes():
             if self.reaction_network.graph.node[node0]["bipartite"] == 1:
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
            if self.reaction_network.graph.node[node]["rxn_type"] == "One electron oxidation":
                self.reaction_network.graph.node[node]["charge_change"] = 1
            elif self.reaction_network.graph.node[node]["rxn_type"] == "One electron reduction":
                self.reaction_network.graph.node[node]["charge_change"] = -1
            elif self.reaction_network.graph.node[node]["rxn_type"] == "water_lithium_reaction":
                self.reaction_network.graph.node[node]["charge_change"] = -2
            elif self.reaction_network.graph.node[node]["rxn_type"] == "water_2e_redox":
                self.reaction_network.graph.node[node]["charge_change"] = -2
            else:
                self.reaction_network.graph.node[node]["charge_change"] = 0
            self.reaction_network.graph.node[node]["steps"] = 1
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
            # if self.reaction_network.graph.nodes[rxn_node]['rxn_type'] == 'LiCO3 -1 + LiEC 1 -> LEDC':
            #     barrier = eV/J*mol * barrier_uni
            #     rate = k_b * T / h * np.exp(-barrier / R / T)
            # elif self.reaction_network.graph.nodes[rxn_node]['rxn_type'] == '2LiEC-RO -> LEDC + C2H4':
            #     barrier = eV/J*mol * barrier_uni
            #     rate = k_b * T / h * np.exp(-barrier / R / T)
            #else:
            reaction_energy = self.reaction_network.graph.node[rxn_node]["free_energy"]
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
            if self.reaction_network.graph.node[rxn_node]["charge_change"] == -1:
                propensity *= num_of_mols[-1]
            elif self.reaction_network.graph.node[rxn_node]["charge_change"] == -2:
                propensity *= 0.5 * num_of_mols[-1] * (num_of_mols[-1] - 1)
            self.propensities.append(propensity)
        return self.propensities

    def remove_gas_reactions(self,xyz_dir):
        # find the indices of gases
        C2H4_mg = MoleculeGraph.with_local_env_strategy(
            Molecule.from_file(os.path.join(xyz_dir, "ethylene.xyz")),
            OpenBabelNN(),
            reorder=False,
            extend_structure=False)
        for entry in self.reaction_network.entries['C2 H4'][5][0]:
            if C2H4_mg.isomorphic_to(entry.mol_graph):
                C2H4_ind = entry.parameters["ind"]
                break

        CO_mg = MoleculeGraph.with_local_env_strategy(
            Molecule.from_file(os.path.join(xyz_dir, "CO.xyz")),
            OpenBabelNN(),
            reorder=False,
            extend_structure=False)
        for entry in self.reaction_network.entries['C1 O1'][1][0]:
            if CO_mg.isomorphic_to(entry.mol_graph):
                CO_ind = entry.parameters["ind"]
                break

        CO2_mg = MoleculeGraph.with_local_env_strategy(
            Molecule.from_file(os.path.join(xyz_dir, "CO2.xyz")),
            OpenBabelNN(),
            reorder=False,
            extend_structure=False)
        for entry in self.reaction_network.entries['C1 O2'][2][0]:
            if CO2_mg.isomorphic_to(entry.mol_graph):
                CO2_ind = entry.parameters["ind"]
                break

        H2_mg = MoleculeGraph.with_local_env_strategy(
            Molecule.from_file(os.path.join(xyz_dir, "H2.xyz")),
            OpenBabelNN(),
            reorder=False,
            extend_structure=False)
        for entry in self.reaction_network.entries['H2'][1][0]:
            if H2_mg.isomorphic_to(entry.mol_graph):
                H2_ind = entry.parameters["ind"]
                break

        PF5_mg = MoleculeGraph.with_local_env_strategy(
            Molecule.from_file(os.path.join(xyz_dir, "PF5.xyz")),
            OpenBabelNN(),
            reorder=False,
            extend_structure=False)
        for entry in self.reaction_network.entries['F5 P1'][5][0]:
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
    
    def remove_Li_reduction_reaction(self, xyz_dir):
        Li_mg = MoleculeGraph.with_local_env_strategy(
            Molecule.from_file(os.path.join(xyz_dir, "Li.xyz")),
            OpenBabelNN(),
            reorder=False,
            extend_structure=False)
        for entry in self.reaction_network.entries['Li1'][0][1]:
            if Li_mg.isomorphic_to(entry.mol_graph):
                Li1_ind = entry.parameters["ind"]
                break
        for entry in self.reaction_network.entries['Li1'][0][0]:
            if Li_mg.isomorphic_to(entry.mol_graph):
                Li0_ind = entry.parameters["ind"]
                break
        for i,rxn_node in enumerate(self.unique_reaction_nodes):
            if rxn_node == '{},{}'.format(str(Li1_ind), str(Li0_ind)):
                self.reaction_rates[i] = 0.0
        return
    
    def add_reactions_to_graph(self,reactants,products,rxn_type,steps,add_to_graph=False, add_to_kmc=True):
        '''
        Only add reaction nodes, but do not connect them to reactants and products
        :param reactants: list of strings of mol id (sorted from small to large numbers)
        :param products: list of strings of mol id (sorted from small to large numbers)
        :param rxn_type: 
        :param steps: number of steps. For n step concerted, it's n; 1 for everything else.
        :param add_to_graph: Whether to connect the reaction node with reactants and products in self.reaction_network.graph. 
                      If False, only the reaction node will be added to the graph.
        :param add_to_kmc: Whether to add to self.unique_reaction_nodes for kmc simulation.
        :return: 
        '''
        reactant_total_charge = np.sum([self.reaction_network.entries_list[int(item)].charge for item in reactants])
        product_total_charge = np.sum([self.reaction_network.entries_list[int(item)].charge for item in products])
        reactant_name = "+".join(reactants)
        product_name = "+".join(products)
        reaction_forward_name = reactant_name + "," + product_name
        reaction_reverse_name = product_name + "," + reactant_name
        total_energy_reactant = np.sum(
            [self.reaction_network.entries_list[int(item)].energy for item in reactants])
        total_energy_product = np.sum([self.reaction_network.entries_list[int(item)].energy for item in products])
        total_free_energy_reactant = np.sum(
            [self.reaction_network.entries_list[int(item)].free_energy for item in reactants])
        total_free_energy_product = np.sum(
            [self.reaction_network.entries_list[int(item)].free_energy for item in products])
        energy_forward = total_energy_product - total_energy_reactant
        energy_reverse = - energy_forward
        total_charge_change = product_total_charge - reactant_total_charge
        free_energy_forward = total_free_energy_product - total_free_energy_reactant + \
                              total_charge_change * self.reaction_network.electron_free_energy
        free_energy_reverse = - free_energy_forward

        self.reaction_network.graph.add_node(reaction_forward_name, rxn_type=rxn_type,
                                             bipartite=1, energy=energy_forward, free_energy=free_energy_forward,
                                             charge_change=total_charge_change, steps=steps)
        self.reaction_network.graph.add_node(reaction_reverse_name, rxn_type=rxn_type, bipartite=1,
                                             energy=energy_reverse,free_energy=free_energy_reverse, 
                                             charge_change=-total_charge_change, steps=steps)
        self.reactions.append([reactants, products])
        self.reactions.append([products, reactants])

        if add_to_kmc:
            self.unique_reaction_nodes.append(reaction_forward_name)
            self.unique_reaction_nodes.append(reaction_reverse_name)
        
        if add_to_graph:
            if len(reactants) == 1:
                if len(products) == 1:
                    self.reaction_network.graph.add_edge(int(reactant_name),
                                                         reaction_forward_name,
                                                         softplus=self.reaction_network.softplus(free_energy_forward),
                                                         exponent=self.reaction_network.exponent(free_energy_forward),
                                                         weight=1.0
                                                         )
                    self.reaction_network.graph.add_edge(reaction_forward_name,
                                                         int(product_name),
                                                         softplus=0.0,
                                                         exponent=0.0,
                                                         weight=1.0
                                                         )
                    self.reaction_network.graph.add_edge(int(product_name),
                                                         reaction_reverse_name,
                                                         softplus=self.reaction_network.softplus(free_energy_reverse),
                                                         exponent=self.reaction_network.exponent(free_energy_reverse),
                                                         weight=1.0
                                                         )
                    self.reaction_network.graph.add_edge(reaction_reverse_name,
                                                         int(reactant_name),
                                                         softplus=0.0,
                                                         exponent=0.0,
                                                         weight=1.0
                                                         )
                elif len(products) == 2:
                    C_PR_D_name = products[0] + "+PR_" + products[1]
                    D_PR_C_name = products[1] + "+PR_" + products[0]
                    node_name_reverse_1 = C_PR_D_name + "," + reactant_name
                    node_name_reverse_2 = D_PR_C_name + "," + reactant_name

                    self.reaction_network.graph.add_edge(int(reactant_name),
                                                         reaction_forward_name,
                                                         softplus=self.reaction_network.softplus(
                                                             free_energy_forward),
                                                         exponent=self.reaction_network.exponent(
                                                             free_energy_forward),
                                                         weight=1.0
                                                         )
                    self.reaction_network.graph.add_edge(reaction_forward_name,
                                                         int(products[0]),
                                                         softplus=0.0,
                                                         exponent=0.0,
                                                         weight=1.0
                                                         )
                    self.reaction_network.graph.add_edge(reaction_forward_name,
                                                         int(products[1]),
                                                         softplus=0.0,
                                                         exponent=0.0,
                                                         weight=1.0
                                                         )

                    self.reaction_network.graph.add_node(node_name_reverse_1,
                                                         rxn_type=rxn_type,
                                                         bipartite=1,
                                                         energy=energy_reverse,
                                                         free_energy=free_energy_reverse,
                                                         charge_change=-total_charge_change,
                                                         steps=steps)
                    self.reaction_network.graph.add_edge(int(products[0]),
                                                         node_name_reverse_1,
                                                         softplus=self.reaction_network.softplus(
                                                             free_energy_reverse),
                                                         exponent=self.reaction_network.exponent(
                                                             free_energy_reverse),
                                                         weight=1.0
                                                         )
                    self.reaction_network.graph.add_edge(node_name_reverse_1,
                                                         reactant_name,
                                                         softplus=0.0,
                                                         exponent=0.0,
                                                         weight=1.0
                                                         )

                    self.reaction_network.graph.add_node(node_name_reverse_2,
                                                         rxn_type=rxn_type,
                                                         bipartite=1,
                                                         energy=energy_reverse,
                                                         free_energy=free_energy_reverse,
                                                         charge_change=-total_charge_change,
                                                         steps=steps)
                    self.reaction_network.graph.add_edge(int(products[1]),
                                                         node_name_reverse_2,
                                                         softplus=self.reaction_network.softplus(
                                                             free_energy_reverse),
                                                         exponent=self.reaction_network.exponent(
                                                             free_energy_reverse),
                                                         weight=1.0
                                                         )
                    self.reaction_network.graph.add_edge(node_name_reverse_2,
                                                         reactant_name,
                                                         softplus=0.0,
                                                         exponent=0.0,
                                                         weight=1.0
                                                         )

            elif len(reactants) == 2:
                if len(products) == 1:
                    A_PR_B_name = reactants[0] + "+PR_" + reactants[1]
                    B_PR_A_name = reactants[1] + "+PR_" + reactants[0]
                    node_name_forward_1 = A_PR_B_name + "," + product_name
                    node_name_forward_2 = B_PR_A_name + "," + product_name

                    self.reaction_network.graph.add_node(node_name_forward_1,
                                                         rxn_type=rxn_type,
                                                         bipartite=1,
                                                         energy=energy_forward,
                                                         free_energy=free_energy_forward,
                                                         charge_change=total_charge_change,
                                                         steps=steps)
                    self.reaction_network.graph.add_edge(int(reactants[0]),
                                                         node_name_forward_1,
                                                         softplus=self.reaction_network.softplus(
                                                             free_energy_forward),
                                                         exponent=self.reaction_network.exponent(
                                                             free_energy_forward),
                                                         weight=1.0
                                                         )
                    self.reaction_network.graph.add_edge(node_name_forward_1,
                                                         int(product_name),
                                                         softplus=0.0,
                                                         exponent=0.0,
                                                         weight=1.0
                                                         )

                    self.reaction_network.graph.add_node(node_name_forward_2,
                                                         rxn_type=rxn_type,
                                                         bipartite=1,
                                                         energy=energy_forward,
                                                         free_energy=free_energy_forward,
                                                         charge_change=total_charge_change,
                                                         steps=steps)
                    self.reaction_network.graph.add_edge(int(reactants[1]),
                                                         node_name_forward_2,
                                                         softplus=self.reaction_network.softplus(
                                                             free_energy_forward),
                                                         exponent=self.reaction_network.exponent(
                                                             free_energy_forward),
                                                         weight=1.0
                                                         )
                    self.reaction_network.graph.add_edge(node_name_forward_2,
                                                         int(product_name),
                                                         softplus=0.0,
                                                         exponent=0.0,
                                                         weight=1.0
                                                         )

                    self.reaction_network.graph.add_edge(int(product_name),
                                                         reaction_reverse_name,
                                                         softplus=self.reaction_network.softplus(
                                                             free_energy_reverse),
                                                         exponent=self.reaction_network.exponent(
                                                             free_energy_reverse),
                                                         weight=1.0
                                                         )
                    self.reaction_network.graph.add_edge(reaction_reverse_name,
                                                         int(reactants[0]),
                                                         softplus=0.0,
                                                         exponent=0.0,
                                                         weight=1.0
                                                         )
                    self.reaction_network.graph.add_edge(reaction_forward_name,
                                                         int(reactants[1]),
                                                         softplus=0.0,
                                                         exponent=0.0,
                                                         weight=1.0
                                                         )

                elif len(products) == 2:
                    A_PR_B_name = reactants[0] + "+PR_" + reactants[1]
                    B_PR_A_name = reactants[1] + "+PR_" + reactants[0]
                    C_PR_D_name = products[0] + "+PR_" + products[1]
                    D_PR_C_name = products[1] + "+PR_" + products[0]

                    node_name_forward_1 = A_PR_B_name + "," + product_name
                    node_name_forward_2 = B_PR_A_name + "," + product_name
                    node_name_reverse_1 = C_PR_D_name + "," + reactant_name
                    node_name_reverse_2 = D_PR_C_name + "," + reactant_name

                    self.reaction_network.graph.add_node(node_name_forward_1,
                                                         rxn_type=rxn_type,
                                                         bipartite=1,
                                                         energy=energy_forward,
                                                         free_energy=free_energy_forward,
                                                         charge_change=total_charge_change,
                                                         steps=steps)
                    self.reaction_network.graph.add_edge(int(reactants[0]),
                                                         node_name_forward_1,
                                                         softplus=self.reaction_network.softplus(
                                                             free_energy_forward),
                                                         exponent=self.reaction_network.exponent(
                                                             free_energy_forward),
                                                         weight=1.0
                                                         )
                    self.reaction_network.graph.add_edge(node_name_forward_1,
                                                         int(products[0]),
                                                         softplus=0.0,
                                                         exponent=0.0,
                                                         weight=1.0
                                                         )
                    self.reaction_network.graph.add_edge(node_name_forward_1,
                                                         int(products[1]),
                                                         softplus=0.0,
                                                         exponent=0.0,
                                                         weight=1.0
                                                         )

                    self.reaction_network.graph.add_node(node_name_forward_2,
                                                         rxn_type=rxn_type,
                                                         bipartite=1,
                                                         energy=energy_forward,
                                                         free_energy=free_energy_forward,
                                                         charge_change=total_charge_change,
                                                         steps=steps)
                    self.reaction_network.graph.add_edge(int(reactants[1]),
                                                         node_name_forward_2,
                                                         softplus=self.reaction_network.softplus(
                                                             free_energy_forward),
                                                         exponent=self.reaction_network.exponent(
                                                             free_energy_forward),
                                                         weight=1.0
                                                         )
                    self.reaction_network.graph.add_edge(node_name_forward_2,
                                                         int(products[0]),
                                                         softplus=0.0,
                                                         exponent=0.0,
                                                         weight=1.0
                                                         )
                    self.reaction_network.graph.add_edge(node_name_forward_2,
                                                         int(products[1]),
                                                         softplus=0.0,
                                                         exponent=0.0,
                                                         weight=1.0
                                                         )

                    self.reaction_network.graph.add_node(node_name_reverse_1,
                                                         rxn_type=rxn_type,
                                                         bipartite=1,
                                                         energy=energy_reverse,
                                                         free_energy=free_energy_reverse,
                                                         charge_change=-total_charge_change,
                                                         steps=steps)
                    self.reaction_network.graph.add_edge(int(products[0]),
                                                         node_name_reverse_1,
                                                         softplus=self.reaction_network.softplus(
                                                             free_energy_reverse),
                                                         exponent=self.reaction_network.exponent(
                                                             free_energy_reverse),
                                                         weight=1.0
                                                         )
                    self.reaction_network.graph.add_edge(node_name_reverse_1,
                                                         int(reactants[0]),
                                                         softplus=0.0,
                                                         exponent=0.0,
                                                         weight=1.0
                                                         )
                    self.reaction_network.graph.add_edge(node_name_reverse_1,
                                                         int(reactants[1]),
                                                         softplus=0.0,
                                                         exponent=0.0,
                                                         weight=1.0
                                                         )

                    self.reaction_network.graph.add_node(node_name_reverse_2,
                                                         rxn_type=rxn_type,
                                                         bipartite=1,
                                                         energy=energy_reverse,
                                                         free_energy=free_energy_reverse,
                                                         charge_change=-total_charge_change,
                                                         steps=steps)
                    self.reaction_network.graph.add_edge(int(products[1]),
                                                         node_name_reverse_2,
                                                         softplus=self.reaction_network.softplus(
                                                             free_energy_reverse),
                                                         exponent=self.reaction_network.exponent(
                                                             free_energy_reverse),
                                                         weight=1.0
                                                         )
                    self.reaction_network.graph.add_edge(node_name_reverse_2,
                                                         int(reactants[0]),
                                                         softplus=0.0,
                                                         exponent=0.0,
                                                         weight=1.0
                                                         )
                    self.reaction_network.graph.add_edge(node_name_reverse_2,
                                                         int(reactants[1]),
                                                         softplus=0.0,
                                                         exponent=0.0,
                                                         weight=1.0
                                                         )
            
        return

    def add_concerted_reactions_2_step(self, num_of_mols, num_thresh, terminate=False):
        '''
        Add concerted reactions on the fly, only for species that have non-zero concentration.
        This function only adds two-step concerted reactions.
        :param num_of_mols:
        :param add_to_graph: Whether to connect the reaction node with reactants and products in self.reaction_network.graph.
                      If False, only the reaction node will be added to the graph.
        :param terminate: Whether to terminate at 2-step concerted.
               If true, do not need to consider uphill 2-step concerted reactions.
        :return:
        '''

        non_zero_indices = [i for i in range(len(num_of_mols)) if num_of_mols[i] > num_thresh]
        for mol_id in non_zero_indices:
            # exclude electron from the species that need to search neighbors from
            if mol_id == self.num_species-1:
                continue
            if mol_id in self.searched_species_2_step:
                continue
            else:
                self.searched_species_2_step.append(mol_id)
            neighbor_rxns = list(self.reaction_network.graph.neighbors(mol_id))
            for rxn0 in neighbor_rxns:
                if (terminate and (self.reaction_network.graph.node[rxn0]["free_energy"] > 0)) or (not terminate):
                    rxn0_reactants = rxn0.split(",")[0].split("+")
                    rxn0_reactants = [reac.replace("PR_","") for reac in rxn0_reactants]
                    if rxn0_reactants == ['2715']:
                        print('LiEC+ monodentate found!')
                    rxn0_products = list(self.reaction_network.graph.neighbors(rxn0))
                    rxn0_products = [str(prod) for prod in rxn0_products]
                    for node1 in rxn0_products:
                        rxn0_products_copy = copy.deepcopy(rxn0_products)
                        rxn0_products_copy.remove(str(node1))
                        middle_products = rxn0_products_copy
                        node1_rxns = list(self.reaction_network.graph.neighbors(int(node1)))
                        for rxn1 in node1_rxns:
                            if (terminate and (self.reaction_network.graph.node[rxn0]["free_energy"] +
                                               self.reaction_network.graph.node[rxn1]["free_energy"] < 0)) \
                                    or (not terminate):
                                rxn1_reactants = rxn1.split(",")[0].split("+")
                                rxn1_reactants = [reac.replace("PR_", "") for reac in rxn1_reactants]
                                #rxn1_reactants_copy = copy.deepcopy(rxn1_reactants)
                                rxn1_reactants.remove(str(node1))
                                middle_reactants = rxn1_reactants
                                rxn1_products = list(self.reaction_network.graph.neighbors(rxn1))
                                rxn1_products = [str(prod) for prod in rxn1_products]
                                total_reactants = rxn0_reactants + middle_reactants
                                total_products = middle_products + rxn1_products
                                total_reactants.sort()
                                total_products.sort()
                                if total_products == ['2720'] and total_reactants == ['2715']:
                                    print('2720, 2715 found!')

                                total_species = total_reactants + total_products
                                total_species_set = list(set(total_species))
                                # remove species that appear both in reactants and products
                                for species in total_species_set:
                                    while (species in total_reactants and species in total_products):
                                        total_reactants.remove(species)
                                        total_products.remove(species)
                                # check the reaction is not in the existing reactions, and both the number of reactants and products less than 2
                                if ([total_reactants, total_products] not in self.reactions) and \
                                        (1 <= len(total_reactants) <= 2) and (1 <= len(total_products) <= 2):
                                    unique_elements = []
                                    for species in total_species_set:
                                        unique_elements += list(self.reaction_network.entries_list[int(species)].molecule.atomic_numbers)
                                    unique_elements = list(set(unique_elements))
                                    reactant_elements, product_elements = [], []
                                    for species in total_reactants:
                                        reactant_elements += list(self.reaction_network.entries_list[int(species)].molecule.atomic_numbers)
                                    for species in total_products:
                                        product_elements += list(self.reaction_network.entries_list[int(species)].molecule.atomic_numbers)
                                    # check stoichiometry
                                    if all(reactant_elements.count(ele) == product_elements.count(ele) for ele in unique_elements):
                                        if terminate:
                                            if total_products == ['2720'] and total_reactants == ['2715']:
                                                print('2720, 2715 persists!')
                                            self.add_reactions_to_graph(total_reactants, total_products,
                                                                        "two step concerted", 2, add_to_graph=False, add_to_kmc=True)
                                        else:
                                            if (self.reaction_network.graph.node[rxn0]["free_energy"] > 0) and \
                                                    (self.reaction_network.graph.node[rxn0]["free_energy"] +
                                                     self.reaction_network.graph.node[rxn1]["free_energy"] < 0):
                                                self.add_reactions_to_graph(total_reactants, total_products,
                                                                            "two step concerted", 2, add_to_graph=True, add_to_kmc=True)
                                            # Not considering downhill-downhill reactions
                                            elif self.reaction_network.graph.node[rxn0]["free_energy"] + \
                                                     self.reaction_network.graph.node[rxn1]["free_energy"] > 0:
                                                self.add_reactions_to_graph(total_reactants, total_products,
                                                                            "two step concerted", 2, add_to_graph=True, add_to_kmc=False)

    def add_concerted_reactions_3_step(self, num_of_mols, num_thresh, terminate=False):
        '''
        Add concerted reactions on the fly, only for species that have non-zero concentration.
        This function adds 3 or 4-step concerted reactions.
        :param num_of_mols:
        :param add_to_graph: Whether to connect the reaction node with reactants and products in self.reaction_network.graph.
                      If False, only the reaction node will be added to the graph.
        :param terminate: Whether to terminate at 3-step concerted.
               If true, do not need to consider uphill 3-step concerted reactions.
        :return:
        '''

        non_zero_indices = [i for i in range(len(num_of_mols)) if num_of_mols[i] > num_thresh]
        for mol_id in non_zero_indices:
            # exclude electron from the species that need to search neighbors from
            if mol_id == self.num_species-1:
                continue
            if mol_id in self.searched_species_3_step:
                continue
            else:
                self.searched_species_3_step.append(mol_id)
            neighbor_rxns = list(self.reaction_network.graph.neighbors(mol_id))
            for rxn0 in neighbor_rxns:
                if self.reaction_network.graph.node[rxn0]['steps'] == 2:
                    if (terminate and (self.reaction_network.graph.node[rxn0]["free_energy"] > 0)) or (not terminate):
                        rxn0_reactants = rxn0.split(",")[0].split("+")
                        rxn0_reactants = [reac.replace("PR_","") for reac in rxn0_reactants]
                        rxn0_products = list(self.reaction_network.graph.neighbors(rxn0))
                        rxn0_products = [str(prod) for prod in rxn0_products]
                        for node1 in rxn0_products:
                            rxn0_products_copy = copy.deepcopy(rxn0_products)
                            rxn0_products_copy.remove(str(node1))
                            middle_products = rxn0_products_copy
                            node1_rxns = list(self.reaction_network.graph.neighbors(int(node1)))
                            for rxn1 in node1_rxns:
                                if self.reaction_network.graph.node[rxn0]['steps'] + self.reaction_network.graph.node[rxn1]['steps'] == 3:
                                    if (terminate and (self.reaction_network.graph.node[rxn0]["free_energy"] +
                                                       self.reaction_network.graph.node[rxn1]["free_energy"] < 0)) \
                                            or (not terminate):
                                        rxn1_reactants = rxn1.split(",")[0].split("+")
                                        rxn1_reactants = [reac.replace("PR_", "") for reac in rxn1_reactants]
                                        #rxn1_reactants_copy = copy.deepcopy(rxn1_reactants)
                                        rxn1_reactants.remove(str(node1))
                                        middle_reactants = rxn1_reactants
                                        rxn1_products = list(self.reaction_network.graph.neighbors(rxn1))
                                        rxn1_products = [str(prod) for prod in rxn1_products]
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
                                                (1 <= len(total_reactants) <= 2) and (1 <= len(total_products) <= 2):
                                            unique_elements = []
                                            for species in total_species_set:
                                                unique_elements += list(self.reaction_network.entries_list[int(species)].molecule.atomic_numbers)
                                            unique_elements = list(set(unique_elements))
                                            reactant_elements, product_elements = [], []
                                            for species in total_reactants:
                                                reactant_elements += list(self.reaction_network.entries_list[int(species)].molecule.atomic_numbers)
                                            for species in total_products:
                                                product_elements += list(self.reaction_network.entries_list[int(species)].molecule.atomic_numbers)
                                            # check stoichiometry
                                            if all(reactant_elements.count(ele) == product_elements.count(ele) for ele in unique_elements):
                                                if terminate:
                                                    self.add_reactions_to_graph(total_reactants, total_products,
                                                                                "three step concerted", 3,
                                                                                add_to_graph=False, add_to_kmc=True)
                                                else:
                                                    if (self.reaction_network.graph.node[rxn0]["free_energy"] > 0) and \
                                                            (self.reaction_network.graph.node[rxn0]["free_energy"] +
                                                             self.reaction_network.graph.node[rxn1]["free_energy"] < 0):
                                                        self.add_reactions_to_graph(total_reactants, total_products,
                                                                                    "three step concerted", 3,
                                                                                    add_to_graph=True, add_to_kmc=True)
                                                    # Not considering downhill-downhill reactions
                                                    elif self.reaction_network.graph.node[rxn0]["free_energy"] + \
                                                            self.reaction_network.graph.node[rxn1]["free_energy"] > 0:
                                                        self.add_reactions_to_graph(total_reactants, total_products,
                                                                                    "three step concerted", 3,
                                                                                    add_to_graph=True, add_to_kmc=False)

        return

    def add_concerted_reactions_4_step(self, num_of_mols, num_thresh, terminate=False):
        '''
        Add concerted reactions on the fly, only for species that have non-zero concentration.
        This function adds 3 or 4-step concerted reactions.
        :param num_of_mols:
        :param add_to_graph: Whether to connect the reaction node with reactants and products in self.reaction_network.graph.
                      If False, only the reaction node will be added to the graph.
        :param terminate: Whether to terminate at 4-step concerted.
               If true, do not need to consider uphill 4-step concerted reactions.
        :return:
        '''

        non_zero_indices = [i for i in range(len(num_of_mols)) if num_of_mols[i] > num_thresh]
        for mol_id in non_zero_indices:
            # exclude electron from the species that need to search neighbors from
            if mol_id == self.num_species-1:
                continue
            if mol_id in self.searched_species_4_step:
                continue
            else:
                self.searched_species_4_step.append(mol_id)
            neighbor_rxns = list(self.reaction_network.graph.neighbors(mol_id))
            for rxn0 in neighbor_rxns:
                if self.reaction_network.graph.node[rxn0]['steps'] == 3:
                    if (terminate and (self.reaction_network.graph.node[rxn0]["free_energy"] > 0)) or (not terminate):
                        rxn0_reactants = rxn0.split(",")[0].split("+")
                        rxn0_reactants = [reac.replace("PR_","") for reac in rxn0_reactants]
                        rxn0_products = list(self.reaction_network.graph.neighbors(rxn0))
                        rxn0_products = [str(prod) for prod in rxn0_products]
                        for node1 in rxn0_products:
                            rxn0_products_copy = copy.deepcopy(rxn0_products)
                            rxn0_products_copy.remove(str(node1))
                            middle_products = rxn0_products_copy
                            node1_rxns = list(self.reaction_network.graph.neighbors(int(node1)))
                            for rxn1 in node1_rxns:
                                if self.reaction_network.graph.node[rxn0]['steps'] + self.reaction_network.graph.node[rxn1]['steps'] == 4:
                                    if (terminate and (self.reaction_network.graph.node[rxn0]["free_energy"] +
                                                       self.reaction_network.graph.node[rxn1]["free_energy"] < 0)) \
                                            or (not terminate):
                                        rxn1_reactants = rxn1.split(",")[0].split("+")
                                        rxn1_reactants = [reac.replace("PR_", "") for reac in rxn1_reactants]
                                        #rxn1_reactants_copy = copy.deepcopy(rxn1_reactants)
                                        rxn1_reactants.remove(str(node1))
                                        middle_reactants = rxn1_reactants
                                        rxn1_products = list(self.reaction_network.graph.neighbors(rxn1))
                                        rxn1_products = [str(prod) for prod in rxn1_products]
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
                                                (1 <= len(total_reactants) <= 2) and (1 <= len(total_products) <= 2):
                                            unique_elements = []
                                            for species in total_species_set:
                                                unique_elements += list(self.reaction_network.entries_list[int(species)].molecule.atomic_numbers)
                                            unique_elements = list(set(unique_elements))
                                            reactant_elements, product_elements = [], []
                                            for species in total_reactants:
                                                reactant_elements += list(self.reaction_network.entries_list[int(species)].molecule.atomic_numbers)
                                            for species in total_products:
                                                product_elements += list(self.reaction_network.entries_list[int(species)].molecule.atomic_numbers)
                                            # check stoichiometry
                                            if all(reactant_elements.count(ele) == product_elements.count(ele) for ele in unique_elements):
                                                if terminate:
                                                    self.add_reactions_to_graph(total_reactants, total_products,
                                                                                "four step concerted", 4,
                                                                                add_to_graph=False, add_to_kmc=True)
                                                else:
                                                    if (self.reaction_network.graph.node[rxn0]["free_energy"] > 0) and \
                                                            (self.reaction_network.graph.node[rxn0]["free_energy"] +
                                                             self.reaction_network.graph.node[rxn1]["free_energy"] < 0):
                                                        self.add_reactions_to_graph(total_reactants, total_products,
                                                                                    "four step concerted", 4,
                                                                                    add_to_graph=True, add_to_kmc=True)
                                                    # Not considering downhill-downhill reactions
                                                    elif self.reaction_network.graph.node[rxn0]["free_energy"] + \
                                                            self.reaction_network.graph.node[rxn1]["free_energy"] > 0:
                                                        self.add_reactions_to_graph(total_reactants, total_products,
                                                                                    "four step concerted", 4,
                                                                                    add_to_graph=True, add_to_kmc=False)
        return

    def search_2_step(self,mol_id, add_to_graph=True, add_to_kmc=True):
        # Only search for downhill-uphill 2-step concerted reactions and add to graph, add to kmc
        if mol_id == self.num_species - 1:
            pass
        if mol_id in self.searched_species_2_step:
            pass
        else:
            self.searched_species_2_step.append(mol_id)
        neighbor_rxns = list(self.reaction_network.graph.neighbors(mol_id))
        for rxn0 in neighbor_rxns:
            if self.reaction_network.graph.node[rxn0]["free_energy"] > 0:
                rxn0_reactants = rxn0.split(",")[0].split("+")
                rxn0_reactants = [reac.replace("PR_", "") for reac in rxn0_reactants]
                rxn0_products = list(self.reaction_network.graph.neighbors(rxn0))
                rxn0_products = [str(prod) for prod in rxn0_products]
                for node1 in rxn0_products:
                    rxn0_products_copy = copy.deepcopy(rxn0_products)
                    rxn0_products_copy.remove(str(node1))
                    middle_products = rxn0_products_copy
                    node1_rxns = list(self.reaction_network.graph.neighbors(int(node1)))
                    for rxn1 in node1_rxns:
                        if self.reaction_network.graph.node[rxn0]["free_energy"] + \
                                self.reaction_network.graph.node[rxn1]["free_energy"] < 0:
                            rxn1_reactants = rxn1.split(",")[0].split("+")
                            rxn1_reactants = [reac.replace("PR_", "") for reac in rxn1_reactants]
                            # rxn1_reactants_copy = copy.deepcopy(rxn1_reactants)
                            rxn1_reactants.remove(str(node1))
                            middle_reactants = rxn1_reactants
                            rxn1_products = list(self.reaction_network.graph.neighbors(rxn1))
                            rxn1_products = [str(prod) for prod in rxn1_products]
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
                                    (1 <= len(total_reactants) <= 2) and (1 <= len(total_products) <= 2):
                                unique_elements = []
                                for species in total_species_set:
                                    unique_elements += list(
                                        self.reaction_network.entries_list[int(species)].molecule.atomic_numbers)
                                unique_elements = list(set(unique_elements))
                                reactant_elements, product_elements = [], []
                                for species in total_reactants:
                                    reactant_elements += list(
                                        self.reaction_network.entries_list[int(species)].molecule.atomic_numbers)
                                for species in total_products:
                                    product_elements += list(
                                        self.reaction_network.entries_list[int(species)].molecule.atomic_numbers)
                                # check stoichiometry
                                if all(reactant_elements.count(ele) == product_elements.count(ele) for ele in
                                       unique_elements):

                                    self.add_reactions_to_graph(total_reactants, total_products,
                                                                "two step concerted", 2, add_to_graph=add_to_graph,
                                                                add_to_kmc=add_to_kmc)

    def search_3_step_from_2_step(self,mol_id, add_to_graph=True, add_to_kmc=True):
        '''
        Only add uphill-uphill-downhill 3-step concerted reactions. The last 2 step taken from self.search_2_step.
        :param mol_id:
        :param add_to_graph:
        :param add_to_kmc:
        :return:
        '''
        if mol_id == self.num_species - 1:
            pass
        if mol_id in self.searched_species_3_step:
            pass
        else:
            self.searched_species_3_step.append(mol_id)
        neighbor_rxns = list(self.reaction_network.graph.neighbors(mol_id))
        for rxn0 in neighbor_rxns:
            if self.reaction_network.graph.node[rxn0]['steps'] == 1:
                if self.reaction_network.graph.node[rxn0]["free_energy"] > 0:
                    rxn0_reactants = rxn0.split(",")[0].split("+")
                    rxn0_reactants = [reac.replace("PR_", "") for reac in rxn0_reactants]
                    rxn0_products = list(self.reaction_network.graph.neighbors(rxn0))
                    rxn0_products = [str(prod) for prod in rxn0_products]
                    for node1 in rxn0_products:
                        rxn0_products_copy = copy.deepcopy(rxn0_products)
                        rxn0_products_copy.remove(str(node1))
                        middle_products = rxn0_products_copy
                        if node1 not in self.searched_species_2_step:
                            self.search_2_step(int(node1),add_to_graph=True, add_to_kmc=True)
                        node1_rxns = list(self.reaction_network.graph.neighbors(int(node1)))
                        for rxn1 in node1_rxns:
                            if self.reaction_network.graph.node[rxn1]['steps'] == 2:
                                if self.reaction_network.graph.node[rxn0]["free_energy"] + self.reaction_network.graph.node[rxn1]["free_energy"] < 0:
                                    rxn1_reactants = rxn1.split(",")[0].split("+")
                                    rxn1_reactants = [reac.replace("PR_", "") for reac in rxn1_reactants]
                                    # rxn1_reactants_copy = copy.deepcopy(rxn1_reactants)
                                    rxn1_reactants.remove(str(node1))
                                    middle_reactants = rxn1_reactants
                                    rxn1_products = list(self.reaction_network.graph.neighbors(rxn1))
                                    rxn1_products = [str(prod) for prod in rxn1_products]
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
                                            (1 <= len(total_reactants) <= 2) and (1 <= len(total_products) <= 2):
                                        unique_elements = []
                                        for species in total_species_set:
                                            unique_elements += list(self.reaction_network.entries_list[
                                                                        int(species)].molecule.atomic_numbers)
                                        unique_elements = list(set(unique_elements))
                                        reactant_elements, product_elements = [], []
                                        for species in total_reactants:
                                            reactant_elements += list(self.reaction_network.entries_list[
                                                                          int(species)].molecule.atomic_numbers)
                                        for species in total_products:
                                            product_elements += list(self.reaction_network.entries_list[
                                                                         int(species)].molecule.atomic_numbers)
                                        # check stoichiometry
                                        if all(reactant_elements.count(ele) == product_elements.count(ele) for ele in
                                               unique_elements):

                                            self.add_reactions_to_graph(total_reactants, total_products,
                                                                        "three step concerted", 3,
                                                                        add_to_graph=add_to_graph, add_to_kmc=add_to_kmc)


    def add_concerted_reactions_2_step_new(self, num_of_mols, num_thresh, terminate=False):
        '''
        Add concerted reactions on the fly, only for species that have non-zero concentration.
        This function only adds two-step concerted reactions.
        :param num_of_mols:
        :param add_to_graph: Whether to connect the reaction node with reactants and products in self.reaction_network.graph.
                      If False, only the reaction node will be added to the graph.
        :param terminate: Whether to terminate at 2-step concerted.
               If true, do not need to consider uphill 2-step concerted reactions.
        :return:
        '''

        non_zero_indices = [i for i in range(len(num_of_mols)) if num_of_mols[i] > num_thresh]
        for mol_id in non_zero_indices:
            if terminate:
                self.search_2_step(mol_id, add_to_graph=False, add_to_kmc=True)
            else:
                self.search_2_step(mol_id, add_to_graph=True, add_to_kmc=True)
        return

    def add_concerted_reactions_3_step_new(self, num_of_mols, num_thresh, terminate=False):
        '''
        Add concerted reactions on the fly, only for species that have non-zero concentration.
        This function adds 3 or 4-step concerted reactions.
        :param num_of_mols:
        :param add_to_graph: Whether to connect the reaction node with reactants and products in self.reaction_network.graph.
                      If False, only the reaction node will be added to the graph.
        :param terminate: Whether to terminate at 3-step concerted.
               If true, do not need to consider uphill 3-step concerted reactions.
        :return:
        '''

        non_zero_indices = [i for i in range(len(num_of_mols)) if num_of_mols[i] > num_thresh]
        for mol_id in non_zero_indices:
            if terminate:
                self.search_3_step_from_2_step(mol_id, add_to_graph=False, add_to_kmc=True)
            else:
                self.search_3_step_from_2_step(mol_id, add_to_graph=True, add_to_kmc=True)
        return

    def direct_method(self, initial_conc, time_span):
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
        # t = np.zeros(max_output_length)
        # x = np.zeros([max_output_length, self.num_species])
        # rxns = np.zeros(max_output_length)
        t = [0]
        #x[0,:] = initial_conc
        x = np.array([initial_conc])
        rxns = []
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

            # if rxn_count + 1 > max_output_length:
            #     t = t[:rxn_count]
            #     x = x[:rxn_count]
            #     print("WARNING:Number of reaction events exceeded the number pre-allocated. Simulation terminated prematurely.")

            #t[rxn_count + 1] = t[rxn_count] + tau
            #x[rxn_count + 1] = x[rxn_count]
            t.append(t[rxn_count] + tau)
            x = np.vstack([x,x[-1]])
            current_reaction = self.unique_reaction_nodes[mu]
            reactants = current_reaction.split(",")[0].split("+")
            reactants = [reac.replace("PR_", "") for reac in reactants]
            products = current_reaction.split(",")[1].split("+")
            for reac in reactants:
                x[rxn_count+1, int(reac)] -= 1
            for prod in products:
                x[rxn_count+1, int(prod)] += 1
            if self.reaction_network.graph.node[current_reaction]["charge_change"] != 0:
                x[rxn_count + 1, -1] += self.reaction_network.graph.node[current_reaction]["charge_change"]
            rxns.append(mu)
            rxn_count += 1

        # t = t[:rxn_count]
        # x = x[:rxn_count,:]
        # rxns = rxns[:rxn_count]
        if t[-1] > time_span:
            t[-1] = time_span
            x[-1,:] = x[rxn_count-1,:]
            rxns[-1] = rxns[rxn_count-1]
        return t, x, rxns, records

    def direct_method_no_record(self, initial_conc, time_span):
        '''
        :param initial_conc
        :param time_span: int, number of time steps
        :param max_output_length
        Gillespie, D.T. (1977) Exact Stochastic Simulation of Coupled Chemical Reactions. J Phys Chem, 81:25, 2340-2361.
        :return: t: time vector (Nreaction_events x 1); x:  species amounts (Nreaction_events x Nspecies);
                 rxns: rxns that have been fired up (Nreaction_events x 1).
        '''
        # t = np.zeros(max_output_length)
        # x = np.zeros([max_output_length, self.num_species])
        # rxns = np.zeros(max_output_length)
        t = [0]
        #x[0,:] = initial_conc
        x = np.array([initial_conc])
        rxns = []
        rxn_count = 0

        while t[rxn_count] < time_span:
            a = self.get_propensities(x[rxn_count,:])
            a0 = np.sum(a)
            random_t = random.uniform(0, 1)
            random_rxn = random.uniform(0, 1)
            tau = -np.log(random_t) / a0
            mu = np.where(np.cumsum(a) >= random_rxn * a0)[0][0]

            # if rxn_count + 1 > max_output_length:
            #     t = t[:rxn_count]
            #     x = x[:rxn_count]
            #     print("WARNING:Number of reaction events exceeded the number pre-allocated. Simulation terminated prematurely.")

            #t[rxn_count + 1] = t[rxn_count] + tau
            #x[rxn_count + 1] = x[rxn_count]
            t.append(t[rxn_count] + tau)
            x = np.vstack([x,x[-1]])
            current_reaction = self.unique_reaction_nodes[mu]
            reactants = current_reaction.split(",")[0].split("+")
            reactants = [reac.replace("PR_", "") for reac in reactants]
            products = current_reaction.split(",")[1].split("+")
            for reac in reactants:
                x[rxn_count+1, int(reac)] -= 1
            for prod in products:
                x[rxn_count+1, int(prod)] += 1
            if self.reaction_network.graph.node[current_reaction]["charge_change"] != 0:
                x[rxn_count + 1, -1] += self.reaction_network.graph.node[current_reaction]["charge_change"]
            rxns.append(mu)
            rxn_count += 1

        # t = t[:rxn_count]
        # x = x[:rxn_count,:]
        # rxns = rxns[:rxn_count]
        if t[-1] > time_span:
            t[-1] = time_span
            x[-1,:] = x[rxn_count-1,:]
            rxns[-1] = rxns[rxn_count-1]
        return t, x, rxns

    def add_two_step_concerted_reactions_on_the_fly(self, initial_conc, time_span, barrier_uni, barrier_bi, xyz_dir,iterations=5,remove_Li_red=False):
        self.searched_species_2_step = []
        iter = 0
        t, x, rxns = 0,0,0
        print("current nums of reactions:", len(self.unique_reaction_nodes))
        while iter < iterations:
            t, x, rxns = self.direct_method_no_record(initial_conc, time_span)
            if not iterations - iter == 1:
                self.add_concerted_reactions_2_step(x[-1,:], 10)
                print("current nums of reactions:", len(self.unique_reaction_nodes))
                self.get_rates(barrier_uni, barrier_bi)
                self.remove_gas_reactions(xyz_dir)
                if remove_Li_red:
                    self.remove_Li_reduction_reaction(xyz_dir)
            iter += 1
        return t,x,rxns

    def add_three_step_concerted_reactions_on_the_fly(self, initial_conc, time_span, barrier_uni, barrier_bi, xyz_dir,iterations=5,remove_Li_red=False):
        self.searched_species_2_step = []
        self.searched_species_3_step = []
        iter = 0
        t, x, rxns = 0,0,0
        print("current nums of reactions:", len(self.unique_reaction_nodes))
        while iter < iterations:
            t, x, rxns = self.direct_method_no_record(initial_conc, time_span)
            if not iterations - iter == 1:
                self.add_concerted_reactions_2_step(x[-1,:], 10, terminate=False)
                self.add_concerted_reactions_3_step(x[-1,:], 10, terminate=True)
                print("current nums of reactions:", len(self.unique_reaction_nodes))
                self.get_rates(barrier_uni, barrier_bi)
                self.remove_gas_reactions(xyz_dir)
                if remove_Li_red:
                    self.remove_Li_reduction_reaction(xyz_dir)
            iter += 1
        return t,x,rxns

    def add_four_step_concerted_reactions_on_the_fly(self, initial_conc, time_span, barrier_uni, barrier_bi, xyz_dir,iterations=5,remove_Li_red=False):
        self.searched_species_2_step = []
        self.searched_species_3_step = []
        self.searched_species_4_step = []
        iter = 0
        t, x, rxns = 0,0,0
        print("current nums of reactions:", len(self.unique_reaction_nodes))
        while iter < iterations:
            t, x, rxns = self.direct_method_no_record(initial_conc, time_span)
            if not iterations - iter == 1:
                self.add_concerted_reactions_2_step(x[-1,:], 10, terminate=False)
                self.add_concerted_reactions_3_step(x[-1,:], 10, terminate=False)
                self.add_concerted_reactions_4_step(x[-1,:], 10, terminate=True)
                print("current nums of reactions:", len(self.unique_reaction_nodes))
                self.get_rates(barrier_uni, barrier_bi)
                self.remove_gas_reactions(xyz_dir)
                if remove_Li_red:
                    self.remove_Li_reduction_reaction(xyz_dir)
            iter += 1
        return t,x,rxns

    def add_two_step_concerted_reactions_on_the_fly_save_intermediates(self, initial_conc, time_span, barrier_uni, barrier_bi, xyz_dir,iterations=5,remove_Li_red=False):
        self.searched_species_2_step = []
        iter = 0
        t, x, rxns = 0,0,0
        print("current nums of reactions at iter {}:".format(iter), len(self.unique_reaction_nodes))
        while iter < iterations:
            self.get_rates(barrier_uni, barrier_bi)
            self.remove_gas_reactions(xyz_dir)
            if remove_Li_red:
                self.remove_Li_reduction_reaction(xyz_dir)
            t, x, rxns = self.direct_method_no_record(initial_conc, time_span)

            EC_mg = MoleculeGraph.with_local_env_strategy(
                Molecule.from_file(os.path.join(xyz_dir, "EC.xyz")),
                OpenBabelNN(),
                reorder=False,
                extend_structure=False)
            EC_mg = metal_edge_extender(EC_mg)

            EC_ind = None
            for entry in self.reaction_network.entries["C3 H4 O3"][10][0]:
                if EC_mg.isomorphic_to(entry.mol_graph):
                    EC_ind = entry.parameters["ind"]
                    break

            Li1_ind = self.reaction_network.entries["Li1"][0][1][0].parameters["ind"]

            np.save('x_iter_{}'.format(iter), x)
            np.save('t_iter_{}'.format(iter), t)
            np.save('rxns_iter_{}'.format(iter), rxns)

            sorted_species_index = np.argsort(x[-1, :])[::-1]
            fig, ax = plt.subplots()
            for i in range(100):
                species_index = sorted_species_index[i]
                if x[-1, int(species_index)] > 0 and int(species_index) != EC_ind and int(
                        species_index) != Li1_ind and int(
                        species_index) != self.num_species - 1:
                    ax.step(t, x[:, int(species_index)], where='mid', label=str(species_index))
            plt.title('KMC concerted iter {}'.format(iter))
            plt.legend(loc='upper left')
            plt.savefig('concerted_iter_{}.png'.format(iter))

            rxns_set = list(set(rxns))
            rxns_count = [list(rxns).count(rxn) for rxn in rxns_set]
            index = np.argsort(rxns_count)[::-1]
            sorted_rxns = np.array(rxns_set)[index]
            x0 = np.arange(len(rxns_set))

            fig, ax = plt.subplots()
            plt.bar(x0, rxns_count)
            plt.title('reaction decomposition concerted iter {}'.format(iter))
            plt.savefig('reaction_decomp_concerted_iter_{}.png'.format(iter))
            for rxn in sorted_rxns:
                rxn = int(rxn)
                print(self.unique_reaction_nodes[rxn], self.reaction_rates[rxn])

            with open("unique_reaction_nodes_iter_{}.txt".format(iter), "wb") as fp:
                pickle.dump(self.unique_reaction_nodes, fp)
            with open("reaction_rates_iter_{}.txt".format(iter), "wb") as fp:
                pickle.dump(self.reaction_rates, fp)

            print('num of species:', self.num_species)

            if not iterations - iter == 1:
                self.add_concerted_reactions_2_step(x[-1,:], 10, terminate=True)
                print("current nums of reactions at iter {}:".format(iter+1), len(self.unique_reaction_nodes))
                self.get_rates(barrier_uni, barrier_bi)
                self.remove_gas_reactions(xyz_dir)
                if remove_Li_red:
                    self.remove_Li_reduction_reaction(xyz_dir)
            iter += 1
        return t,x,rxns

    def add_three_step_concerted_reactions_on_the_fly_save_intermediates(self, initial_conc, time_span, barrier_uni, barrier_bi, xyz_dir,iterations=5,remove_Li_red=False):
        self.searched_species_2_step = []
        self.searched_species_3_step = []
        iter = 0
        t, x, rxns = 0,0,0
        print("current nums of reactions at iter {}:".format(iter), len(self.unique_reaction_nodes))
        while iter < iterations:
            t, x, rxns = self.direct_method_no_record(initial_conc, time_span)

            EC_mg = MoleculeGraph.with_local_env_strategy(
                Molecule.from_file(os.path.join(xyz_dir, "EC.xyz")),
                OpenBabelNN(),
                reorder=False,
                extend_structure=False)
            EC_mg = metal_edge_extender(EC_mg)

            EC_ind = None
            for entry in self.reaction_network.entries["C3 H4 O3"][10][0]:
                if EC_mg.isomorphic_to(entry.mol_graph):
                    EC_ind = entry.parameters["ind"]
                    break

            Li1_ind = self.reaction_network.entries["Li1"][0][1][0].parameters["ind"]

            np.save('x_iter_{}'.format(iter), x)
            np.save('t_iter_{}'.format(iter), t)
            np.save('rxns_iter_{}'.format(iter), rxns)

            sorted_species_index = np.argsort(x[-1, :])[::-1]
            fig, ax = plt.subplots()
            for i in range(100):
                species_index = sorted_species_index[i]
                if x[-1, int(species_index)] > 0 and int(species_index) != EC_ind and int(
                        species_index) != Li1_ind and int(
                        species_index) != self.num_species - 1:
                    ax.step(t, x[:, int(species_index)], where='mid', label=str(species_index))
            plt.title('KMC concerted iter {}'.format(iter))
            plt.legend(loc='upper left')
            plt.savefig('concerted_iter_{}.png'.format(iter))

            rxns_set = list(set(rxns))
            rxns_count = [list(rxns).count(rxn) for rxn in rxns_set]
            index = np.argsort(rxns_count)[::-1]
            sorted_rxns = np.array(rxns_set)[index]
            x0 = np.arange(len(rxns_set))

            fig, ax = plt.subplots()
            plt.bar(x0, rxns_count)
            plt.title('reaction decomposition concerted iter {}'.format(iter))
            plt.savefig('reaction_decomp_concerted_iter_{}.png'.format(iter))
            for rxn in sorted_rxns:
                rxn = int(rxn)
                print(self.unique_reaction_nodes[rxn], self.reaction_rates[rxn])

            with open("unique_reaction_nodes_iter_{}.txt".format(iter), "wb") as fp:
                pickle.dump(self.unique_reaction_nodes, fp)
            with open("reaction_rates_iter_{}.txt".format(iter), "wb") as fp:
                pickle.dump(self.reaction_rates, fp)

            print('num of species:', self.num_species)

            if not iterations - iter == 1:
                self.add_concerted_reactions_2_step(x[-1,:], 10, terminate=False)
                self.add_concerted_reactions_3_step(x[-1, :], 10, terminate=True)
                print("current nums of reactions at iter {}:".format(iter+1), len(self.unique_reaction_nodes))
                self.get_rates(barrier_uni, barrier_bi)
                self.remove_gas_reactions(xyz_dir)
                if remove_Li_red:
                    self.remove_Li_reduction_reaction(xyz_dir)
            iter += 1
        return t,x,rxns

    def add_three_step_concerted_reactions_on_the_fly_save_intermediates_new(self, initial_conc, time_span, barrier_uni, barrier_bi, xyz_dir,iterations=5,remove_Li_red=False):
        self.searched_species_2_step = []
        self.searched_species_3_step = []
        iter = 0
        t, x, rxns = 0,0,0
        print("current nums of reactions at iter {}:".format(iter), len(self.unique_reaction_nodes))
        while iter < iterations:
            t, x, rxns = self.direct_method_no_record(initial_conc, time_span)

            EC_mg = MoleculeGraph.with_local_env_strategy(
                Molecule.from_file(os.path.join(xyz_dir, "EC.xyz")),
                OpenBabelNN(),
                reorder=False,
                extend_structure=False)
            EC_mg = metal_edge_extender(EC_mg)

            EC_ind = None
            for entry in self.reaction_network.entries["C3 H4 O3"][10][0]:
                if EC_mg.isomorphic_to(entry.mol_graph):
                    EC_ind = entry.parameters["ind"]
                    break

            Li1_ind = self.reaction_network.entries["Li1"][0][1][0].parameters["ind"]

            np.save('x_iter_{}'.format(iter), x)
            np.save('t_iter_{}'.format(iter), t)
            np.save('rxns_iter_{}'.format(iter), rxns)

            sorted_species_index = np.argsort(x[-1, :])[::-1]
            fig, ax = plt.subplots()
            for i in range(100):
                species_index = sorted_species_index[i]
                if x[-1, int(species_index)] > 0 and int(species_index) != EC_ind and int(
                        species_index) != Li1_ind and int(
                        species_index) != self.num_species - 1:
                    ax.step(t, x[:, int(species_index)], where='mid', label=str(species_index))
            plt.title('KMC concerted iter {}'.format(iter))
            plt.legend(loc='upper left')
            plt.savefig('concerted_iter_{}.png'.format(iter))

            rxns_set = list(set(rxns))
            rxns_count = [list(rxns).count(rxn) for rxn in rxns_set]
            index = np.argsort(rxns_count)[::-1]
            sorted_rxns = np.array(rxns_set)[index]
            x0 = np.arange(len(rxns_set))

            fig, ax = plt.subplots()
            plt.bar(x0, rxns_count)
            plt.title('reaction decomposition concerted iter {}'.format(iter))
            plt.savefig('reaction_decomp_concerted_iter_{}.png'.format(iter))
            for rxn in sorted_rxns:
                rxn = int(rxn)
                print(self.unique_reaction_nodes[rxn], self.reaction_rates[rxn])

            with open("unique_reaction_nodes_iter_{}.txt".format(iter), "wb") as fp:
                pickle.dump(self.unique_reaction_nodes, fp)
            with open("reaction_rates_iter_{}.txt".format(iter), "wb") as fp:
                pickle.dump(self.reaction_rates, fp)

            print('num of species:', self.num_species)

            if not iterations - iter == 1:
                self.add_concerted_reactions_2_step_new(x[-1,:], 10, terminate=False)
                self.add_concerted_reactions_3_step_new(x[-1, :], 10, terminate=True)
                print("current nums of reactions at iter {}:".format(iter+1), len(self.unique_reaction_nodes))
                self.get_rates(barrier_uni, barrier_bi)
                self.remove_gas_reactions(xyz_dir)
                if remove_Li_red:
                    self.remove_Li_reduction_reaction(xyz_dir)
            iter += 1
        return t,x,rxns

    def add_four_step_concerted_reactions_on_the_fly_save_intermediates(self, initial_conc, time_span, barrier_uni, barrier_bi, xyz_dir,iterations=5,remove_Li_red=False):
        self.searched_species_2_step = []
        self.searched_species_3_step = []
        self.searched_species_4_step = []
        iter = 0
        t, x, rxns = 0,0,0
        print("current nums of reactions at iter {}:".format(iter), len(self.unique_reaction_nodes))
        while iter < iterations:
            t, x, rxns = self.direct_method_no_record(initial_conc, time_span)

            EC_mg = MoleculeGraph.with_local_env_strategy(
                Molecule.from_file(os.path.join(xyz_dir, "EC.xyz")),
                OpenBabelNN(),
                reorder=False,
                extend_structure=False)
            EC_mg = metal_edge_extender(EC_mg)

            EC_ind = None
            for entry in self.reaction_network.entries["C3 H4 O3"][10][0]:
                if EC_mg.isomorphic_to(entry.mol_graph):
                    EC_ind = entry.parameters["ind"]
                    break

            Li1_ind = self.reaction_network.entries["Li1"][0][1][0].parameters["ind"]

            np.save('x_iter_{}'.format(iter), x)
            np.save('t_iter_{}'.format(iter), t)
            np.save('rxns_iter_{}'.format(iter), rxns)

            sorted_species_index = np.argsort(x[-1, :])[::-1]
            fig, ax = plt.subplots()
            for i in range(100):
                species_index = sorted_species_index[i]
                if x[-1, int(species_index)] > 0 and int(species_index) != EC_ind and int(
                        species_index) != Li1_ind and int(
                        species_index) != self.num_species - 1:
                    ax.step(t, x[:, int(species_index)], where='mid', label=str(species_index))
            plt.title('KMC concerted iter {}'.format(iter))
            plt.legend(loc='upper left')
            plt.savefig('concerted_iter_{}.png'.format(iter))

            rxns_set = list(set(rxns))
            rxns_count = [list(rxns).count(rxn) for rxn in rxns_set]
            index = np.argsort(rxns_count)[::-1]
            sorted_rxns = np.array(rxns_set)[index]
            x0 = np.arange(len(rxns_set))

            fig, ax = plt.subplots()
            plt.bar(x0, rxns_count)
            plt.title('reaction decomposition concerted iter {}'.format(iter))
            plt.savefig('reaction_decomp_concerted_iter_{}.png'.format(iter))
            for rxn in sorted_rxns:
                rxn = int(rxn)
                print(self.unique_reaction_nodes[rxn], self.reaction_rates[rxn])

            with open("unique_reaction_nodes_iter_{}.txt".format(iter), "wb") as fp:
                pickle.dump(self.unique_reaction_nodes, fp)
            with open("reaction_rates_iter_{}.txt".format(iter), "wb") as fp:
                pickle.dump(self.reaction_rates, fp)

            print('num of species:', self.num_species)

            if not iterations - iter == 1:
                self.add_concerted_reactions_2_step(x[-1,:], 10, terminate=False)
                self.add_concerted_reactions_3_step(x[-1, :], 10, terminate=False)
                self.add_concerted_reactions_4_step(x[-1, :], 10, terminate=True)

                print("current nums of reactions at iter {}:".format(iter+1), len(self.unique_reaction_nodes))
                self.get_rates(barrier_uni, barrier_bi)
                self.remove_gas_reactions(xyz_dir)
                if remove_Li_red:
                    self.remove_Li_reduction_reaction(xyz_dir)
            iter += 1
        return t,x,rxns


if __name__ == '__main__':
    prod_entries = []
    entries = loadfn("/Users/xiaowei_xie/pymatgen/pymatgen/analysis/reaction_network/smd_production_entries.json")
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

    test_dir = '/Users/xiaowei_xie/Desktop/Sam_production/xyzs/'
    EC_mg = MoleculeGraph.with_local_env_strategy(
        Molecule.from_file(os.path.join(test_dir, "EC.xyz")),
        OpenBabelNN(),
        reorder=False,
        extend_structure=False)
    EC_mg = metal_edge_extender(EC_mg)

    H2O_mg = MoleculeGraph.with_local_env_strategy(
        Molecule.from_file(os.path.join(test_dir, "water.xyz")),
        OpenBabelNN(),
        reorder=False,
        extend_structure=False)
    H2O_mg = metal_edge_extender(H2O_mg)
    EC_ind = None
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
    SS.get_rates(1.0841025975148306,1.3009231170177968)
    SS.remove_gas_reactions('/Users/xiaowei_xie/Desktop/Sam_production/xyzs/')
    xyz_dir = '/Users/xiaowei_xie/Desktop/Sam_production/xyzs/'

    Li_mg = MoleculeGraph.with_local_env_strategy(
        Molecule.from_file(os.path.join(xyz_dir, "Li.xyz")),
        OpenBabelNN(),
        reorder=False,
        extend_structure=False)
    for entry in SS.reaction_network.entries['Li1'][0][1]:
        if Li_mg.isomorphic_to(entry.mol_graph):
            Li1_ind = entry.parameters["ind"]
            break
    for entry in SS.reaction_network.entries['Li1'][0][0]:
        if Li_mg.isomorphic_to(entry.mol_graph):
            Li0_ind = entry.parameters["ind"]
            break
    for i, rxn_node in enumerate(SS.unique_reaction_nodes):
        if rxn_node == '{},{}'.format(str(Li1_ind), str(Li0_ind)):
            print('found')
            SS.reaction_rates[i] = 0.0

    t, x, rxns = SS.direct_method_no_record(initial_conc, 10000)



    t, x, rxns = SS.add_four_step_concerted_reactions_on_the_fly_save_intermediates(initial_conc, 10000,
                                                                1.0841025975148306, 1.3009231170177968, xyz_dir,
                                                                iterations=2)
    '''
    
    t, x, rxns = SS.add_concerted_reactions_on_the_fly_save_intermediates(initial_conc, 1000,
                                                                1.0841025975148306, 1.3009231170177968, xyz_dir,
                                                                iterations=3)

    
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
    plt.savefig('concerted_iter_0.png')

    rxns_set = list(set(rxns))
    rxns_count = [list(rxns).count(rxn) for rxn in rxns_set]
    index = np.argsort(rxns_count)[::-1]
    sorted_rxns = np.array(rxns_set)[index]
    x0 = np.arange(len(rxns_set))
    fig, ax = plt.subplots()
    plt.bar(x0, rxns_count)
    # plt.xticks(x, ([str(int(rxn)) for rxn in rxns_set]))
    plt.title('reaction decomposition')
    plt.savefig('reaction_decomp_concerted_iter_0.png')
    for rxn in sorted_rxns:
        rxn = int(rxn)
        print(SS.unique_reaction_nodes[rxn], SS.reaction_rates[rxn])
    '''

    '''
    t, x, rxns, records = SS.direct_method(initial_conc,1000000,10000000)

    sorted_species_index = np.argsort(x[-1,:])[::-1]
    fig, ax = plt.subplots()
    for i in range(100):
        species_index = sorted_species_index[i]
        if x[-1,int(species_index)] > 0 and int(species_index) != EC_ind and int(species_index) != Li1_ind and int(species_index) != SS.num_species-1:
            ax.step(t,x[:,int(species_index)],where='mid', label=str(species_index))
            #ax.plot(T,X[:,int(species_index)], 'C0o', alpha=0.5)
    plt.title('test')
    plt.legend(loc='upper left')
    plt.show()

    rxns_set = list(set(rxns))
    rxns_count = [list(rxns).count(rxn) for rxn in rxns_set]
    index = np.argsort(rxns_count)[::-1]
    sorted_rxns = np.array(rxns_set)[index]
    x0 = np.arange(len(rxns_set))
    fig, ax = plt.subplots()
    plt.bar(x0, rxns_count)
    # plt.xticks(x, ([str(int(rxn)) for rxn in rxns_set]))
    plt.show()
    for rxn in sorted_rxns:
        rxn = int(rxn)
        print(SS.unique_reaction_nodes[rxn], SS.reaction_rates[rxn])
    '''

    '''
    for i in range(len(RN.entries_list)):
        mol = RN.entries_list[i].molecule
        charge = mol.charge
        mol.to('xyz','/Users/xiaowei_xie/pymatgen/pymatgen/analysis/reaction_network/mols_no_metal_edge_extender/'+str(i)+'_'+str(charge)+'.xyz')
    '''


    '''
    reaction_energies = []
    for rxn_node in SS.reaction_nodes:
        reaction_energy = SS.graph.node[rxn_node]["free_energy"]
        reaction_energies.append(reaction_energy)'''
    '''
    max_output_length = 1000000
    t = np.zeros(max_output_length)
    x = np.zeros([max_output_length, SS.num_species])
    rxns = np.zeros(max_output_length)
    t[0] = 0
    x[0, :] = initial_conc
    rxn_count = 0

    a = SS.get_propensities(x[rxn_count, :])
    a0 = np.sum(a)
    random_t = random.uniform(0, 1)
    random_rxn = random.uniform(0, 1)
    tau = -np.log(random_t) / a0
    mu = np.where(np.cumsum(a) >= random_rxn * a0)[0][0]
    
    path = '/Users/xiaowei_xie/pymatgen/pymatgen/analysis/reaction_network/test_mols/'
    LEDC_entry = RN.entries['C4 H4 Li2 O6'][17][0][0]
    LEDC_entry.molecule.to('xyz',path+'LEDC.xyz')

    LiEC_RO_entry = RN.entries['C3 H4 Li1 O3'][10][0][0]
    LiEC_RO_entry.molecule.to('xyz',path+'LiEC_RO.xyz')

    C2H4_entry = RN.entries['C2 H4'][5][0][0]
    C2H4_entry.molecule.to('xyz', path + 'C2H4.xyz')

    LiCO3_minus_entry = RN.entries['C1 Li1 O3'][5][-1][0]
    LiCO3_minus_entry.molecule.to('xyz', path + 'LiCO3_minus.xyz')

    LiEC_plus_entry = RN.entries['C3 H4 Li1 O3'][11][1][0]
    LiEC_plus_entry.molecule.to('xyz', path + 'LiEC_plus.xyz')'''