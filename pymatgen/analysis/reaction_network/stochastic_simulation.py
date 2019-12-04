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
            # if self.graph.nodes[rxn_node]['rxn_type'] == 'LiCO3 -1 + LiEC 1 -> LEDC':
            #     barrier = eV/J*mol * barrier_uni
            #     rate = k_b * T / h * np.exp(-barrier / R / T)
            # elif self.graph.nodes[rxn_node]['rxn_type'] == '2LiEC-RO -> LEDC + C2H4':
            #     barrier = eV/J*mol * barrier_uni
            #     rate = k_b * T / h * np.exp(-barrier / R / T)
            #else:
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


if __name__ == '__main__':
    prod_entries = []
    entries = loadfn("/Users/xiaowei_xie/pymatgen/pymatgen/analysis/reaction_network/LiEC_reextended_entries.json")
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
    for entry in RN.entries["C3 H5 Li1 O4"][13][0]:
        if LEMC_mg.isomorphic_to(entry.mol_graph):
            LEMC_ind = entry.parameters["ind"]
            break
    for entry in RN.entries["H2 O1"][2][0]:
        if H2O_mg.isomorphic_to(entry.mol_graph):
            H2O_ind = entry.parameters["ind"]
            break

    for entry in RN.entries['C4 H4 Li2 O6'][15][0]:
        if LEDC_mg.isomorphic_to(entry.mol_graph):
            LEDC_ind = entry.parameters["ind"]
            break
    Li1_ind = RN.entries["Li1"][0][1][0].parameters["ind"]
    OHminus_ind = RN.entries["H1 O1"][1][-1][0].parameters["ind"]

    initial_conc = np.zeros(SS.num_species)
    initial_conc[EC_ind] = 15000
    initial_conc[Li1_ind] = 1000
    initial_conc[H2O_ind] = 30
    initial_conc[-1] = 1000
    SS.get_rates(1.0841025975148306,1.3009231170177968)
    SS.remove_gas_reactions()
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