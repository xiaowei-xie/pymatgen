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
import random
from simanneal import Annealer
from multiprocessing import cpu_count
from pathos.multiprocessing import ProcessingPool as Pool
import math
import time
from ase.units import Hartree, eV, kcal, mol


__author__ = "Xiaowei Xie"
__copyright__ = "Copyright 2019, The Materials Project"
__version__ = "1.0"
__maintainer__ = "Xiaowei Xie"
__email__ = "xxw940407@icloud.com"
__status__ = "Alpha"
__date__ = "11/18/19"

logger = logging.getLogger(__name__)
kb = 8.617333262145 * 1e-5

class SimulatedAnnealing(Annealer):
    """
    Class for simulated annealing from ReactionNetwork class
    Args:
        state: a dictionary with species indices as keys, showing the number of species for each key.
        E.g. {1: 1, 2: 2, 3: 0}.

    """
    def __init__(self, state, reaction_network):
        self.reaction_network = reaction_network
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
        # self.species_nodes = []
        # for node in self.reaction_network.graph.nodes():
        #     if self.reaction_network.graph.nodes[node]["bipartite"] == 0:
        #         self.species_nodes.append(node)
        self.fired_reactions = []
        super(SimulatedAnnealing, self).__init__(state)
        return

    def move(self):
        """Fire a reaction. Have to ensure the starting materials for a reaction to fire."""

        initial_energy = self.energy()
        self.possible_reactions_to_fire = []
        for i in range(len(self.reactions)):
            reaction = self.reactions[i]
            reactants, products = reaction[0], reaction[1]
            unique_reactants = list(set(reactants))
            enough_materials = True
            charge_change = np.sum([self.reaction_network.entries_list[int(prod)].charge for prod in products]) - \
                            np.sum([self.reaction_network.entries_list[int(reac)].charge for reac in reactants])
            for reac in unique_reactants:
                num_reactants = reactants.count(reac)
                if num_reactants > self.state[int(reac)]:
                    enough_materials = False
            if enough_materials and self.state['e'] > - charge_change:
                self.possible_reactions_to_fire.append(reaction)

        rxn_idx = random.randint(0, len(self.possible_reactions_to_fire)-1)
        rxn_to_fire = self.possible_reactions_to_fire[rxn_idx]
        self.fired_reactions.append(rxn_to_fire)
        reactants, products = rxn_to_fire[0], rxn_to_fire[1]
        unique_reactants, unique_products = list(set(reactants)), list(set(products))
        for reac in unique_reactants:
            num_reactants = reactants.count(reac)
            self.state[int(reac)] -= num_reactants
        for prod in unique_products:
            num_products = products.count(prod)
            self.state[int(prod)] += num_products
        charge_change = np.sum([self.reaction_network.entries_list[int(prod)].charge for prod in products]) - \
                        np.sum([self.reaction_network.entries_list[int(reac)].charge for reac in reactants])
        if charge_change != 0:
            self.state['e'] += charge_change
        return self.energy() - initial_energy

    def energy(self):
        """Calculates the length of the route."""
        e = 0
        for key in self.state.keys():
            if key == 'e':
                e += self.state[key] * self.reaction_network.electron_free_energy
            else:
                e += self.state[key] * self.reaction_network.entries_list[key].free_energy
        return e

    def anneal(self, num):
        """Minimizes the energy of a system by simulated annealing.
        Parameters
        state : an initial arrangement of the system
        Returns
        (state, energy): the best state and energy found.
        """
        step = 0
        self.start = time.time()

        # Precompute factor for exponential cooling from Tmax to Tmin
        if self.Tmin <= 0.0:
            raise Exception('Exponential cooling requires a minimum "\
                "temperature greater than zero.')
        Tfactor = -math.log(self.Tmax / self.Tmin)

        # Note initial state
        T = self.Tmax
        E = self.energy()
        prevState = self.copy_state(self.state)
        prevEnergy = E
        self.best_state = self.copy_state(self.state)
        self.best_energy = E
        trials, accepts, improves = 0, 0, 0
        if self.updates > 0:
            updateWavelength = self.steps / self.updates
            self.update(step, T, E, None, None)

        # Attempt moves to new states
        while step < self.steps and not self.user_exit:
            step += 1
            T = self.Tmax * math.exp(Tfactor * step / self.steps)
            dE = self.move()
            if dE is None:
                E = self.energy()
                dE = E - prevEnergy
            else:
                E += dE
            trials += 1
            if dE > 0.0 and math.exp(-dE /kb / T) < random.random():
                # Restore previous state
                self.state = self.copy_state(prevState)
                E = prevEnergy
            else:
                # Accept new state and compare to best state
                accepts += 1
                if dE < 0.0:
                    improves += 1
                prevState = self.copy_state(self.state)
                prevEnergy = E
                if E < self.best_energy:
                    self.best_state = self.copy_state(self.state)
                    self.best_energy = E
            if self.updates > 1:
                if (step // updateWavelength) > ((step - 1) // updateWavelength):
                    self.update(
                        step, T, E, accepts / trials, improves / trials)
                    trials, accepts, improves = 0, 0, 0

        self.state = self.copy_state(self.best_state)
        if self.save_state_on_exit:
            self.save_state()

        # Return best state and energy
        return self.best_state, self.best_energy, self.fired_reactions

def SA_multiprocess(SA, name, nums, num_processors):
    # name: filename to save as
    # nums: numbers of SA runs
    args = [(i) for i in np.arange(nums)]
    pool = Pool(num_processors)
    results = pool.map(SA.anneal, args)
    fired_reactions_all = []
    best_state_all = []
    best_energy_all = []
    for i in range(len(results)):
        best_state = results[i][0]
        best_energy = results[i][1]
        fired_reactions = results[i][2]
        fired_reactions_all.append(fired_reactions)
        best_state_all.append(best_state)
        best_energy_all.append(best_energy)
    dumpfn(best_energy_all, name + "_best_energy.json")
    dumpfn(best_state_all, name + "_best_state.json")
    dumpfn(fired_reactions_all, name + "_fired_reactions.json")

    return

if __name__ == "__main__":
    prod_entries = []
    entries = loadfn("/Users/xiaoweixie/PycharmProjects/electrolyte/LEMC/smd_production_entries_LEMC_20200409.json")
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
    EC_mg = MoleculeGraph.with_local_env_strategy(
        Molecule.from_file("/Users/xiaoweixie/Desktop/Sam_production/xyzs/EC.xyz"),
        OpenBabelNN(),
        reorder=False,
        extend_structure=False)
    # EC_mg = metal_edge_extender(EC_mg)

    LiEC_mg = MoleculeGraph.with_local_env_strategy(
        Molecule.from_file("/Users/xiaoweixie/Desktop/Sam_production/xyzs/LiEC_bi.xyz"),
        OpenBabelNN(),
        reorder=False,
        extend_structure=False)
    # LiEC_mg = metal_edge_extender(LiEC_mg)

    LEMC_mg = MoleculeGraph.with_local_env_strategy(
        Molecule.from_file("/Users/xiaoweixie/Desktop/Sam_production/xyzs/LEMC.xyz"),
        OpenBabelNN(),
        reorder=False,
        extend_structure=False)
    # LEMC_mg = metal_edge_extender(LEMC_mg)

    LEDC_mg = MoleculeGraph.with_local_env_strategy(
        Molecule.from_file("/Users/xiaoweixie/Desktop/Sam_production/xyzs/LEDC.xyz"),
        OpenBabelNN(),
        reorder=False,
        extend_structure=False)
    # LEDC_mg = metal_edge_extender(LEDC_mg)

    H2O_mg = MoleculeGraph.with_local_env_strategy(
        Molecule.from_file("/Users/xiaoweixie/Desktop/Sam_production/xyzs/water.xyz"),
        OpenBabelNN(),
        reorder=False,
        extend_structure=False)

    # LiCO3_minus_mg = MoleculeGraph.with_local_env_strategy(
    #     Molecule.from_file("/Users/xiaoweixie/Desktop/Sam_production/xyzs/LiCO3.xyz"),
    #     OpenBabelNN(),
    #     reorder=False,
    #     extend_structure=False)
    # LiCO3_minus_mg = metal_edge_extender(LiCO3_minus_mg)

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
    for entry in RN.entries["C4 H4 Li2 O6"][15][0]:
        if LEDC_mg.isomorphic_to(entry.mol_graph):
            if entry.free_energy == -16910.7035955349:
                LEDC_ind = entry.parameters["ind"]
                break
    for entry in RN.entries["C3 H5 Li1 O4"][12][0]:
        if LEMC_mg.isomorphic_to(entry.mol_graph):
            if entry.free_energy == -11587.839161760392:
                LEMC_ind = entry.parameters["ind"]
                break
    for entry in RN.entries["C3 H4 Li1 O3"][11][0]:
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
    #         if entry.free_energy == -7389.618831945432:
    #             LiCO3_minus_ind = entry.parameters["ind"]
    #             break

    Li1_ind = RN.entries["Li1"][0][1][0].parameters["ind"]

    print("EC_ind", EC_ind)
    print("LEDC_ind", LEDC_ind)
    print("LEMC_ind", LEMC_ind)
    print("Li1_ind", Li1_ind)
    print("LiEC_ind", LiEC_ind)
    print("LiCO3_minus_ind", LiCO3_minus_ind)

    RN.find_concerted_candidates_equal('LEMC_test_2_multiprocess')
    RN.add_concerted_reactions_from_list(read_file=True, file_name='mgcf/LEMC_small_network/LEMC_test_2_multiprocess_equal')

    species_nodes = []
    for node in RN.graph.nodes():
        if RN.graph.nodes[node]["bipartite"] == 0:
            species_nodes.append(node)
    species_nodes.append('e')

    initial_state = {30:10, 47:10, 45:10, 'e':10}
    state = {}
    for key in species_nodes:
        if key in initial_state.keys():
            state[key] = initial_state[key]
        else:
            state[key] = 0

    SA = SimulatedAnnealing(state, RN)
    schedule = {'tmax':2e5,'tmin':0.013, 'steps':1000, 'updates':100}
    #SA.set_schedule(SA.auto(minutes=0.2,steps=1000))
    SA.set_schedule(schedule)
    #SA_multiprocess(SA,'SA_multiprocess_test',2, 2)
    itinerary, miles, fired_reactions = SA.anneal(1)
    '''
    SA = SimulatedAnnealing(state, RN)
    SA.set_schedule(SA.auto(minutes=0.2))
    SA.copy_strategy = "deepcopy"

    itinerary, miles = SA.anneal()'''
# Start: -116012.02817005367
# 5 LEMC + 5 LEC + 5 H2O = -126278.82833202537
#Tmax 100.0
#Tmin 0.013
# steps 53
# updates 100
