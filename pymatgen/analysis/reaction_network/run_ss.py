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
from pymatgen.analysis.reaction_network.reaction_network import ReactionNetwork
from monty.serialization import dumpfn, loadfn
import random
import os
import matplotlib.pyplot as plt
from ase.units import eV, J, mol
from pymatgen.core.composition import CompositionError
from pymatgen.analysis.reaction_network.stochastic_simulation import StochaticSimulation

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
SS.get_rates(1.0841025975148306, 1.3009231170177968)
SS.remove_gas_reactions()
t, x, rxns, records = SS.direct_method(initial_conc, 1000000, 10000000)

sorted_species_index = np.argsort(x[-1, :])[::-1]
fig, ax = plt.subplots()
for i in range(100):
    species_index = sorted_species_index[i]
    if x[-1, int(species_index)] > 0 and int(species_index) != EC_ind and int(species_index) != Li1_ind and int(
            species_index) != SS.num_species - 1:
        ax.step(t, x[:, int(species_index)], where='mid', label=str(species_index))
        # ax.plot(T,X[:,int(species_index)], 'C0o', alpha=0.5)
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