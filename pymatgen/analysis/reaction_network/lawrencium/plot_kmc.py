import json
import numpy as np
import matplotlib.pyplot as plt
from pymatgen.analysis.graphs import MoleculeGraph, MolGraphSplitError
from pymatgen.analysis.local_env import OpenBabelNN
from pymatgen import Molecule
from pymatgen.analysis.fragmenter import metal_edge_extender
from pymatgen.entries.mol_entry import MoleculeEntry
from pymatgen.analysis.reaction_network.reaction_network import ReactionNetwork
from monty.serialization import dumpfn, loadfn
import os
from pymatgen.analysis.reaction_network.stochastic_simulation import StochaticSimulation

x = np.load('/Users/xiaowei_xie/pymatgen/pymatgen/analysis/reaction_network/lawrencium/x_iter_0.npy')
t = np.load('/Users/xiaowei_xie/pymatgen/pymatgen/analysis/reaction_network/lawrencium/t_iter_0.npy')
rxns = np.load('/Users/xiaowei_xie/pymatgen/pymatgen/analysis/reaction_network/lawrencium/rxns_iter_0.npy')
#with open('/Users/xiaowei_xie/pymatgen/pymatgen/analysis/reaction_network/lawrencium/records_iter_0.json') as data_file:
#    record = json.load(data_file)

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

test_dir = '/Users/xiaowei_xie/Sam_production/xyzs/'
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

sorted_species_index = np.argsort(x[-1, :])[::-1]
fig, ax = plt.subplots()
for i in range(100):
    species_index = sorted_species_index[i]
    if x[-1, int(species_index)] > 0 and int(species_index) != EC_ind and int(species_index) != Li1_ind and int(
            species_index) != SS.num_species - 1:
        ax.step(t, x[:, int(species_index)], where='mid', label=str(species_index))
        # ax.plot(T,X[:,int(species_index)], 'C0o', alpha=0.5)
plt.title('KMC concerted iter 0')
plt.legend(loc='upper left')
#plt.savefig('concerted_iter_1.png')

rxns_set = list(set(rxns))
rxns_count = [list(rxns).count(rxn) for rxn in rxns_set]
index = np.argsort(rxns_count)[::-1]
sorted_rxns = np.array(rxns_set)[index]
x0 = np.arange(len(rxns_set))
fig, ax = plt.subplots()
plt.bar(x0, rxns_count)
# plt.xticks(x, ([str(int(rxn)) for rxn in rxns_set]))
plt.title('reaction decomposition concerted iter 0')
#plt.savefig('reaction_decomp_concerted_iter_1.png')

