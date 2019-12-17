import numpy as np
from pymatgen.analysis.graphs import MoleculeGraph, MolGraphSplitError
from pymatgen.analysis.local_env import OpenBabelNN
from pymatgen import Molecule
from pymatgen.analysis.fragmenter import metal_edge_extender
from pymatgen.entries.mol_entry import MoleculeEntry
from pymatgen.analysis.reaction_network.reaction_network import ReactionNetwork
from monty.serialization import dumpfn, loadfn
import os
from pymatgen.analysis.reaction_network.stochastic_simulation import StochaticSimulation
import matplotlib.pyplot as plt
import json
import pickle

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
SS.get_rates(1.0841025975148306, 1.3009231170177968)
xyz_dir = '/home/xiaowei/Sam_production/xyzs/'
SS.remove_gas_reactions(xyz_dir)

t, x, rxns = SS.add_two_step_concerted_reactions_on_the_fly_save_intermediates(initial_conc, 1000000,
                                                            1.0841025975148306, 1.3009231170177968, xyz_dir,
                                                            iterations=10, remove_Li_red=True)