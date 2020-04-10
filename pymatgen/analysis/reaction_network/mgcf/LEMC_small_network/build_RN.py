import os
import unittest
import time
import copy
from pymatgen.core.structure import Molecule
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import OpenBabelNN
from pymatgen.util.testing import PymatgenTest
from pymatgen.analysis.reaction_network.reaction_network import ReactionNetwork
from pymatgen.analysis.fragmenter import metal_edge_extender
from pymatgen.entries.mol_entry import MoleculeEntry
from monty.serialization import loadfn, dumpfn
import openbabel as ob
import networkx as nx
from pymatgen.analysis.reaction_network.extract_reactions import *

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
dumpfn(RN,"/Users/xiaoweixie/pymatgen/pymatgen/analysis/reaction_network/mgcf/LEMC_small_network/LEMC_small_RN.json")


EC_mg = MoleculeGraph.with_local_env_strategy(
    Molecule.from_file("/Users/xiaoweixie/Desktop/Sam_production/xyzs/EC.xyz"),
    OpenBabelNN(),
    reorder=False,
    extend_structure=False)
#EC_mg = metal_edge_extender(EC_mg)

LiEC_mg = MoleculeGraph.with_local_env_strategy(
    Molecule.from_file("/Users/xiaoweixie/Desktop/Sam_production/xyzs/LiEC_bi.xyz"),
    OpenBabelNN(),
    reorder=False,
    extend_structure=False)
#LiEC_mg = metal_edge_extender(LiEC_mg)

LEMC_mg = MoleculeGraph.with_local_env_strategy(
    Molecule.from_file("/Users/xiaoweixie/Desktop/Sam_production/xyzs/LEMC.xyz"),
    OpenBabelNN(),
    reorder=False,
    extend_structure=False)
#LEMC_mg = metal_edge_extender(LEMC_mg)

LEDC_mg = MoleculeGraph.with_local_env_strategy(
    Molecule.from_file("/Users/xiaoweixie/Desktop/Sam_production/xyzs/LEDC.xyz"),
    OpenBabelNN(),
    reorder=False,
    extend_structure=False)
#LEDC_mg = metal_edge_extender(LEDC_mg)

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

RN.find_concerted_general_2('LEMC_test_2')