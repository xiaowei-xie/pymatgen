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

RN.unique_mol_graphs = []
for entry in RN.entries_list:
    mol_graph = entry.mol_graph
    RN.unique_mol_graphs.append(mol_graph)

RN.unique_mol_graphs_new = []
# For duplicate mol graphs, create a map between later species with former ones
RN.unique_mol_graph_dict = {}

for i in range(len(RN.unique_mol_graphs)):
    mol_graph = RN.unique_mol_graphs[i]
    found = False
    for j in range(len(RN.unique_mol_graphs_new)):
        new_mol_graph = RN.unique_mol_graphs_new[j]
        if mol_graph.isomorphic_to(new_mol_graph):
            found = True
            RN.unique_mol_graph_dict[i] = j
            continue
    if not found:
        RN.unique_mol_graph_dict[i] = len(RN.unique_mol_graphs_new)
        RN.unique_mol_graphs_new.append(mol_graph)
dumpfn(RN.unique_mol_graph_dict, "LEMC_test_2" + "_unique_mol_graph_map.json")
# find all molecule pairs that satisfy the stoichiometry constraint
RN.stoi_list, RN.species_same_stoi_dict = identify_same_stoi_mol_pairs(RN.unique_mol_graphs_new)
RN.reac_prod_dict = {}
for i, key in enumerate(RN.species_same_stoi_dict.keys()):
    species_list = RN.species_same_stoi_dict[key]
    new_species_list_reactant = []
    new_species_list_product = []
    for species in species_list:
        new_species_list_reactant.append(species)
        new_species_list_product.append(species)
    if new_species_list_reactant != [] and new_species_list_product != []:
        RN.reac_prod_dict[key] = {'reactants': new_species_list_reactant, 'products': new_species_list_product}

RN.valid_reactions_dict = {}
for i, key in enumerate(list(RN.reac_prod_dict.keys())):
    print(key)
    RN.valid_reactions_dict[key] = []
    reactants = RN.reac_prod_dict[key]['reactants']
    products = RN.reac_prod_dict[key]['products']
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
                    mol_graph1 = RN.unique_mol_graphs[int(split_reac[0])]
                    mol_graph2 = RN.unique_mol_graphs[int(split_prod[0])]
                    if identify_self_reactions(mol_graph1, mol_graph2):
                        if [reac, prod] not in RN.valid_reactions_dict[key]:
                            RN.valid_reactions_dict[key].append([reac, prod])
                elif (len(split_reac) == 2 and len(split_prod) == 1):
                    assert split_prod[0] not in split_reac
                    mol_graphs1 = [RN.unique_mol_graphs[int(split_reac[0])],
                                   RN.unique_mol_graphs[int(split_reac[1])]]
                    mol_graphs2 = [RN.unique_mol_graphs[int(split_prod[0])]]
                    if identify_reactions_AB_C(mol_graphs1, mol_graphs2):
                        if [reac, prod] not in RN.valid_reactions_dict[key]:
                            RN.valid_reactions_dict[key].append([reac, prod])
                elif (len(split_reac) == 1 and len(split_prod) == 2):
                    mol_graphs1 = [RN.unique_mol_graphs[int(split_prod[0])],
                                   RN.unique_mol_graphs[int(split_prod[1])]]
                    mol_graphs2 = [RN.unique_mol_graphs[int(split_reac[0])]]
                    if identify_reactions_AB_C(mol_graphs1, mol_graphs2):
                        if [reac, prod] not in RN.valid_reactions_dict[key]:
                            RN.valid_reactions_dict[key].append([reac, prod])
                elif (len(split_reac) == 2 and len(split_prod) == 2):
                    # RN reaction
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
                        mol_graph1 = RN.unique_mol_graphs[int(new_split_reac)]
                        mol_graph2 = RN.unique_mol_graphs[int(new_split_prod)]
                        if identify_self_reactions(mol_graph1, mol_graph2):
                            if [new_split_reac, new_split_prod] not in RN.valid_reactions_dict[key]:
                                RN.valid_reactions_dict[key].append([new_split_reac, new_split_prod])
                    # A + B -> C + D
                    else:
                        mol_graphs1 = [RN.unique_mol_graphs[int(split_reac[0])],
                                       RN.unique_mol_graphs[int(split_reac[1])]]
                        mol_graphs2 = [RN.unique_mol_graphs[int(split_prod[0])],
                                       RN.unique_mol_graphs[int(split_prod[1])]]
                        if identify_reactions_AB_CD(mol_graphs1, mol_graphs2):
                            if [reac, prod] not in RN.valid_reactions_dict[key]:
                                RN.valid_reactions_dict[key].append([reac, prod])
    dumpfn(RN.valid_reactions_dict, "LEMC_test_2" + "_valid_concerted_rxns_" + str(key) + ".json")