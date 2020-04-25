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
EC_mg = metal_edge_extender(EC_mg)

LiEC_mg = MoleculeGraph.with_local_env_strategy(
    Molecule.from_file("/Users/xiaoweixie/Desktop/Sam_production/xyzs/LiEC_bi.xyz"),
    OpenBabelNN(),
    reorder=False,
    extend_structure=False)
LiEC_mg = metal_edge_extender(LiEC_mg)

LEMC_mg = MoleculeGraph.with_local_env_strategy(
    Molecule.from_file("/Users/xiaoweixie/Desktop/Sam_production/xyzs/LEMC.xyz"),
    OpenBabelNN(),
    reorder=False,
    extend_structure=False)
LEMC_mg = metal_edge_extender(LEMC_mg)

LEDC_mg = MoleculeGraph.with_local_env_strategy(
    Molecule.from_file("/Users/xiaoweixie/Desktop/Sam_production/xyzs/LEDC.xyz"),
    OpenBabelNN(),
    reorder=False,
    extend_structure=False)
LEDC_mg = metal_edge_extender(LEDC_mg)

H2O_mg =  MoleculeGraph.with_local_env_strategy(
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
for entry in RN.entries["C4 H4 Li2 O6"][17][0]:
    if LEDC_mg.isomorphic_to(entry.mol_graph):
        if entry.free_energy == -16910.7035955349:
            LEDC_ind = entry.parameters["ind"]
            break
for entry in RN.entries["C3 H5 Li1 O4"][13][0]:
    if LEMC_mg.isomorphic_to(entry.mol_graph):
        if entry.free_energy == -11587.839161760392:
            LEMC_ind = entry.parameters["ind"]
            break
for entry in RN.entries["C3 H4 Li1 O3"][12][0]:
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



# RN.find_concerted_candidates_equal('LEMC_test_2_multiprocess')
# RN.add_concerted_reactions_from_list(read_file=True, file_name='LEMC_test_2_multiprocess_equal')
#
# starts = [LiEC_ind]
# target = LEDC_ind
# weight = "softplus"
# max_iter=100
# RN.num_starts = len(starts)
# PR_paths = RN.solve_prerequisites(starts,target,weight)

'''
RN.PR_record = RN.build_PR_record()
RN.min_cost = {}
RN.num_starts = None

# LEDC_PR_paths, LEDC_paths = RN.find_paths([EC_ind,Li1_ind,H2O_ind],LEDC_ind,weight="softplus",num_paths=10)
# for path in LEDC_paths:
#     for val in path:
#         print(val, path[val])
#     print()

starts = [EC_ind,Li1_ind,H2O_ind]
target = LEDC_ind
weight = "softplus"
max_iter=100
RN.num_starts = len(starts)
#PR_paths = RN.solve_prerequisites(starts,target,weight)
'''
'''
PRs = {}
old_solved_PRs = []
new_solved_PRs = ["placeholder"]
orig_graph = copy.deepcopy(RN.graph)
old_attrs = {}
new_attrs = {}

for start in starts:
    PRs[start] = {}
for PR in PRs:
    for start in starts:
        if start == PR:
            PRs[PR][start] = RN.characterize_path([start], weight)
        else:
            PRs[PR][start] = "no_path"
    old_solved_PRs.append(PR)
    RN.min_cost[PR] = PRs[PR][PR]["cost"]
for node in RN.graph.nodes():
    if RN.graph.nodes[node]["bipartite"] == 0 and node != target:
        if node not in PRs:
            PRs[node] = {}
            
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

relevant_nodes = []
for node in RN.graph.nodes():
    if RN.graph.nodes[node]["bipartite"] == 0 and node not in old_solved_PRs and node != target:
        relevant_nodes.append(node)


for node in RN.graph.nodes():
    if RN.graph.nodes[node]["bipartite"] == 0 and node not in old_solved_PRs and node != target:
        print('node:', node)
        for start in starts:
            if start not in PRs[node]:
                path_exists = True
                try:
                    length, dij_path = nx.algorithms.simple_paths._bidirectional_dijkstra(
                        RN.graph,
                        source=hash(start),
                        target=hash(node),
                        ignore_nodes=RN.find_or_remove_bad_nodes([target, node]),
                        weight=weight)
                except nx.exception.NetworkXNoPath:
                    PRs[node][start] = "no_path"
                    path_exists = False
                    cost_from_start[node][start] = "no_path"
                if path_exists:
                    if len(dij_path) > 1 and len(dij_path) % 2 == 1:
                        path = RN.characterize_path(dij_path, weight, old_solved_PRs)
                        print(path)
                        cost_from_start[node][start] = path["cost"]
                        if len(path["unsolved_prereqs"]) == 0:
                            PRs[node][start] = path
                            # print("Solved PR",node,PRs[node])
                        if path["cost"] < min_cost[node]:
                            min_cost[node] = path["cost"]
                    else:
                        print("Does this ever happen?")
'''
'''
solved_PRs = copy.deepcopy(old_solved_PRs)
new_solved_PRs = []
for PR in PRs:
    if PR not in solved_PRs:
        if len(PRs[PR].keys()) == RN.num_starts:
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
                if num_beaten == RN.num_starts - 1:
                    solved_PRs.append(PR)
                    new_solved_PRs.append(PR)

# new_solved_PRs = []
# for PR in solved_PRs:
#     if PR not in old_solved_PRs:
#         new_solved_PRs.append(PR)

print(ii, len(old_solved_PRs), len(new_solved_PRs))
attrs = {}

for PR_ind in min_cost:
    for rxn_node in RN.PR_record[PR_ind]:
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
nx.set_edge_attributes(RN.graph, attrs)
RN.min_cost = copy.deepcopy(min_cost)
old_solved_PRs = copy.deepcopy(solved_PRs)
ii += 1
old_attrs = copy.deepcopy(new_attrs)
new_attrs = copy.deepcopy(attrs)
'''

'''
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
    for node in RN.graph.nodes():
        if RN.graph.nodes[node]["bipartite"] == 0 and node not in old_solved_PRs and node != target:
            for start in starts:
                if start not in PRs[node]:
                    path_exists = True
                    try:
                        length, dij_path = nx.algorithms.simple_paths._bidirectional_dijkstra(
                            RN.graph,
                            source=hash(start),
                            target=hash(node),
                            ignore_nodes=RN.find_or_remove_bad_nodes([target, node]),
                            weight=weight)
                    except nx.exception.NetworkXNoPath:
                        PRs[node][start] = "no_path"
                        path_exists = False
                        cost_from_start[node][start] = "no_path"
                    if path_exists:
                        if len(dij_path) > 1 and len(dij_path) % 2 == 1:
                            path = RN.characterize_path(dij_path, weight, old_solved_PRs)
                            print(path)
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
            if len(PRs[PR].keys()) == RN.num_starts:
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
                    if num_beaten == RN.num_starts - 1:
                        solved_PRs.append(PR)
                        new_solved_PRs.append(PR)

    # new_solved_PRs = []
    # for PR in solved_PRs:
    #     if PR not in old_solved_PRs:
    #         new_solved_PRs.append(PR)

    print(ii, len(old_solved_PRs), len(new_solved_PRs))
    attrs = {}

    for PR_ind in min_cost:
        for rxn_node in RN.PR_record[PR_ind]:
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
    nx.set_edge_attributes(RN.graph, attrs)
    RN.min_cost = copy.deepcopy(min_cost)
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
#                 path_dict = RN.characterize_path(PRs[PR][start]["path"],weight,PRs,True)
#                 if abs(path_dict["cost"]-path_dict["pure_cost"])>0.0001:
#                     print("WARNING: cost mismatch for PR",PR,path_dict["cost"],path_dict["pure_cost"],path_dict["full_path"])
#         if not path_found:
#             print("No path found from any start to PR",PR)
#     else:
#         print("Unsolvable path from any start to PR",PR)
'''
'''
#RN.find_concerted_general_2('LEMC_test_2')
RN.add_concerted_reactions_2(read_file=True, file_name='LEMC_test_2', num=143)

num_concerted_rxns = 0
for node in RN.graph.nodes.data()._nodes.keys():
    if RN.graph.nodes[node]['bipartite'] == 1:
        if RN.graph.nodes[node]['rxn_type'] == "concerted":
            num_concerted_rxns += 1
            print(node)
#6226
#RN.solve_prerequisites([EC_ind,Li1_ind,H2O_ind],LEMC_ind,weight="softplus",max_iter=100)

LEDC_PR_paths, LEDC_paths = RN.find_paths([LiEC_ind],LEDC_ind,weight="softplus",num_paths=10)
for path in LEDC_paths:
    for val in path:
        print(val, path[val])
    print()

LEMC_PR_paths, LEMC_paths = RN.find_paths([LiEC_ind,H2O_ind],LEMC_ind,weight="softplus",num_paths=10)
for path in LEMC_paths:
    for val in path:
        print(val, path[val])
    print()

PRs = RN.solve_prerequisites([LiEC_ind],LEDC_ind,weight="softplus")

for key in RN.reac_prod_dict:
    reactants = RN.reac_prod_dict[key]['reactants']
    if '12_12' in reactants:
        print(key)

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

# for i in range(len(RN.unique_mol_graphs_new)):
#     mol_graph = RN.unique_mol_graphs_new[i]
#     mol_graph.molecule.to('xyz','/Users/xiaoweixie/pymatgen/pymatgen/analysis/reaction_network/mgcf/LEMC_small_network/unique_mol_graphs_new/'+str(i)+'.xyz')
'''

# RN.find_concerted_candidates_equal('LEMC_test_2_multiprocess')
# RN.multiprocess_equal('LEMC_test_2_multiprocess',2)


'''
starts = [EC_ind,Li1_ind,H2O_ind]
target = LEDC_ind
weight = "softplus"
max_iter=100
RN.num_starts = len(starts)



LEMC_PR_paths, LEMC_paths = RN.find_paths([EC_ind,Li1_ind,H2O_ind],LEMC_ind,weight="softplus",num_paths=10)
for path in LEMC_paths:
    for val in path:
        print(val, path[val])
    print()
'''
'''
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
'''
'''
RN.valid_reactions_dict = {}
for i, key in enumerate(list(RN.reac_prod_dict.keys())[3:4]):
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

RN.add_concerted_reactions_2()

num_concerted_rxns = 0
for node in RN.graph.nodes.data()._nodes.keys():
    if RN.graph.nodes[node]['bipartite'] == 1:
        if RN.graph.nodes[node]['rxn_type'] == "concerted":
            num_concerted_rxns += 1
            print(node)

num_nonconcerted_rxns = 0
for node in RN.graph.nodes.data()._nodes.keys():
    if RN.graph.nodes[node]['bipartite'] == 1:
        if RN.graph.nodes[node]['rxn_type'] != "concerted":
            num_nonconcerted_rxns += 1
            print(node)

for key in RN.reac_prod_dict.keys():
    num = len(RN.reac_prod_dict[key]['reactants'])
    print(key, num)'''