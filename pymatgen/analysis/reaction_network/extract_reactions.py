from pymatgen.analysis.graphs import MoleculeGraph, MolGraphSplitError
from atomate.qchem.database import QChemCalcDb
from pymatgen import Molecule
import copy
from itertools import combinations_with_replacement, combinations
from pymatgen.analysis.fragmenter import open_ring
from monty.serialization import dumpfn, loadfn
import numpy as np
import pathos
import gzip

def convert_atomic_numbers_to_stoi_dict(atomic_numbers):
    '''

    :param atomic_numbers: a list of atomic numbers
    :return: {'Li':1, '110':0,'C':3,...} zero padding for non-existing elements
    '''
    atomic_num_to_element = {1:'H',3:'Li',6:'C',8:'O',9:'F',15:'P'}
    elements = ['H','Li','C','O','F','P']
    stoi_dict = {}

    for num in atomic_numbers:
        if atomic_num_to_element[num] in stoi_dict.keys():
            stoi_dict[atomic_num_to_element[num]] += 1
        else:
            stoi_dict[atomic_num_to_element[num]] = 1
    for ele in elements:
        if ele not in stoi_dict.keys():
            stoi_dict[ele] = 0
    return stoi_dict

def combine_stoi_dict(stoi_dict1, stoi_dict2):
    new_stoi_dict = {'C': stoi_dict1['C'] + stoi_dict2['C'], 'O': stoi_dict1['O'] + stoi_dict2['O'],
                     'H': stoi_dict1['H'] + stoi_dict2['H'], 'Li': stoi_dict1['Li'] + stoi_dict2['Li'],
                     'P': stoi_dict1['P'] + stoi_dict2['P'], 'F':stoi_dict1['F'] + stoi_dict2['F']}
    return new_stoi_dict

def combine_stoi_dict_triple(stoi_dict1, stoi_dict2, stoi_dict3):
    new_stoi_dict = {'C': stoi_dict1['C'] + stoi_dict2['C'] + stoi_dict3['C'], 'O': stoi_dict1['O'] + stoi_dict2['O'] + stoi_dict3['O'],
                     'H': stoi_dict1['H'] + stoi_dict2['H'] + stoi_dict3['H'], 'Li': stoi_dict1['Li'] + stoi_dict2['Li'] + stoi_dict3['Li'],
                     'P': stoi_dict1['P'] + stoi_dict2['P'] + stoi_dict3['P'], 'F':stoi_dict1['F'] + stoi_dict2['F'] + stoi_dict3['F']}
    return new_stoi_dict


def identify_same_stoi_mol_pairs(mol_graphs):
    '''
    :param mol_graphs: A list of mol_graphs
    :return: A dictionary with all mol pairs(or single molecule) that adds up to the same stoichiometry
    '''
    stoi_list = []
    final_dict = {}
    num_mols = len(mol_graphs)
    all_mol_pair_index = list(combinations_with_replacement(range(num_mols), 2))
    for mol_pair in all_mol_pair_index:
        index1 = mol_pair[0]
        index2 = mol_pair[1]
        pair_key = str(index1) + '_' + str(index2)
        mol1 = mol_graphs[index1].molecule
        mol2 = mol_graphs[index2].molecule
        stoi_dict1 = convert_atomic_numbers_to_stoi_dict(mol1.atomic_numbers)
        stoi_dict2 = convert_atomic_numbers_to_stoi_dict(mol2.atomic_numbers)
        stoi_dict = combine_stoi_dict(stoi_dict1, stoi_dict2)
        if stoi_dict in stoi_list:
            index_in_list = stoi_list.index(stoi_dict)
            final_dict[index_in_list].append(pair_key)
        else:
            final_dict[len(stoi_list)] = [pair_key]
            stoi_list.append(stoi_dict)
    for i, mol_graph in enumerate(mol_graphs):
        mol = mol_graph.molecule
        stoi_dict = convert_atomic_numbers_to_stoi_dict(mol.atomic_numbers)
        if stoi_dict in stoi_list:
            index_in_list = stoi_list.index(stoi_dict)
            final_dict[index_in_list].append(str(i))
        else:
            final_dict[len(stoi_list)] = [str(i)]
            stoi_list.append(stoi_dict)

    return stoi_list, final_dict

def identify_same_stoi_mol_pairs_ABC_DE(mol_graphs):
    '''
    For identifying A + B + C -> D + E reaction candidates that match the stoichiometry.
    TODO: Only allowing triples to include H/Li/F single atom for now.
    :param mol_graphs: A list of mol_graphs
    :return: A dictionary with all mol pairs(or single molecule) that adds up to the same stoichiometry
    '''
    stoi_list = []
    final_dict = {}
    num_mols = len(mol_graphs)
    all_mol_triple_index = list(combinations_with_replacement(range(num_mols), 3))
    all_mol_pair_index = list(combinations_with_replacement(range(num_mols), 2))

    for mol_pair in all_mol_pair_index:
        index1 = mol_pair[0]
        index2 = mol_pair[1]
        pair_key = str(index1) + '_' + str(index2)
        mol1 = mol_graphs[index1].molecule
        mol2 = mol_graphs[index2].molecule
        stoi_dict1 = convert_atomic_numbers_to_stoi_dict(mol1.atomic_numbers)
        stoi_dict2 = convert_atomic_numbers_to_stoi_dict(mol2.atomic_numbers)
        stoi_dict = combine_stoi_dict(stoi_dict1, stoi_dict2)
        if stoi_dict in stoi_list:
            index_in_list = stoi_list.index(stoi_dict)
            final_dict[index_in_list].append(pair_key)
        else:
            final_dict[len(stoi_list)] = [pair_key]
            stoi_list.append(stoi_dict)

    allowed_symbol_list = ['Li', 'H', 'F']
    for mol_triple in all_mol_triple_index:
        index1 = mol_triple[0]
        index2 = mol_triple[1]
        index3 = mol_triple[2]

        triple_key = str(index1) + '_' + str(index2) + '_' + str(index3)
        mol1 = mol_graphs[index1].molecule
        mol2 = mol_graphs[index2].molecule
        mol3 = mol_graphs[index3].molecule
        if (len(mol1) == 1 and mol1.sites[0].specie.symbol in allowed_symbol_list) or \
                (len(mol2) == 1 and mol2.sites[0].specie.symbol in allowed_symbol_list) or \
                (len(mol3) == 1 and mol3.sites[0].specie.symbol in allowed_symbol_list):

            stoi_dict1 = convert_atomic_numbers_to_stoi_dict(mol1.atomic_numbers)
            stoi_dict2 = convert_atomic_numbers_to_stoi_dict(mol2.atomic_numbers)
            stoi_dict3 = convert_atomic_numbers_to_stoi_dict(mol3.atomic_numbers)
            stoi_dict = combine_stoi_dict_triple(stoi_dict1, stoi_dict2, stoi_dict3)
            if stoi_dict in stoi_list:
                index_in_list = stoi_list.index(stoi_dict)
                final_dict[index_in_list].append(triple_key)
            else:
                final_dict[len(stoi_list)] = [triple_key]
                stoi_list.append(stoi_dict)

    for i, mol_graph in enumerate(mol_graphs):
        mol = mol_graph.molecule
        stoi_dict = convert_atomic_numbers_to_stoi_dict(mol.atomic_numbers)
        if stoi_dict in stoi_list:
            index_in_list = stoi_list.index(stoi_dict)
            final_dict[index_in_list].append(str(i))
        else:
            final_dict[len(stoi_list)] = [str(i)]
            stoi_list.append(stoi_dict)

    return stoi_list, final_dict

def is_equivalent(mol_graph1, mol_graph2):
    is_equivalent = False
    if mol_graph1.molecule.composition.alphabetical_formula == mol_graph1.molecule.composition.alphabetical_formula:
        if mol_graph1.isomorphic_to(mol_graph2):
            is_equivalent = True
    return is_equivalent

def check_in_list(test_mol_graph, mol_graphs):
    '''
    Check is test_mol_graph is in mol_graphs
    :param test_mol_graph:
    :param mol_graphs:
    :return: True or False
    '''
    is_in_list = False
    for mol_graph in mol_graphs:
        if test_mol_graph.molecule.composition.alphabetical_formula == mol_graph.molecule.composition.alphabetical_formula:
            if test_mol_graph.isomorphic_to(mol_graph):
                is_in_list = True
                break
    return is_in_list

def find_one_same_mol(mol_graphs1, mol_graphs2):
    '''
    Find one same mol graph in two lists and return two lists eliminating those same mols
    :param mol_graphs1:
    :param mol_graphs2:
    :return:
    '''
    found_one_equivalent_graph = False
    mol_graphs1_copy = copy.deepcopy(mol_graphs1)
    mol_graphs2_copy = copy.deepcopy(mol_graphs2)
    for i, graph1 in enumerate(mol_graphs1):
        for j, graph2 in enumerate(mol_graphs2):
            if is_equivalent(graph1, graph2):
                found_one_equivalent_graph = True
                mol_graphs1_copy.pop(i)
                mol_graphs2_copy.pop(j)
                return str(found_one_equivalent_graph), mol_graphs1_copy, mol_graphs2_copy
    return str(found_one_equivalent_graph), mol_graphs1_copy, mol_graphs2_copy


def check_same_mol_graphs(mol_graphs1, mol_graphs2):
    '''
    Check is two mol graphs list are identical, assuming every mol graph in one list is unique
    :param mol_graphs1:
    :param mol_graphs2:
    :return: True or False
    '''
    is_the_same = False
    mol_graphs1_copy = copy.deepcopy(mol_graphs1)
    mol_graphs2_copy = copy.deepcopy(mol_graphs2)
    sorted_formula_1 = sorted([mol_graph.molecule.composition.alphabetical_formula for mol_graph in mol_graphs1])
    sorted_formula_2 = sorted([mol_graph.molecule.composition.alphabetical_formula for mol_graph in mol_graphs2])
    if sorted_formula_1 == sorted_formula_2:
        while mol_graphs1_copy != [] and mol_graphs2_copy != []:
            found_one_equivalent_graph, mol_graphs1_copy, mol_graphs2_copy = find_one_same_mol(mol_graphs1_copy, mol_graphs2_copy)
            if found_one_equivalent_graph == 'False':
                return is_the_same
        is_the_same = True
    return is_the_same

def check_mol_graphs_in_list(mol_graphs, mol_graphs_list):
    '''
    Check if a mol graphs list is in a list of list of mol_graphs
    :param mol_graphs:
    :param mol_graphs_list:
    :return: True or False
    '''
    is_in_list = False
    if mol_graphs_list == []:
        return is_in_list
    for mol_graphs_orig in mol_graphs_list:
        if check_same_mol_graphs(mol_graphs, mol_graphs_orig):
            is_in_list = True
            break
    return is_in_list

def break_one_bond_in_one_mol(mol_graph):
    all_possible_fragments = []
    if len(mol_graph.graph.edges) != 0:
        for edge in mol_graph.graph.edges:
            bond = [(edge[0], edge[1])]
            try:
                mol_graph_copy = copy.deepcopy(mol_graph)
                frags1 = mol_graph_copy.split_molecule_subgraphs(bond, allow_reverse=True)
                if not check_mol_graphs_in_list(frags1,all_possible_fragments):
                    all_possible_fragments.append(frags1)
            except MolGraphSplitError:
                mol_graph_copy = copy.deepcopy(mol_graph)
                frag1 = open_ring(mol_graph_copy, bond, 10000)
                if not check_mol_graphs_in_list([frag1],all_possible_fragments):
                    all_possible_fragments.append([frag1])
    if not check_mol_graphs_in_list([mol_graph],all_possible_fragments):
        all_possible_fragments.append([mol_graph])
    return all_possible_fragments

def break_two_bonds_in_one_mol(mol_graph):
    '''
    Break two bonds in one single molecule and generate all the possible fragments, including itself,
    including fragments from breaking only one bond
    :param mol_graph:
    :return: A list of list of fragments
    '''
    all_possible_fragments = []
    if len(mol_graph.graph.edges) != 0:
        for edge in mol_graph.graph.edges:
            bond = [(edge[0], edge[1])]
            #print('bond:',bond)
            try:
                mol_graph_copy = copy.deepcopy(mol_graph)
                frags1 = mol_graph_copy.split_molecule_subgraphs(bond, allow_reverse=True)
                #print('original length:',len(frags1))
                if not check_mol_graphs_in_list(frags1,all_possible_fragments):
                    all_possible_fragments.append(frags1)
                #print('second length:',len(frags1))
                for i in range(2):
                    #print(i)
                    frags1_copy = copy.deepcopy(frags1)
                    frag = frags1_copy[i]
                    if len(frag.graph.edges) != 0:
                        for edge2 in frag.graph.edges:
                            bond2 = [(edge2[0],edge2[1])]
                            #print('bond2:',bond2)
                            #print('modified length:',len(frags1))
                            try:
                                frag_copy = copy.deepcopy(frag)
                                frags2 = frag_copy.split_molecule_subgraphs(bond2, allow_reverse=True)
                                frags1_new = copy.deepcopy(frags1)
                                frags1_new_new = []
                                if i == 0:
                                    frags1_new_new = [frags1_new[1]]
                                elif i == 1:
                                    frags1_new_new = [frags1_new[0]]
                                if not check_mol_graphs_in_list(frags2+frags1_new_new, all_possible_fragments):
                                    all_possible_fragments.append(frags2+frags1_new_new)

                            except MolGraphSplitError:
                                frag_copy = copy.deepcopy(frag)
                                frag2 = open_ring(frag_copy, bond2, 10000)
                                frags1_new = copy.deepcopy(frags1)
                                frags1_new_new = []
                                if i == 0:
                                    frags1_new_new = [frags1_new[1]]
                                elif i == 1:
                                    frags1_new_new = [frags1_new[0]]
                                if not check_mol_graphs_in_list([frag2]+frags1_new_new, all_possible_fragments):
                                    all_possible_fragments.append([frag2]+frags1_new_new)


            except MolGraphSplitError:
                mol_graph_copy = copy.deepcopy(mol_graph)
                frag1 = open_ring(mol_graph_copy, bond, 10000)
                if not check_mol_graphs_in_list([frag1],all_possible_fragments):
                    all_possible_fragments.append([frag1])
                if len(frag1.graph.edges) != 0:
                    for edge2 in frag1.graph.edges:
                        bond2 = [(edge2[0],edge2[1])]
                        #print('bond2_2:',bond2)
                        try:
                            frag1_copy = copy.deepcopy(frag1)
                            frags2 = frag1_copy.split_molecule_subgraphs(bond2, allow_reverse=True)
                            if not check_mol_graphs_in_list(frags2, all_possible_fragments):
                                all_possible_fragments.append(frags2)
                        except MolGraphSplitError:
                            frag1_copy = copy.deepcopy(frag1)
                            frag2 = open_ring(frag1_copy, bond2, 10000)
                            if not check_mol_graphs_in_list([frag2],all_possible_fragments):
                                all_possible_fragments.append([frag2])
    if not check_mol_graphs_in_list([mol_graph],all_possible_fragments):
        all_possible_fragments.append([mol_graph])

    return all_possible_fragments

def open_ring_in_one_mol(mol_graph):
    '''
    Generate all possible ring opened fragments. Have to be ring opening
    :param mol_graph:
    :return: A list of fragments
    '''
    all_possible_fragments = []
    if len(mol_graph.find_rings()) != 0:
        for edge in mol_graph.graph.edges:
            bond = [(edge[0],edge[1])]
            try:
                frag = open_ring(mol_graph, bond, 10000)
                if not check_in_list(frag, all_possible_fragments):
                    all_possible_fragments.append(frag)
            except:
                continue
    return all_possible_fragments

def is_ring_isomorphic(mol_graph1, mol_graph2):
    '''
    See if mol_graph1 and mol_graph2 can be equivalent by opening a ring
    :param mol_graph1:
    :param mol_graph2:
    :return:
    '''
    is_ring_isomorphic = False
    if mol_graph1.molecule.composition.alphabetical_formula == mol_graph1.molecule.composition.alphabetical_formula:
        if mol_graph1.isormorphic_to(mol_graph2):
            is_ring_isomorphic = True
        else:
            frags1 = open_ring_in_one_mol(mol_graph1)
            if frags1 != []:
                if check_in_list(mol_graph2, frags1):
                    is_ring_isomorphic = True
                else:
                    frags2 = open_ring_in_one_mol(mol_graph2)
                    if check_in_list(mol_graph1, frags2):
                        is_ring_isomorphic = True
    return is_ring_isomorphic

def identify_self_reactions(mol_graph1, mol_graph2):
    '''
    break A, B once each. Not considering breaking two or more bonds in a mol.
    :param mol_graph1:
    :param mol_graph2:
    :return:
    '''
    is_self_reaction = False
    A = mol_graph1
    B = mol_graph2
    frags_A_one_step = break_one_bond_in_one_mol(A)
    frags_B_one_step = break_one_bond_in_one_mol(B)
    for item_A in frags_A_one_step:
        for item_B in frags_B_one_step:
            if check_same_mol_graphs(item_A, item_B):
                is_self_reaction = True
                return is_self_reaction
    return is_self_reaction

def identify_self_reactions_record(mol_graph1, mol_graph2, num1, num2, one_bond_dict):
    '''
    break A, B once each. Not considering breaking two or more bonds in a mol.
    :param mol_graph1:
    :param mol_graph2:
    :return:
    '''
    is_self_reaction = False
    A = mol_graph1
    B = mol_graph2
    if num1 in one_bond_dict.keys():
        frags_A_one_step = one_bond_dict[num1]
    else:
        frags_A_one_step = break_one_bond_in_one_mol(A)
        one_bond_dict[num1] = frags_A_one_step
    if num2 in one_bond_dict.keys():
        frags_B_one_step = one_bond_dict[num2]
    else:
        frags_B_one_step = break_one_bond_in_one_mol(B)
        one_bond_dict[num2] = frags_B_one_step
    for item_A in frags_A_one_step:
        for item_B in frags_B_one_step:
            if check_same_mol_graphs(item_A, item_B):
                is_self_reaction = True
                return str(is_self_reaction), one_bond_dict
    return str(is_self_reaction), one_bond_dict

def identify_self_reactions_record_one_bond_breakage(mol_graph1, mol_graph2, num1, num2, one_bond_dict):
    '''
    break A, B once each. Not considering breaking two or more bonds in a mol.
    :param mol_graph1:
    :param mol_graph2:
    :return:
    '''
    is_self_reaction = False
    A = mol_graph1
    B = mol_graph2

    if num1 in one_bond_dict.keys():
        frags_A_one_step = one_bond_dict[num1]
    else:
        frags_A_one_step = break_one_bond_in_one_mol(A)
        one_bond_dict[num1] = frags_A_one_step
    if num2 in one_bond_dict.keys():
        frags_B_one_step = one_bond_dict[num2]
    else:
        frags_B_one_step = break_one_bond_in_one_mol(B)
        one_bond_dict[num2] = frags_B_one_step
    for item_A in frags_A_one_step:
        if check_same_mol_graphs(item_A, [B]):
            is_self_reaction = True
            return str(is_self_reaction), one_bond_dict
    for item_B in frags_B_one_step:
        if check_same_mol_graphs(item_B, [A]):
            is_self_reaction = True
            return str(is_self_reaction), one_bond_dict
    return str(is_self_reaction), one_bond_dict

def identify_reactions_AB_C(mol_graphs1, mol_graphs2):
    '''
    A + B -> C type reactions
    1. A, B each break once, C break twice
    2. A or B break twice, the other intact, C break twice
    :param mol_graphs1: 2 components A and B
    :param mol_graphs2: 1 component C
    :return: True or False
    '''
    is_reactions_AB_C = False
    assert len(mol_graphs1) == 2 and len(mol_graphs2) == 1
    A = mol_graphs1[0]
    B = mol_graphs1[1]
    C = mol_graphs2[0]

    frags_A_one_step = break_one_bond_in_one_mol(A)
    frags_B_one_step = break_one_bond_in_one_mol(B)
    frags_C_two_step = break_two_bonds_in_one_mol(C)

    # A B each break once, C break twice
    for item_A in frags_A_one_step:
        for item_B in frags_B_one_step:
            for item_C in frags_C_two_step:
                    if check_same_mol_graphs(item_A + item_B, item_C):
                        is_reactions_AB_C = True
                        print('AB each once!',flush=True)
                        return is_reactions_AB_C

    # A or B break twice, the other intact, C break twice
    frags_A_two_step = break_two_bonds_in_one_mol(A)
    frags_B_two_step = break_two_bonds_in_one_mol(B)

    for item_A in frags_A_two_step:
            for item_C in frags_C_two_step:
                    if check_same_mol_graphs(item_A + [B], item_C):
                        is_reactions_AB_C = True
                        print('AC twice, B intact!',flush=True)
                        return is_reactions_AB_C

    for item_B in frags_B_two_step:
            for item_C in frags_C_two_step:
                    if check_same_mol_graphs([A] + item_B, item_C):
                        is_reactions_AB_C = True
                        print('BC twice, A intact!',flush=True)
                        return is_reactions_AB_C

    return is_reactions_AB_C

def identify_reactions_AB_C_break1_form1(mol_graphs1, mol_graphs2):
    '''
    A + B -> C type reactions
    A or B break once, C break once
    :param mol_graphs1: 2 components A and B
    :param mol_graphs2: 1 component C
    :return: True or False
    '''
    is_reactions_AB_C = False
    assert len(mol_graphs1) == 2 and len(mol_graphs2) == 1
    A = mol_graphs1[0]
    B = mol_graphs1[1]
    C = mol_graphs2[0]

    frags_A_one_step = break_one_bond_in_one_mol(A)
    frags_B_one_step = break_one_bond_in_one_mol(B)
    frags_C_one_step = break_one_bond_in_one_mol(C)

    # A C break once
    for item_A in frags_A_one_step:
        for item_C in frags_C_one_step:
            if check_same_mol_graphs(item_A + [B], item_C):
                is_reactions_AB_C = True
                print('A once, C once!',flush=True)
                return is_reactions_AB_C

    # B C break once
    for item_B in frags_B_one_step:
        for item_C in frags_C_one_step:
            if check_same_mol_graphs(item_B + [A], item_C):
                is_reactions_AB_C = True
                print('B once, C once!',flush=True)
                return is_reactions_AB_C

    return is_reactions_AB_C

def identify_reactions_AB_C_record(mol_graphs1, mol_graphs2, nums1, nums2, one_bond_dict, two_bond_dict):
    '''
    A + B -> C type reactions
    1. A, B each break once, C break twice
    2. A or B break twice, the other intact, C break twice
    :param mol_graphs1: 2 components A and B
    :param mol_graphs2: 1 component C
    :return: True or False
    '''
    is_reactions_AB_C = False
    assert len(mol_graphs1) == 2 and len(mol_graphs2) == 1
    A = mol_graphs1[0]
    B = mol_graphs1[1]
    C = mol_graphs2[0]
    num_A = nums1[0]
    num_B = nums1[1]
    num_C = nums2[0]

    if num_A in one_bond_dict.keys():
        frags_A_one_step = one_bond_dict[num_A]
    else:
        frags_A_one_step = break_one_bond_in_one_mol(A)
        one_bond_dict[num_A] = frags_A_one_step

    if num_B in one_bond_dict.keys():
        frags_B_one_step = one_bond_dict[num_B]
    else:
        frags_B_one_step = break_one_bond_in_one_mol(B)
        one_bond_dict[num_B] = frags_B_one_step

    if num_C in two_bond_dict.keys():
        frags_C_two_step = two_bond_dict[num_C]
    else:
        frags_C_two_step = break_two_bonds_in_one_mol(C)
        two_bond_dict[num_C] = frags_C_two_step

    # A B each break once, C break twice
    for item_A in frags_A_one_step:
        for item_B in frags_B_one_step:
            for item_C in frags_C_two_step:
                    if check_same_mol_graphs(item_A + item_B, item_C):
                        is_reactions_AB_C = True
                        print('AB each once!',flush=True)
                        return str(is_reactions_AB_C), one_bond_dict, two_bond_dict

    # A or B break twice, the other intact, C break twice
    if num_A in two_bond_dict.keys():
        frags_A_two_step = two_bond_dict[num_A]
    else:
        frags_A_two_step = break_two_bonds_in_one_mol(A)
        two_bond_dict[num_A] = frags_A_two_step

    if num_B in two_bond_dict.keys():
        frags_B_two_step = two_bond_dict[num_B]
    else:
        frags_B_two_step = break_two_bonds_in_one_mol(B)
        two_bond_dict[num_B] = frags_B_two_step

    for item_A in frags_A_two_step:
            for item_C in frags_C_two_step:
                    if check_same_mol_graphs(item_A + [B], item_C):
                        is_reactions_AB_C = True
                        print('AC twice, B intact!',flush=True)
                        return str(is_reactions_AB_C), one_bond_dict, two_bond_dict

    for item_B in frags_B_two_step:
            for item_C in frags_C_two_step:
                    if check_same_mol_graphs([A] + item_B, item_C):
                        is_reactions_AB_C = True
                        print('BC twice, A intact!',flush=True)
                        return str(is_reactions_AB_C), one_bond_dict, two_bond_dict

    return str(is_reactions_AB_C), one_bond_dict, two_bond_dict

def identify_reactions_AB_C_record_one_bond_breakage(mol_graphs1, mol_graphs2, nums1, nums2, one_bond_dict):
    '''
    A + B -> C type reactions
    one bond breakage
    :param mol_graphs1: 2 components A and B
    :param mol_graphs2: 1 component C
    :return: True or False
    '''
    is_reactions_AB_C = False
    assert len(mol_graphs1) == 2 and len(mol_graphs2) == 1
    A = mol_graphs1[0]
    B = mol_graphs1[1]
    C = mol_graphs2[0]
    num_A = nums1[0]
    num_B = nums1[1]
    num_C = nums2[0]

    if num_C in one_bond_dict.keys():
        frags_C_one_step = one_bond_dict[num_C]
    else:
        frags_C_one_step = break_one_bond_in_one_mol(C)
        one_bond_dict[num_C] = frags_C_one_step

    # A B each break once, C break twice

    for item_C in frags_C_one_step:
        if check_same_mol_graphs([A] + B, item_C):
            is_reactions_AB_C = True
            print('AB each once!',flush=True)
    return str(is_reactions_AB_C), one_bond_dict


def identify_reactions_AB_CD(mol_graphs1, mol_graphs2):
    '''
    Identify reactions type A + B -> C + D
    1. A, B, C, D all break once, creating A1, A2, B1, B2 == C1, C2, D1, D2
    2. one of A, B breaks twice; C, D both break once each. i.e. A + B -> A1 + A2 + A3 + B == C1 + C2 + D1 + D2
    3. one of C, D breaks twice; A, B both break once each. i.e. A + B -> A1 + A2 + B1 + B2 == C1 + C2 + C3 + D
    4. one of A, B breaks twice; one of C, D breaks twice. i.e. A + B -> A1 + A2 + A3 + B == C1 + C2 + C3 + D
    :param mol_graphs1:
    :param mol_graphs2:
    :return: True or False
    '''
    is_reactions_AB_CD = False
    A = mol_graphs1[0]
    B = mol_graphs1[1]
    C = mol_graphs2[0]
    D = mol_graphs2[1]

    frags_A_one_step = break_one_bond_in_one_mol(A)
    frags_B_one_step = break_one_bond_in_one_mol(B)
    frags_C_one_step = break_one_bond_in_one_mol(C)
    frags_D_one_step = break_one_bond_in_one_mol(D)
    # break each mol once (scenario 1)
    for item_A in frags_A_one_step:
        for item_B in frags_B_one_step:
            for item_C in frags_C_one_step:
                for item_D in frags_D_one_step:
                    if check_same_mol_graphs(item_A + item_B, item_C + item_D) == True:
                        is_reactions_AB_CD = True
                        print('ABCD each once!',flush=True)
                        return is_reactions_AB_CD

    frags_A_two_step = break_two_bonds_in_one_mol(A)
    frags_B_two_step = break_two_bonds_in_one_mol(B)
    frags_C_two_step = break_two_bonds_in_one_mol(C)
    frags_D_two_step = break_two_bonds_in_one_mol(D)

    # break one mol two steps (scenario 2)
    for item_A in frags_A_two_step:
        for item_C in frags_C_one_step:
            for item_D in frags_D_one_step:
                if check_same_mol_graphs(item_A + [B], item_C + item_D):
                    is_reactions_AB_CD = True
                    print('break A twice, CD once!',flush=True)
                    return is_reactions_AB_CD

    for item_B in frags_B_two_step:
        for item_C in frags_C_one_step:
            for item_D in frags_D_one_step:
                if check_same_mol_graphs([A] + item_B, item_C + item_D):
                    is_reactions_AB_CD = True
                    print('break B twice, CD once!',flush=True)
                    return is_reactions_AB_CD

    for item_C in frags_C_two_step:
        for item_A in frags_A_one_step:
            for item_B in frags_B_one_step:
                if check_same_mol_graphs(item_A + item_B, item_C + [D]):
                    is_reactions_AB_CD = True
                    print('break C twice, AB once!',flush=True)
                    return is_reactions_AB_CD

    for item_D in frags_D_two_step:
        for item_A in frags_A_one_step:
            for item_B in frags_B_one_step:
                if check_same_mol_graphs(item_A + item_B, [C] + item_D):
                    is_reactions_AB_CD = True
                    print('break D twice, AB once!',flush=True)
                    return is_reactions_AB_CD

    # break two mol two steps (scenario 3)
    for item_A in frags_A_two_step:
        for item_C in frags_C_two_step:
            if check_same_mol_graphs(item_A + [B], item_C + [D]):
                is_reactions_AB_CD = True
                print('break AC twice, BD intact',flush=True)
                return is_reactions_AB_CD

    for item_A in frags_A_two_step:
        for item_D in frags_D_two_step:
            if check_same_mol_graphs(item_A + [B], [C] + item_D):
                is_reactions_AB_CD = True
                print('break AD twice, BC intact',flush=True)
                return is_reactions_AB_CD

    for item_B in frags_B_two_step:
        for item_C in frags_C_two_step:
            if check_same_mol_graphs([A] + item_B, item_C + [D]):
                is_reactions_AB_CD = True
                print('break BC twice, AD intact',flush=True)
                return is_reactions_AB_CD

    for item_B in frags_B_two_step:
        for item_D in frags_D_two_step:
            if check_same_mol_graphs([A] + item_B, [C] + item_D):
                is_reactions_AB_CD = True
                print('break AC twice, BD intact',flush=True)
                return is_reactions_AB_CD

    return is_reactions_AB_CD

def identify_reactions_ABC_DE(mol_graphs1, mol_graphs2):
    '''
    Identify reactions type A + B + C -> D + E
    Allowing <=2 bond formation, <=2 bond breaking.
    TODO: Only allowing a single Li/H to be on the ABC side for now.
    If one of A,B,C is a single atom molecule, then it can't break a bond

    Scenerios:
    1. A,B/A,C/B,C break once, D,E break once
    2. one of A,B,C breaks twice; D,E break once
    3. A,B/A,C/B,C break once, one of D,E breaks twice
    4. One of A,B,C breaks twic; one of D,E breaks twice

    :param mol_graphs1:
    :param mol_graphs2:
    :return: True or False
    '''
    is_reactions_ABC_DE = False
    assert len(mol_graphs1) == 3
    assert len(mol_graphs2) == 2
    A = mol_graphs1[0]
    B = mol_graphs1[1]
    C = mol_graphs1[2]
    D = mol_graphs2[0]
    E = mol_graphs2[1]

    if len(A.molecule) == 1:
        pass
    elif len(B.molecule) == 1:
        #swap A and B
        A, B = B, A
    elif len(C.molecule) == 1:
        #swap A and C
        A, C = C, A

    frags_B_one_step = break_one_bond_in_one_mol(B)
    frags_C_one_step = break_one_bond_in_one_mol(C)
    frags_D_one_step = break_one_bond_in_one_mol(D)
    frags_E_one_step = break_one_bond_in_one_mol(E)

    # break each mol once (scenario 1)
    for item_B in frags_B_one_step:
        for item_C in frags_C_one_step:
            for item_D in frags_D_one_step:
                for item_E in frags_E_one_step:
                    if check_same_mol_graphs([A] + item_B + item_C, item_D + item_E) == True:
                        is_reactions_ABC_DE = True
                        print('BCDE each once!',flush=True)
                        return is_reactions_ABC_DE

    frags_B_two_step = break_two_bonds_in_one_mol(B)
    frags_C_two_step = break_two_bonds_in_one_mol(C)
    frags_D_two_step = break_two_bonds_in_one_mol(D)
    frags_E_two_step = break_two_bonds_in_one_mol(E)

    # break one mol two steps (scenario 2)
    for item_B in frags_B_two_step:
        for item_D in frags_D_one_step:
            for item_E in frags_E_one_step:
                if check_same_mol_graphs(item_B + [A,C], item_D + item_E):
                    is_reactions_ABC_DE = True
                    print('break B twice, DE once!',flush=True)
                    return is_reactions_ABC_DE

    for item_C in frags_C_two_step:
        for item_D in frags_D_one_step:
            for item_E in frags_E_one_step:
                if check_same_mol_graphs([A,B] + item_C, item_D + item_E):
                    is_reactions_ABC_DE = True
                    print('break C twice, DE once!',flush=True)
                    return is_reactions_ABC_DE

    for item_D in frags_D_two_step:
        for item_B in frags_B_one_step:
            for item_C in frags_C_one_step:
                if check_same_mol_graphs([A] + item_B + item_C, item_D + [E]):
                    is_reactions_ABC_DE = True
                    print('break D twice, BC once!',flush=True)
                    return is_reactions_ABC_DE

    for item_E in frags_E_two_step:
        for item_B in frags_B_one_step:
            for item_C in frags_C_one_step:
                if check_same_mol_graphs([A] + item_B + item_C, [D] + item_E):
                    is_reactions_ABC_DE = True
                    print('break E twice, BC once!',flush=True)
                    return is_reactions_ABC_DE

    # break two mol two steps (scenario 3)
    for item_B in frags_B_two_step:
        for item_D in frags_D_two_step:
            if check_same_mol_graphs(item_B + [A,C], item_D + [E]):
                is_reactions_ABC_DE = True
                print('break BD twice, ACE intact',flush=True)
                return is_reactions_ABC_DE

    for item_B in frags_B_two_step:
        for item_E in frags_E_two_step:
            if check_same_mol_graphs(item_B + [A,C], [D] + item_E):
                is_reactions_ABC_DE = True
                print('break BE twice, ACD intact',flush=True)
                return is_reactions_ABC_DE

    for item_C in frags_C_two_step:
        for item_D in frags_D_two_step:
            if check_same_mol_graphs([A,B] + item_C, item_D + [E]):
                is_reactions_ABC_DE = True
                print('break CD twice, ABE intact',flush=True)
                return is_reactions_ABC_DE

    for item_C in frags_C_two_step:
        for item_E in frags_E_two_step:
            if check_same_mol_graphs([A,B] + item_C, [D] + item_E):
                is_reactions_ABC_DE = True
                print('break CE twice, ABD intact',flush=True)
                return is_reactions_ABC_DE

    return is_reactions_ABC_DE

def identify_reactions_AB_CD_break1_form1(mol_graphs1, mol_graphs2):
    '''
    Identify reactions type A + B -> C + D with break1 form1
    1. break A once, break C once
    2. break A once, break D once
    3. break B once, break C once
    4. break B once, break D once
    :param mol_graphs1:
    :param mol_graphs2:
    :return: True or False
    '''
    is_reactions_AB_CD = False
    A = mol_graphs1[0]
    B = mol_graphs1[1]
    C = mol_graphs2[0]
    D = mol_graphs2[1]

    frags_A_one_step = break_one_bond_in_one_mol(A)
    frags_B_one_step = break_one_bond_in_one_mol(B)
    frags_C_one_step = break_one_bond_in_one_mol(C)
    frags_D_one_step = break_one_bond_in_one_mol(D)

    # A C break once
    for item_A in frags_A_one_step:
        for item_C in frags_C_one_step:
            if check_same_mol_graphs(item_A + [B], item_C + [D]):
                is_reactions_AB_CD = True
                print('A once, C once!',flush=True)
                return is_reactions_AB_CD

    # A D break once
    for item_A in frags_A_one_step:
        for item_D in frags_D_one_step:
            if check_same_mol_graphs(item_A + [B], [C] + item_D):
                is_reactions_AB_CD = True
                print('A once, D once!',flush=True)
                return is_reactions_AB_CD

    # B C break once
    for item_B in frags_B_one_step:
        for item_C in frags_C_one_step:
            if check_same_mol_graphs(item_B + [A], item_C + [D]):
                is_reactions_AB_CD = True
                print('B once, C once!',flush=True)
                return is_reactions_AB_CD

    # B D break once
    for item_B in frags_B_one_step:
        for item_D in frags_D_one_step:
            if check_same_mol_graphs(item_B + [A], [C] + item_D):
                is_reactions_AB_CD = True
                print('B once, D once!',flush=True)
                return is_reactions_AB_CD

    return is_reactions_AB_CD

def identify_reactions_AB_CD_record(mol_graphs1, mol_graphs2, nums1, nums2, one_bond_dict, two_bond_dict):
    '''
    Identify reactions type A + B -> C + D
    1. A, B, C, D all break once, creating A1, A2, B1, B2 == C1, C2, D1, D2
    2. one of A, B breaks twice; C, D both break once each. i.e. A + B -> A1 + A2 + A3 + B == C1 + C2 + D1 + D2
    3. one of C, D breaks twice; A, B both break once each. i.e. A + B -> A1 + A2 + B1 + B2 == C1 + C2 + C3 + D
    4. one of A, B breaks twice; one of C, D breaks twice. i.e. A + B -> A1 + A2 + A3 + B == C1 + C2 + C3 + D
    :param mol_graphs1:
    :param mol_graphs2:
    :return: True or False
    '''
    is_reactions_AB_CD = False
    A = mol_graphs1[0]
    B = mol_graphs1[1]
    C = mol_graphs2[0]
    D = mol_graphs2[1]
    num_A = nums1[0]
    num_B = nums1[1]
    num_C = nums2[0]
    num_D = nums2[1]
    
    if num_A in one_bond_dict.keys():
        frags_A_one_step = one_bond_dict[num_A]
    else:
        frags_A_one_step = break_one_bond_in_one_mol(A)
        one_bond_dict[num_A] = frags_A_one_step

    if num_B in one_bond_dict.keys():
        frags_B_one_step = one_bond_dict[num_B]
    else:
        frags_B_one_step = break_one_bond_in_one_mol(B)
        one_bond_dict[num_B] = frags_B_one_step
    
    if num_C in one_bond_dict.keys():
        frags_C_one_step = one_bond_dict[num_C]
    else:
        frags_C_one_step = break_one_bond_in_one_mol(C)
        one_bond_dict[num_C] = frags_C_one_step

    if num_D in one_bond_dict.keys():
        frags_D_one_step = one_bond_dict[num_D]
    else:
        frags_D_one_step = break_one_bond_in_one_mol(D)
        one_bond_dict[num_D] = frags_D_one_step

    # break each mol once (scenario 1)
    for item_A in frags_A_one_step:
        for item_B in frags_B_one_step:
            for item_C in frags_C_one_step:
                for item_D in frags_D_one_step:
                    if check_same_mol_graphs(item_A + item_B, item_C + item_D) == True:
                        is_reactions_AB_CD = True
                        print('ABCD each once!', flush=True)
                        return str(is_reactions_AB_CD),one_bond_dict, two_bond_dict
    
    if num_A in two_bond_dict.keys():
        frags_A_two_step = two_bond_dict[num_A]
    else:
        frags_A_two_step = break_two_bonds_in_one_mol(A)
        two_bond_dict[num_A] = frags_A_two_step

    if num_B in two_bond_dict.keys():
        frags_B_two_step = two_bond_dict[num_B]
    else:
        frags_B_two_step = break_two_bonds_in_one_mol(B)
        two_bond_dict[num_B] = frags_B_two_step
        
    if num_C in two_bond_dict.keys():
        frags_C_two_step = two_bond_dict[num_C]
    else:
        frags_C_two_step = break_two_bonds_in_one_mol(C)
        two_bond_dict[num_C] = frags_C_two_step

    if num_D in two_bond_dict.keys():
        frags_D_two_step = two_bond_dict[num_D]
    else:
        frags_D_two_step = break_two_bonds_in_one_mol(D)
        two_bond_dict[num_D] = frags_D_two_step

    # break one mol two steps (scenario 2)
    for item_A in frags_A_two_step:
        for item_C in frags_C_one_step:
            for item_D in frags_D_one_step:
                if check_same_mol_graphs(item_A + [B], item_C + item_D):
                    is_reactions_AB_CD = True
                    print('break A twice, CD once!',flush=True)
                    return str(is_reactions_AB_CD),one_bond_dict, two_bond_dict

    for item_B in frags_B_two_step:
        for item_C in frags_C_one_step:
            for item_D in frags_D_one_step:
                if check_same_mol_graphs([A] + item_B, item_C + item_D):
                    is_reactions_AB_CD = True
                    print('break B twice, CD once!', flush=True)
                    return str(is_reactions_AB_CD), one_bond_dict, two_bond_dict

    for item_C in frags_C_two_step:
        for item_A in frags_A_one_step:
            for item_B in frags_B_one_step:
                if check_same_mol_graphs(item_A + item_B, item_C + [D]):
                    is_reactions_AB_CD = True
                    print('break C twice, AB once!', flush=True)
                    return str(is_reactions_AB_CD), one_bond_dict, two_bond_dict

    for item_D in frags_D_two_step:
        for item_A in frags_A_one_step:
            for item_B in frags_B_one_step:
                if check_same_mol_graphs(item_A + item_B, [C] + item_D):
                    is_reactions_AB_CD = True
                    print('break D twice, AB once!', flush=True)
                    return str(is_reactions_AB_CD), one_bond_dict, two_bond_dict

    # break two mol two steps (scenario 3)
    for item_A in frags_A_two_step:
        for item_C in frags_C_two_step:
            if check_same_mol_graphs(item_A + [B], item_C + [D]):
                is_reactions_AB_CD = True
                print('break AC twice, BD intact', flush=True)
                return str(is_reactions_AB_CD), one_bond_dict, two_bond_dict

    for item_A in frags_A_two_step:
        for item_D in frags_D_two_step:
            if check_same_mol_graphs(item_A + [B], [C] + item_D):
                is_reactions_AB_CD = True
                print('break AD twice, BC intact', flush=True)
                return str(is_reactions_AB_CD), one_bond_dict, two_bond_dict

    for item_B in frags_B_two_step:
        for item_C in frags_C_two_step:
            if check_same_mol_graphs([A] + item_B, item_C + [D]):
                is_reactions_AB_CD = True
                print('break BC twice, AD intact', flush=True)
                return str(is_reactions_AB_CD), one_bond_dict, two_bond_dict

    for item_B in frags_B_two_step:
        for item_D in frags_D_two_step:
            if check_same_mol_graphs([A] + item_B, [C] + item_D):
                is_reactions_AB_CD = True
                print('break AC twice, BD intact', flush=True)
                return str(is_reactions_AB_CD), one_bond_dict, two_bond_dict

    return str(is_reactions_AB_CD), one_bond_dict, two_bond_dict


def identify_reactions_AB_CD_record_one_bond_each(mol_graphs1, mol_graphs2, nums1, nums2, one_bond_dict):
    '''
    Identify reactions type A + B -> C + D
    1. A, B, C, D all break once, creating A1, A2, B1, B2 == C1, C2, D1, D2
    no 2. one of A, B breaks twice; C, D both break once each. i.e. A + B -> A1 + A2 + A3 + B == C1 + C2 + D1 + D2
    no 3. one of C, D breaks twice; A, B both break once each. i.e. A + B -> A1 + A2 + B1 + B2 == C1 + C2 + C3 + D
    no 4. one of A, B breaks twice; one of C, D breaks twice. i.e. A + B -> A1 + A2 + A3 + B == C1 + C2 + C3 + D
    :param mol_graphs1:
    :param mol_graphs2:
    :return: True or False
    '''
    is_reactions_AB_CD = False
    A = mol_graphs1[0]
    B = mol_graphs1[1]
    C = mol_graphs2[0]
    D = mol_graphs2[1]
    num_A = nums1[0]
    num_B = nums1[1]
    num_C = nums2[0]
    num_D = nums2[1]

    if num_A in one_bond_dict.keys():
        frags_A_one_step = one_bond_dict[num_A]
    else:
        frags_A_one_step = break_one_bond_in_one_mol(A)
        one_bond_dict[num_A] = frags_A_one_step

    if num_B in one_bond_dict.keys():
        frags_B_one_step = one_bond_dict[num_B]
    else:
        frags_B_one_step = break_one_bond_in_one_mol(B)
        one_bond_dict[num_B] = frags_B_one_step

    if num_C in one_bond_dict.keys():
        frags_C_one_step = one_bond_dict[num_C]
    else:
        frags_C_one_step = break_one_bond_in_one_mol(C)
        one_bond_dict[num_C] = frags_C_one_step

    if num_D in one_bond_dict.keys():
        frags_D_one_step = one_bond_dict[num_D]
    else:
        frags_D_one_step = break_one_bond_in_one_mol(D)
        one_bond_dict[num_D] = frags_D_one_step

    # break each mol once (scenario 1)
    for item_A in frags_A_one_step:
        for item_B in frags_B_one_step:
            for item_C in frags_C_one_step:
                for item_D in frags_D_one_step:
                    if check_same_mol_graphs(item_A + item_B, item_C + item_D) == True:
                        is_reactions_AB_CD = True
                        print('ABCD each once!', flush=True)
                        return str(is_reactions_AB_CD), one_bond_dict

    return str(is_reactions_AB_CD), one_bond_dict

class FindConcertedReactions:
    def __init__(self, entries_list, name):
        """
        class for finding concerted reactions
        Args:
        :param entries_list, entries_list = [MoleculeEntry]
        :param name: name for saving various dicts.
        """
        self.entries_list = entries_list
        self.name = name

        return

    def find_concerted_candidates(self):
        '''
        Find concerted reaction candidates by finding reactant-product pairs that match the stoichiometry.
        Args:
        :param entries: ReactionNetwork(input_entries).entries_list, entries_list = [MoleculeEntry]
        :param name: name for saving self.unique_mol_graph_dict.
        :return: self.concerted_rxns_to_determine: [['15_43', '19_43']]: [[str(reactants),str(products)]]
                 reactants and products are separated by "_".
                 The number correspond to the index of a mol_graph in self.unique_mol_graphs_new.
        '''
        print("Finding concerted reaction candidates!", flush=True)
        self.unique_mol_graphs_new = []
        # For duplicate mol graphs, create a map between later species with former ones
        # Only determine once for each unique mol_graph.
        self.unique_mol_graph_dict = {}

        for i in range(len(self.entries_list)):
            mol_graph = self.entries_list[i].mol_graph
            found = False
            for j in range(len(self.unique_mol_graphs_new)):
                new_mol_graph = self.unique_mol_graphs_new[j]
                if mol_graph.molecule.composition.alphabetical_formula == new_mol_graph.molecule.composition.alphabetical_formula and mol_graph.isomorphic_to(new_mol_graph):
                    found = True
                    self.unique_mol_graph_dict[i] = j
                    continue
            if not found:
                self.unique_mol_graph_dict[i] = len(self.unique_mol_graphs_new)
                self.unique_mol_graphs_new.append(mol_graph)
        #dumpfn(self.unique_mol_graph_dict, self.name + "_unique_mol_graph_map.json")
        # find all molecule pairs that satisfy the stoichiometry constraint
        stoi_list, species_same_stoi_dict = identify_same_stoi_mol_pairs(self.unique_mol_graphs_new)
        '''
        reac_prod_dict = {}
        for i, key in enumerate(species_same_stoi_dict.keys()):
            species_list = species_same_stoi_dict[key]
            new_species_list_reactant = []
            new_species_list_product = []
            for species in species_list:
                new_species_list_reactant.append(species)
                new_species_list_product.append(species)
            if new_species_list_reactant != [] and new_species_list_product != []:
                reac_prod_dict[key] = {'reactants': new_species_list_reactant,
                                            'products': new_species_list_product}
                
        self.concerted_rxns_to_determine = []
        for key in reac_prod_dict.keys():
            reactants = reac_prod_dict[key]['reactants']
            products = reac_prod_dict[key]['products']
            for j in range(len(reactants)):
                reac = reactants[j]
                for k in range(len(products)):
                    prod = products[k]
                    if k <= j:
                        continue
                    else:
                        self.concerted_rxns_to_determine.append([reac, prod])'''
        self.concerted_rxns_to_determine = []
        for i, key in enumerate(species_same_stoi_dict.keys()):
            species_list = species_same_stoi_dict[key]
            new_species_list_reactant = []
            new_species_list_product = []
            for species in species_list:
                new_species_list_reactant.append(species)
                new_species_list_product.append(species)
            if new_species_list_reactant != [] and new_species_list_product != []:
                reactants = new_species_list_reactant
                products = new_species_list_product
                for j in range(len(reactants)):
                    reac = reactants[j]
                    for k in range(len(products)):
                        prod = products[k]
                        if k <= j:
                            continue
                        else:
                            self.concerted_rxns_to_determine.append([reac, prod])

        print('number of concerted candidates:',len(self.concerted_rxns_to_determine), flush=True)
        dumpfn(self.concerted_rxns_to_determine, 'concerted_candidates.json')

        return

    def find_concerted_candidates_ABC_DE(self):
        '''
        Find concerted reaction candidates by finding reactant-product pairs that match the stoichiometry,
        for some A + B + C -> D + E reactions with single Li/H/F in A/B/C.
        This adds on to the previous function.
        Args:
        :param entries: ReactionNetwork(input_entries).entries_list, entries_list = [MoleculeEntry]
        :param name: name for saving self.unique_mol_graph_dict.
        :return: self.concerted_rxns_to_determine: [['15_43', '19_43']]: [[str(reactants),str(products)]]
                 reactants and products are separated by "_".
                 The number correspond to the index of a mol_graph in self.unique_mol_graphs_new.
        '''
        print("Finding concerted reaction candidates!", flush=True)
        self.unique_mol_graphs_new = []
        # For duplicate mol graphs, create a map between later species with former ones
        # Only determine once for each unique mol_graph.
        self.unique_mol_graph_dict = {}

        for i in range(len(self.entries_list)):
            mol_graph = self.entries_list[i].mol_graph
            found = False
            for j in range(len(self.unique_mol_graphs_new)):
                new_mol_graph = self.unique_mol_graphs_new[j]
                if mol_graph.molecule.composition.alphabetical_formula == new_mol_graph.molecule.composition.alphabetical_formula and mol_graph.isomorphic_to(
                        new_mol_graph):
                    found = True
                    self.unique_mol_graph_dict[i] = j
                    continue
            if not found:
                self.unique_mol_graph_dict[i] = len(self.unique_mol_graphs_new)
                self.unique_mol_graphs_new.append(mol_graph)
        # dumpfn(self.unique_mol_graph_dict, self.name + "_unique_mol_graph_map.json")
        # find all molecule pairs that satisfy the stoichiometry constraint
        stoi_list, species_same_stoi_dict = identify_same_stoi_mol_pairs_ABC_DE(self.unique_mol_graphs_new)

        self.concerted_rxns_to_determine = []
        for i, key in enumerate(species_same_stoi_dict.keys()):
            species_list = species_same_stoi_dict[key]
            new_species_list_reactant = []
            new_species_list_product = []
            for species in species_list:
                new_species_list_reactant.append(species)
                new_species_list_product.append(species)
            if new_species_list_reactant != [] and new_species_list_product != []:
                reactants = new_species_list_reactant
                products = new_species_list_product
                for j in range(len(reactants)):
                    reac = reactants[j]
                    for k in range(len(products)):
                        prod = products[k]
                        if k <= j:
                            continue
                        else:
                            split_reac = reac.split('_')
                            split_prod = prod.split('_')
                            if (len(split_reac) == 2 and len(split_prod) == 3) or (len(split_reac) == 3 and len(split_prod) == 2):
                                if len(split_reac) == 2 and len(split_prod) == 3:
                                    split_reac, split_prod = split_prod, split_reac
                                if (split_prod[0] in split_reac) or (split_prod[1] in split_reac):
                                    continue
                                else:
                                    self.concerted_rxns_to_determine.append([reac, prod])

        print('number of ABC_DE concerted candidates:', len(self.concerted_rxns_to_determine), flush=True)
        dumpfn(self.concerted_rxns_to_determine, 'concerted_candidates_ABC_DE.json')

        return

    def find_concerted_break2_form2(self, args):
        '''
        Determine whether one reaction in self.concerted_rxns_to_determine is a <=2 bond break, <=2 bond formation concerted reaction.
        Note that if a reaction is elementary (in class "RedoxReaction", "IntramolSingleBondChangeReaction", "IntermolecularReaction",
        "CoordinationBondChangeReaction"), it is also considered concerted. It has to be removed later on in the ReactionNetwork class.

        :param index: Index in self.concerted_rxns_to_determine
        :return: valid_reactions:[['15_43', '19_43']]: [[str(reactants),str(products)]]
                 reactants and products are separated by "_".
                 The number correspond to the index of a mol_graph in self.unique_mol_graphs_new.
        '''
        index, restart = args[0],args[1]
        print('current index:', index, flush=True)
        valid_reactions = []

        reac = self.concerted_rxns_to_determine[index][0]
        prod = self.concerted_rxns_to_determine[index][1]

        print('reactant:', reac, flush=True)
        print('product:', prod, flush=True)
        if restart and [reac, prod] in self.loaded_valid_reactions:
            valid_reactions.append([reac, prod])
            print('found!', flush=True)
            output = "valid_reactions_break2_form2"
            with open(output, 'a+') as f:
                rxn = [reac, prod]
                f.write(str(rxn)+'\n')
            return valid_reactions
        if restart and [reac, prod] in self.loaded_invalid_reactions:
            print('found!', flush=True)
            output_not_concerted = "invalid_reactions_break2_form2"
            with open(output_not_concerted, 'a+') as f:
                rxn = [reac, prod]
                f.write(str(rxn)+'\n')
            return valid_reactions

        split_reac = reac.split('_')
        split_prod = prod.split('_')
        found = False
        if (len(split_reac) == 1 and len(split_prod) == 1):
            mol_graph1 = self.unique_mol_graphs_new[int(split_reac[0])]
            mol_graph2 = self.unique_mol_graphs_new[int(split_prod[0])]
            if identify_self_reactions(mol_graph1, mol_graph2):
                if [reac, prod] not in valid_reactions:
                    found = True
                    valid_reactions.append([reac, prod])
        elif (len(split_reac) == 2 and len(split_prod) == 1):
            assert split_prod[0] not in split_reac
            mol_graphs1 = [self.unique_mol_graphs_new[int(split_reac[0])],
                           self.unique_mol_graphs_new[int(split_reac[1])]]
            mol_graphs2 = [self.unique_mol_graphs_new[int(split_prod[0])]]
            if identify_reactions_AB_C(mol_graphs1, mol_graphs2):
                if [reac, prod] not in valid_reactions:
                    found = True
                    valid_reactions.append([reac, prod])
        elif (len(split_reac) == 1 and len(split_prod) == 2):
            mol_graphs1 = [self.unique_mol_graphs_new[int(split_prod[0])],
                           self.unique_mol_graphs_new[int(split_prod[1])]]
            mol_graphs2 = [self.unique_mol_graphs_new[int(split_reac[0])]]
            if identify_reactions_AB_C(mol_graphs1, mol_graphs2):
                if [reac, prod] not in valid_reactions:
                    found = True
                    valid_reactions.append([reac, prod])
        elif (len(split_reac) == 2 and len(split_prod) == 2):
            # self reaction
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
                mol_graph1 = self.unique_mol_graphs_new[int(new_split_reac)]
                mol_graph2 = self.unique_mol_graphs_new[int(new_split_prod)]
                if identify_self_reactions(mol_graph1, mol_graph2):
                    if [new_split_reac, new_split_prod] not in valid_reactions:
                        found = True
                        valid_reactions.append([new_split_reac, new_split_prod])
            # A + B -> C + D
            else:
                mol_graphs1 = [self.unique_mol_graphs_new[int(split_reac[0])],
                               self.unique_mol_graphs_new[int(split_reac[1])]]
                mol_graphs2 = [self.unique_mol_graphs_new[int(split_prod[0])],
                               self.unique_mol_graphs_new[int(split_prod[1])]]
                if identify_reactions_AB_CD(mol_graphs1, mol_graphs2):
                    if [reac, prod] not in valid_reactions:
                        found = True
                        valid_reactions.append([reac, prod])
        #output = "valid_reactions-{}".format(pathos.helpers.mp.currentProcess().getPid())
        output = "valid_reactions_break2_form2"
        output_not_concerted = "invalid_reactions_break2_form2"
        if found:
            with open(output, 'a+') as f:
                rxn = [reac, prod]
                f.write(str(rxn)+'\n')
        elif not found:
            with open(output_not_concerted, 'a+') as f:
                rxn = [reac, prod]
                f.write(str(rxn) + '\n')
        return valid_reactions

    def find_concerted_break2_form2_ABC_DE(self, args):
        '''
        Determine whether one reaction in self.concerted_rxns_to_determine is a <=2 bond break, <=2 bond formation concerted reaction,
        only for some A + B + C -> D + E reactions with single Li/H/F in A/B/C.
        This adds on to the previous function.

        :param index: Index in self.concerted_rxns_to_determine
        :return: valid_reactions:[['15_43', '19_43']]: [[str(reactants),str(products)]]
                 reactants and products are separated by "_".
                 The number correspond to the index of a mol_graph in self.unique_mol_graphs_new.
        '''
        index, restart = args[0],args[1]
        print('current index:', index, flush=True)
        valid_reactions = []

        reac = self.concerted_rxns_to_determine[index][0]
        prod = self.concerted_rxns_to_determine[index][1]

        print('reactant:', reac, flush=True)
        print('product:', prod, flush=True)
        if restart and [reac, prod] in self.loaded_valid_reactions:
            valid_reactions.append([reac, prod])
            print('found!', flush=True)
            output = "valid_reactions_break2_form2_ABC_DE"
            with open(output, 'a+') as f:
                rxn = [reac, prod]
                f.write(str(rxn)+'\n')
            return valid_reactions
        if restart and [reac, prod] in self.loaded_invalid_reactions:
            print('found!', flush=True)
            output_not_concerted = "invalid_reactions_break2_form2_ABC_DE"
            with open(output_not_concerted, 'a+') as f:
                rxn = [reac, prod]
                f.write(str(rxn)+'\n')
            return valid_reactions

        split_reac = reac.split('_')
        split_prod = prod.split('_')
        found = False
        if (len(split_reac) == 2 and len(split_prod) == 3):
            split_reac, split_prod = split_prod, split_reac
        if (len(split_reac) == 3 and len(split_prod) == 2):
            if (split_prod[0] in split_reac) or (split_prod[1] in split_reac):
                # should be taken care of by general break2-form2 reactions with <=2 reactants and products.
                pass
            else:
                mol_graphs1 = [self.unique_mol_graphs_new[int(split_reac[0])],
                               self.unique_mol_graphs_new[int(split_reac[1])],
                               self.unique_mol_graphs_new[int(split_reac[2])]]
                mol_graphs2 = [self.unique_mol_graphs_new[int(split_prod[0])],
                               self.unique_mol_graphs_new[int(split_prod[1])]]
                if identify_reactions_ABC_DE(mol_graphs1, mol_graphs2):
                    if [reac, prod] not in valid_reactions:
                        found = True
                        valid_reactions.append([reac, prod])

        #output = "valid_reactions-{}".format(pathos.helpers.mp.currentProcess().getPid())
        output = "valid_reactions_break2_form2_ABC_DE"
        output_not_concerted = "invalid_reactions_break2_form2_ABC_DE"
        if found:
            with open(output, 'a+') as f:
                rxn = [reac, prod]
                f.write(str(rxn)+'\n')
        elif not found:
            with open(output_not_concerted, 'a+') as f:
                rxn = [reac, prod]
                f.write(str(rxn) + '\n')
        return valid_reactions

    def find_concerted_break1_form1(self, args):
        '''
        Determine whether one reaction in self.concerted_rxns_to_determine is a <=1 bond break, <=1 bond formation concerted reaction.
        Note that if a reaction is elementary (in class "RedoxReaction", "IntramolSingleBondChangeReaction", "IntermolecularReaction",
        "CoordinationBondChangeReaction"), it is also considered concerted. It has to be removed later on in the ReactionNetwork class.

        :param index: Index in self.concerted_rxns_to_determine
        :return: valid_reactions:[['15_43', '19_43']]: [[str(reactants),str(products)]]
                 reactants and products are separated by "_".
                 The number correspond to the index of a mol_graph in self.unique_mol_graphs_new.
        '''
        index, restart = args[0],args[1]
        print('current index:',index, flush=True)

        valid_reactions = []
        reac = self.concerted_rxns_to_determine[index][0]
        prod = self.concerted_rxns_to_determine[index][1]

        print('reactant:', reac, flush=True)
        print('product:', prod, flush=True)
        if restart and [reac, prod] in self.loaded_valid_reactions:
            valid_reactions.append([reac, prod])
            print('found!', flush=True)
            output = "valid_reactions_break1_form1"
            with open(output, 'a+') as f:
                rxn = [reac, prod]
                f.write(str(rxn)+'\n')
            return valid_reactions
        if restart and [reac, prod] in self.loaded_invalid_reactions:
            print('found!',flush=True)
            output_not_concerted = "invalid_reactions_break1_form1"
            with open(output_not_concerted, 'a+') as f:
                rxn = [reac, prod]
                f.write(str(rxn)+'\n')
            return valid_reactions

        split_reac = reac.split('_')
        split_prod = prod.split('_')
        found = False
        if (len(split_reac) == 1 and len(split_prod) == 1):
            mol_graph1 = self.unique_mol_graphs_new[int(split_reac[0])]
            mol_graph2 = self.unique_mol_graphs_new[int(split_prod[0])]
            if identify_self_reactions(mol_graph1, mol_graph2):
                if [reac, prod] not in valid_reactions:
                    found = True
                    valid_reactions.append([reac, prod])
        elif (len(split_reac) == 2 and len(split_prod) == 1):
            assert split_prod[0] not in split_reac
            mol_graphs1 = [self.unique_mol_graphs_new[int(split_reac[0])],
                           self.unique_mol_graphs_new[int(split_reac[1])]]
            mol_graphs2 = [self.unique_mol_graphs_new[int(split_prod[0])]]
            if identify_reactions_AB_C_break1_form1(mol_graphs1, mol_graphs2):
                if [reac, prod] not in valid_reactions:
                    found = True
                    valid_reactions.append([reac, prod])
        elif (len(split_reac) == 1 and len(split_prod) == 2):
            mol_graphs1 = [self.unique_mol_graphs_new[int(split_prod[0])],
                           self.unique_mol_graphs_new[int(split_prod[1])]]
            mol_graphs2 = [self.unique_mol_graphs_new[int(split_reac[0])]]
            if identify_reactions_AB_C_break1_form1(mol_graphs1, mol_graphs2):
                if [reac, prod] not in valid_reactions:
                    found = True
                    valid_reactions.append([reac, prod])
        elif (len(split_reac) == 2 and len(split_prod) == 2):
            # self reaction
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
                mol_graph1 = self.unique_mol_graphs_new[int(new_split_reac)]
                mol_graph2 = self.unique_mol_graphs_new[int(new_split_prod)]
                if identify_self_reactions(mol_graph1, mol_graph2):
                    if [new_split_reac, new_split_prod] not in valid_reactions:
                        found = True
                        valid_reactions.append([new_split_reac, new_split_prod])
            # A + B -> C + D
            else:
                mol_graphs1 = [self.unique_mol_graphs_new[int(split_reac[0])],
                               self.unique_mol_graphs_new[int(split_reac[1])]]
                mol_graphs2 = [self.unique_mol_graphs_new[int(split_prod[0])],
                               self.unique_mol_graphs_new[int(split_prod[1])]]
                if identify_reactions_AB_CD_break1_form1(mol_graphs1, mol_graphs2):
                    if [reac, prod] not in valid_reactions:
                        found = True
                        valid_reactions.append([reac, prod])
        output = "valid_reactions_break1_form1"
        output_not_concerted = "invalid_reactions_break1_form1"
        #output = "valid_reactions-{}".format(pathos.helpers.mp.currentProcess().getPid())
        if found:
            with open(output, 'a+') as f:
                rxn = [reac, prod]
                f.write(str(rxn)+'\n')
        elif not found:
            with open(output_not_concerted, 'a+') as f:
                rxn = [reac, prod]
                f.write(str(rxn) + '\n')
        return valid_reactions

    def find_concerted_multiprocess(self, num_processors, reaction_type="break2_form2", restart=False):
        '''
        Use multiprocessing to determine concerted reactions in parallel.
        Args:
        :param num_processors:
        :param reaction_type: Can choose from "break2_form2" and "break1_form1"
        :return: self.valid_reactions:[['15_43', '19_43']]: [[str(reactants),str(products)]]
                 reactants and products are separated by "_".
                 The number correspond to the index of a mol_graph in self.unique_mol_graphs_new.
        '''
        if restart:
            self.loaded_valid_reactions = []
            self.loaded_invalid_reactions = []
            valid_reactions_to_load = "valid_reactions_" + reaction_type
            invalid_reactions_to_load = "invalid_reactions_" + reaction_type
            f = open(valid_reactions_to_load, "r")
            contents = f.readlines()
            f.close()
            self.loaded_valid_reactions = []
            for i,content in enumerate(contents):
                new_content = content.replace('\n', '').replace("'", "").strip('][').split(', ')
                print('new_content:',new_content, flush=True)
                self.loaded_valid_reactions.append(new_content)
                self.concerted_rxns_to_determine.remove(new_content)

            f = open(invalid_reactions_to_load, "r")
            contents = f.readlines()
            f.close()
            for i,content in enumerate(contents):
                new_content = content.replace('\n', '').replace("'", "").strip('][').split(', ')
                print('new_content:',new_content, flush=True)
                self.loaded_invalid_reactions.append(new_content)
                self.concerted_rxns_to_determine.remove(new_content)

        print("Finding concerted reactions!")
        if reaction_type == "break2_form2":
            func = self.find_concerted_break2_form2
            print("Reaction type: break2 form2", flush=True)
        elif reaction_type == "break1_form1":
            func = self.find_concerted_break1_form1
            print("Reaction type: break1 form1",flush=True)
        elif reaction_type == "break2_form2_ABC_DE":
            func = self.find_concerted_break2_form2_ABC_DE
            print("Reaction type: break2 form2 ABC_DE",flush=True)


        from pathos.multiprocessing import ProcessingPool as Pool
        nums = list(np.arange(len(self.concerted_rxns_to_determine)))
        args = [(i, restart) for i in nums]
        pool = Pool(num_processors)
        results = pool.map(func, args)
        self.valid_reactions = []
        for i in range(len(results)):
            valid_reactions = results[i]
            self.valid_reactions += valid_reactions
        if restart:
            self.valid_reactions += self.loaded_valid_reactions
        #dumpfn(self.valid_reactions, name + "_valid_concerted_rxns.json")
        return

    def find_concerted_multiprocess_no_saving(self, num_processors, reaction_type="break2_form2"):
        '''
        Use multiprocessing to determine concerted reactions in parallel.
        Args:
        :param num_processors:
        :param reaction_type: Can choose from "break2_form2" and "break1_form1"
        :return: self.valid_reactions:[['15_43', '19_43']]: [[str(reactants),str(products)]]
                 reactants and products are separated by "_".
                 The number correspond to the index of a mol_graph in self.unique_mol_graphs_new.
        '''
        print("Finding concerted reactions!",flush=True)
        if reaction_type == "break2_form2":
            func = self.find_concerted_break2_form2
            print("Reaction type: break2 form2",flush=True)
        elif reaction_type == "break1_form1":
            func = self.find_concerted_break1_form1
            print("Reaction type: break1 form1",flush=True)
        from pathos.multiprocessing import ProcessingPool as Pool
        nums = list(np.arange(len(self.concerted_rxns_to_determine)))
        args = [(i) for i in nums]
        pool = Pool(num_processors)
        results = pool.map(func, args)
        # self.valid_reactions = []
        # for i in range(len(results)):
        #     valid_reactions = results[i]
        #     self.valid_reactions += valid_reactions
        #dumpfn(self.valid_reactions, name + "_valid_concerted_rxns.json")
        return

    def get_final_concerted_reactions(self, name, num_processors, reaction_type="break2_form2", restart=False):
        '''
        This is for getting the final set of concerted reactions: entry index corresponds to the index in self.entries_list.
        Args:
        :param name: name for saving self.valid_reactions. self.valid_reactions has the following form:
               [["0_1", "6_46"]]: [[str(reactants), str(products)]] reactants and products are separated by "_".
               The number correspond to the index of a mol_graph in self.unique_mol_graphs_new.
        :param num_processors:
        :param reaction_type: Can choose from "break2_form2" and "break1_form1"

        :return: [['15_43', '19_43']]: [[str(reactants),str(products)]]
                 reactants and products are separated by "_".
                 The number correspond to the index of a mol_graph in self.entries_list.
        '''
        if reaction_type == 'break2_form2_ABC_DE':
            self.find_concerted_candidates_ABC_DE()
        else:
            self.find_concerted_candidates()
        self.find_concerted_multiprocess(num_processors, reaction_type, restart=restart)
        print("Summarizing concerted reactions!",flush=True)
        self.final_concerted_reactions = []
        if reaction_type == 'break2_form2_ABC_DE':
            for i in range(len(self.valid_reactions)):
                rxn = self.valid_reactions[i]
                reactant_nodes = rxn[0].split('_')
                product_nodes = rxn[1].split('_')
                reactant_candidates = []
                product_candidates = []
                for reac in reactant_nodes:
                    reac_cands = []
                    for map_key in self.unique_mol_graph_dict.keys():
                        if self.unique_mol_graph_dict[map_key] == int(reac):
                            reac_cands.append(map_key)
                    reactant_candidates.append(reac_cands)
                for prod in product_nodes:
                    prod_cands = []
                    for map_key in self.unique_mol_graph_dict.keys():
                        if self.unique_mol_graph_dict[map_key] == int(prod):
                            prod_cands.append(map_key)
                    product_candidates.append(prod_cands)
                print('reactant candidates:', reactant_candidates,flush=True)
                print('product candidates:', product_candidates,flush=True)
                if len(reactant_candidates) == 2 and len(product_candidates) == 3:
                    reactant_candidates, product_candidates = product_candidates, reactant_candidates
                if len(reactant_candidates) == 3 and len(product_candidates) == 2:
                    for j in reactant_candidates[0]:
                        for k in reactant_candidates[1]:
                            for p in reactant_candidates[2]:
                                for m in product_candidates[0]:
                                    for n in product_candidates[1]:
                                        reactant_index_list = [int(j),int(k),int(p)]
                                        reactant_index_list.sort()
                                        reactant_name = str(reactant_index_list[0]) + '_' + str(reactant_index_list[1]) + '_' + str(reactant_index_list[2])
                                        if int(m) <= int(n):
                                            product_name = str(m) + '_' + str(n)
                                        else:
                                            product_name = str(n) + '_' + str(m)
                                        self.final_concerted_reactions.append([reactant_name, product_name])
            dumpfn(self.final_concerted_reactions, name + '_concerted_rxns_'+reaction_type+'.json')

        else:
            for i in range(len(self.valid_reactions)):
                rxn = self.valid_reactions[i]
                reactant_nodes = rxn[0].split('_')
                product_nodes = rxn[1].split('_')
                reactant_candidates = []
                product_candidates = []
                for reac in reactant_nodes:
                    reac_cands = []
                    for map_key in self.unique_mol_graph_dict.keys():
                        if self.unique_mol_graph_dict[map_key] == int(reac):
                            reac_cands.append(map_key)
                    reactant_candidates.append(reac_cands)
                for prod in product_nodes:
                    prod_cands = []
                    for map_key in self.unique_mol_graph_dict.keys():
                        if self.unique_mol_graph_dict[map_key] == int(prod):
                            prod_cands.append(map_key)
                    product_candidates.append(prod_cands)
                print('reactant candidates:',reactant_candidates,flush=True)
                print('product candidates:',product_candidates,flush=True)

                if len(reactant_candidates) == 1 and len(product_candidates) == 1:
                    for j in reactant_candidates[0]:
                        for k in product_candidates[0]:
                            self.final_concerted_reactions.append([str(j),str(k)])

                elif len(reactant_candidates) == 2 and len(product_candidates) == 1:
                    for j in reactant_candidates[0]:
                        for k in reactant_candidates[1]:
                            for m in product_candidates[0]:
                                if int(j) <= int(k):
                                    reactant_name = str(j) + '_' + str(k)
                                else:
                                    reactant_name = str(k) + '_' + str(j)
                                self.final_concerted_reactions.append([reactant_name,str(m)])

                elif len(reactant_candidates) == 1 and len(product_candidates) == 2:
                    for j in reactant_candidates[0]:
                        for m in product_candidates[0]:
                            for n in product_candidates[1]:
                                if int(m) <= int(n):
                                    product_name = str(m) + '_' + str(n)
                                else:
                                    product_name = str(n) + '_' + str(m)
                                self.final_concerted_reactions.append([str(j),product_name])

                elif len(reactant_candidates) == 2 and len(product_candidates) == 2:
                    for j in reactant_candidates[0]:
                        for k in reactant_candidates[1]:
                            for m in product_candidates[0]:
                                for n in product_candidates[1]:
                                    if int(j) <= int(k):
                                        reactant_name = str(j) + '_' + str(k)
                                    else:
                                        reactant_name = str(k) + '_' + str(j)
                                    if int(m) <= int(n):
                                        product_name = str(m) + '_' + str(n)
                                    else:
                                        product_name = str(n) + '_' + str(m)
                                    self.final_concerted_reactions.append([reactant_name,product_name])
            dumpfn(self.final_concerted_reactions, name + '_concerted_rxns_'+reaction_type+'.json')
        return self.final_concerted_reactions

if __name__ == '__main__':
    mol = Molecule.from_file('/Users/xiaowei_xie/PycharmProjects/electrolyte/recombination_final_2/LiEC_LPF6_water_recomb_mols/175.xyz')
    B = MoleculeGraph.with_local_env_strategy(mol, OpenBabelNN(),
                                                           reorder=False,
                                                           extend_structure=False)

    mol1 = Molecule.from_file('/Users/xiaowei_xie/PycharmProjects/electrolyte/recombination_final_2/LiEC_LPF6_water_recomb_mols/114.xyz')
    A = MoleculeGraph.with_local_env_strategy(mol1, OpenBabelNN(),
                                                           reorder=False,
                                                           extend_structure=False)
    mol2 = Molecule.from_file('/Users/xiaowei_xie/PycharmProjects/electrolyte/recombination_final_2/LiEC_LPF6_water_recomb_mols/144.xyz')
    C =  MoleculeGraph.with_local_env_strategy(mol2, OpenBabelNN(),
                                                           reorder=False,
                                                           extend_structure=False)
    mol3 = Molecule.from_file('/Users/xiaowei_xie/PycharmProjects/electrolyte/recombination_final_2/LiEC_LPF6_water_recomb_mols/158.xyz')
    D =  MoleculeGraph.with_local_env_strategy(mol3, OpenBabelNN(),
                                                           reorder=False,
                                                           extend_structure=False)

    frags_A_two_step = break_two_bonds_in_one_mol(A)
    frags_D_two_step = break_two_bonds_in_one_mol(D)

    # break one mol two steps (scenario 2)
    for item_A in frags_A_two_step:
            for item_D in frags_D_two_step:
                print(check_same_mol_graphs(item_A + [B], [C] + item_D))






