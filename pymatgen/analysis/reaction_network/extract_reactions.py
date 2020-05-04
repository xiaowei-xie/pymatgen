from pymatgen.analysis.graphs import MoleculeGraph, MolGraphSplitError
from atomate.qchem.database import QChemCalcDb
from pymatgen import Molecule
from pymatgen.analysis.reaction_network.fragment_recombination_util import *
import copy
from itertools import combinations_with_replacement, combinations
from pymatgen.analysis.fragmenter import open_ring

def convert_atomic_numbers_to_stoi_dict(atomic_numbers):
    '''

    :param atomic_numbers: a list of atomic numbers
    :return: {'Li':1, '110':0,'C':3,...} zero padding for non-existing elements
    '''
    atomic_num_to_element = {1:'H',3:'Li',6:'C',8:'O',9:'F',15:'110'}
    elements = ['H','Li','C','O','F','110']
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
                     '110': stoi_dict1['110'] + stoi_dict2['110'], 'F':stoi_dict1['F'] + stoi_dict2['F']}
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

def identify_self_reactions_old(mol_graph1, mol_graph2):
    '''
    # Check if we can get to the same two fragments by breaking one bond on each of the mols.
    # Different ring closure if they both break a bond in the ring
    # One fragment comes from breaking a bond in the ring in another fragment
    # Not considering breaking two bonds and forming two bonds in one mol???
    :param mol_graph1:
    :param mol_graph2:
    :return: bool: is self reaction or not
    '''
    is_self_reaction = False
    if len(mol_graph1.graph.edges) != 0:
        for edge1 in mol_graph1.graph.edges:
            bond1 = [(edge1[0], edge1[1])]
            try:
                frags1 = mol_graph1.split_molecule_subgraphs(bond1, allow_reverse=True)
                if len(mol_graph2.graph.edges) != 0:
                    for edge2 in mol_graph2.graph.edges:
                        bond2 = [(edge2[0], edge2[1])]
                        try:
                            frags2 = mol_graph2.split_molecule_subgraphs(bond2, allow_reverse=True)
                            if frags1[0].molecule.composition.alphabetical_formula == frags2[0].molecule.composition.alphabetical_formula and \
                                    frags1[1].molecule.composition.alphabetical_formula == frags2[1].molecule.composition.alphabetical_formula:
                                if (frags1[0].isomorphic_to(frags2[0]) and frags1[1].isomorphic_to(frags2[1])):
                                    is_self_reaction = True
                                    return is_self_reaction
                            elif frags1[0].molecule.composition.alphabetical_formula == frags2[1].molecule.composition.alphabetical_formula and \
                                    frags1[1].molecule.composition.alphabetical_formula == frags2[0].molecule.composition.alphabetical_formula:
                                if (frags1[0].isomorphic_to(frags2[1]) and frags1[1].isomorphic_to(frags2[0])):
                                    is_self_reaction = True
                                    return is_self_reaction

                        except MolGraphSplitError:
                            frag2 = open_ring(mol_graph2, bond2, 10000)
                            if frag2.molecule.composition.alphabetical_formula == mol_graph1.composition.alphabetical_formula:
                                if frag2.isomorphic_to(mol_graph1):
                                    is_self_reaction = True
                                    return is_self_reaction

            except MolGraphSplitError:
                frag1 = open_ring(mol_graph1, bond1, 10000)
                if len(mol_graph2.graph.edges) != 0:
                    for edge2 in mol_graph2.graph.edges:
                        bond2 = [(edge2[0], edge2[1])]
                        try:
                            frag2 = open_ring(mol_graph2, bond2, 10000)
                            if frag1.molecule.composition.alphabetical_formula == frag2.composition.alphabetical_formula:
                                if frag1.isomorphic_to(frag2):
                                    is_self_reaction = True
                                    return is_self_reaction
                        except:
                            if frag1.molecule.composition.alphabetical_formula == mol_graph2.composition.alphabetical_formula:
                                if frag1.isomorphic_to(mol_graph2):
                                    is_self_reaction = True
                                    return is_self_reaction
    return is_self_reaction

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
                        print('AB each once!')
                        return is_reactions_AB_C

    # A or B break twice, the other intact, C break twice
    frags_A_two_step = break_two_bonds_in_one_mol(A)
    frags_B_two_step = break_two_bonds_in_one_mol(B)

    for item_A in frags_A_two_step:
            for item_C in frags_C_two_step:
                    if check_same_mol_graphs(item_A + [B], item_C):
                        is_reactions_AB_C = True
                        print('AC twice, B intact!')
                        return is_reactions_AB_C

    for item_B in frags_B_two_step:
            for item_C in frags_C_two_step:
                    if check_same_mol_graphs([A] + item_B, item_C):
                        is_reactions_AB_C = True
                        print('BC twice, A intact!')
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
                print('A once, C once!')
                return is_reactions_AB_C

    # B C break once
    for item_B in frags_B_one_step:
        for item_C in frags_C_one_step:
            if check_same_mol_graphs(item_B + [A], item_C):
                is_reactions_AB_C = True
                print('B once, C once!')
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
                        print('AB each once!')
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
                        print('AC twice, B intact!')
                        return str(is_reactions_AB_C), one_bond_dict, two_bond_dict

    for item_B in frags_B_two_step:
            for item_C in frags_C_two_step:
                    if check_same_mol_graphs([A] + item_B, item_C):
                        is_reactions_AB_C = True
                        print('BC twice, A intact!')
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
            print('AB each once!')
    return str(is_reactions_AB_C), one_bond_dict


def identify_reactions_AB_C_old(mol_graphs1, mol_graphs2):
    '''
    A + B -> C type reactions
    1. single bond breakage, break C once and check if the two fragments correspond to A and B
    2. break C twice,
       if two fragments: check if is A and B;
       if 3 fragments: check if A or B or A-RO or B-RO in the 3 fragments, then check if breaking the other one (A or B)
                       can create the other 2 fragments
    3. Is this enough???
    :param mol_graphs1: 2 components A and B
    :param mol_graphs2: 1 component C
    :return: True or False
    '''
    is_AB_C_reaction = False
    assert len(mol_graphs1) == 2 and len(mol_graphs2) == 1
    if len(mol_graphs2[0].graph.edges) != 0:
        for edge in mol_graphs2[0].graph.edges:
            bond = [(edge[0], edge[1])]
            try:
                #break C only once
                frags = mol_graphs2[0].split_molecule_subgraphs(bond, allow_reverse=True)
                if check_same_mol_graphs(frags, mol_graphs1):
                    is_AB_C_reaction = True
                    return is_AB_C_reaction
            except MolGraphSplitError:
                continue

        all_possible_fragments = break_two_bonds_in_one_mol(mol_graphs2[0])
        for frags in all_possible_fragments:
            if len(frags) == 2:
                if check_same_mol_graphs(frags, mol_graphs1):
                    is_AB_C_reaction = True
                    return is_AB_C_reaction
            else:
                # len(frags) == 3
                if check_in_list(mol_graphs1[0],frags) or \
                    any(check_in_list(test_mol_graph, frags) for test_mol_graph in open_ring_in_one_mol(mol_graphs1[0])):
                    if len(mol_graphs1[1].graph.edges) != 0:
                        for edge2 in mol_graphs1[1].graph.edges:
                            bond2 = [(edge2[0], edge2[1])]
                            try:
                                frags2 = mol_graphs1[1].split_molecule_subgraphs(bond2, allow_reverse=True)
                                if check_same_mol_graphs(frags2+[mol_graphs1[0]],frags):
                                    is_AB_C_reaction = True
                                    return is_AB_C_reaction
                            except MolGraphSplitError:
                                # Have to be able to break one bond and form two fragments
                                continue
                elif check_in_list(mol_graphs1[1],frags) or \
                    any(check_in_list(test_mol_graph, frags) for test_mol_graph in open_ring_in_one_mol(mol_graphs1[1])):
                    if len(mol_graphs1[0].graph.edges) != 0:
                        for edge2 in mol_graphs1[0].graph.edges:
                            bond2 = [(edge2[0], edge2[1])]
                            try:
                                frags2 = mol_graphs1[0].split_molecule_subgraphs(bond2, allow_reverse=True)
                                if check_same_mol_graphs(frags2+[mol_graphs1[1]],frags):
                                    is_AB_C_reaction = True
                                    return is_AB_C_reaction
                            except MolGraphSplitError:
                                # Have to be able to break one bond and form two fragments
                                continue
    return is_AB_C_reaction

def identify_reactions_AB_CD_3_components(mol_graphs1, mol_graphs2):
    '''
    A + B -> A1 + A2 + B -> A1 + (A2 + B) == C + D or
    A + B -> A + B1 + B2 -> (A + B1) + B2 == C + D
    try break A first, one of the fragments has to be equivalent to C, then check if breaking D can create A2 and B
    then try break B, same strategy
    :param mol_graphs1: a list of 2 different mol graphs
    :param mol_graphs2: a list of 2 different mol graphs
    :return:
    '''
    is_3_components_reaction = False
    A = mol_graphs1[0]
    B = mol_graphs1[1]
    C = mol_graphs2[0]
    D = mol_graphs2[1]
    if len(A.graph.edges) != 0:
        for edge_A in A.graph.edges:
            bond_A = [(edge_A[0], edge_A[1])]
            try:
                frags_A = A.split_molecule_subgraphs(bond_A, allow_reverse=True)
                if is_equivalent(frags_A[0],C):
                    # D should be frags_A[1] + B
                    if len(D.graph.edges) != 0:
                        for edge_D in D.graph.edges:
                            bond_D = [(edge_D[0], edge_D[1])]
                            try:
                                frags_D = D.split_molecule_subgraphs(bond_D, allow_reverse=True)
                                if (is_equivalent(frags_D[0],frags_A[1]) and is_ring_isomorphic(frags_D[1], B)) or \
                                    (is_equivalent(frags_D[1], frags_A[1]) and is_ring_isomorphic(frags_D[0], B)) or \
                                    (is_equivalent(frags_D[0], B) and is_ring_isomorphic(frags_D[1], frags_A[1])) or \
                                        (is_equivalent(frags_D[1], B) and is_ring_isomorphic(frags_D[0], frags_A[1])):
                                    is_3_components_reaction = True
                                    return is_3_components_reaction
                            except:
                                # D must be able to be splitted into 2 fragments
                                continue
                elif is_equivalent(frags_A[1],C):
                    # D should be frags_A[0] + B
                    if len(D.graph.edges) != 0:
                        for edge_D in D.graph.edges:
                            bond_D = [(edge_D[0], edge_D[1])]
                            try:
                                frags_D = D.split_molecule_subgraphs(bond_D, allow_reverse=True)
                                if (is_equivalent(frags_D[0], frags_A[0]) and is_ring_isomorphic(frags_D[1], B)) or \
                                        (is_equivalent(frags_D[1], frags_A[0]) and is_ring_isomorphic(frags_D[0], B)) or \
                                        (is_equivalent(frags_D[0], B) and is_ring_isomorphic(frags_D[1], frags_A[0])) or \
                                        (is_equivalent(frags_D[1], B) and is_ring_isomorphic(frags_D[0], frags_A[0])):
                                    is_3_components_reaction = True
                                    return is_3_components_reaction
                            except:
                                # D must be able to be splitted into 2 fragments
                                continue

                elif is_equivalent(frags_A[1], D):
                    # C should be frags_A[0] + B
                    if len(C.graph.edges) != 0:
                        for edge_C in C.graph.edges:
                            bond_C = [(edge_C[0], edge_C[1])]
                            try:
                                frags_C = C.split_molecule_subgraphs(bond_C, allow_reverse=True)
                                if (is_equivalent(frags_C[0], frags_A[0]) and is_ring_isomorphic(frags_C[1], B)) or \
                                        (is_equivalent(frags_C[1], frags_A[0]) and is_ring_isomorphic(frags_C[0], B)) or \
                                        (is_equivalent(frags_C[0], B) and is_ring_isomorphic(frags_C[1], frags_A[0])) or \
                                        (is_equivalent(frags_C[1], B) and is_ring_isomorphic(frags_C[0], frags_A[0])):
                                    is_3_components_reaction = True
                                    return is_3_components_reaction
                            except:
                                # C must be able to be splitted into 2 fragments
                                continue

                elif is_equivalent(frags_A[0], D):
                    # C should be frags_A[1] + B
                    if len(C.graph.edges) != 0:
                        for edge_C in C.graph.edges:
                            bond_C = [(edge_C[0], edge_C[1])]
                            try:
                                frags_C = C.split_molecule_subgraphs(bond_C, allow_reverse=True)
                                if (is_equivalent(frags_C[0], frags_A[1]) and is_ring_isomorphic(frags_C[1], B)) or \
                                        (is_equivalent(frags_C[1], frags_A[1]) and is_ring_isomorphic(frags_C[0], B)) or \
                                        (is_equivalent(frags_C[0], B) and is_ring_isomorphic(frags_C[1], frags_A[1])) or \
                                        (is_equivalent(frags_C[1], B) and is_ring_isomorphic(frags_C[0], frags_A[1])):
                                    is_3_components_reaction = True
                                    return is_3_components_reaction
                            except:
                                # C must be able to be splitted into 2 fragments
                                continue

            except:
                # A must be able to break into 2 fragments to make sure there is one fragment present in C + D
                continue
    # same thing for B
    if len(B.graph.edges) != 0:
        for edge_B in B.graph.edges:
            bond_B = [(edge_B[0], edge_B[1])]
            try:
                frags_B = B.split_molecule_subgraphs(bond_B, allow_reverse=True)
                if is_equivalent(frags_B[0],C):
                    # D should be frags_B[1] + A
                    if len(D.graph.edges) != 0:
                        for edge_D in D.graph.edges:
                            bond_D = [(edge_D[0], edge_D[1])]
                            try:
                                frags_D = D.split_molecule_subgraphs(bond_D, allow_reverse=True)
                                if (is_equivalent(frags_D[0],frags_B[1]) and is_ring_isomorphic(frags_D[1], A)) or \
                                    (is_equivalent(frags_D[1], frags_B[1]) and is_ring_isomorphic(frags_D[0], A)) or \
                                    (is_equivalent(frags_D[0], A) and is_ring_isomorphic(frags_D[1], frags_B[1])) or \
                                        (is_equivalent(frags_D[1], A) and is_ring_isomorphic(frags_D[0], frags_B[1])):
                                    is_3_components_reaction = True
                                    return is_3_components_reaction
                            except:
                                # D must be able to be splitted into 2 fragments
                                continue
                elif is_equivalent(frags_B[1],C):
                    # D should be frags_B[0] + B
                    if len(D.graph.edges) != 0:
                        for edge_D in D.graph.edges:
                            bond_D = [(edge_D[0], edge_D[1])]
                            try:
                                frags_D = D.split_molecule_subgraphs(bond_D, allow_reverse=True)
                                if (is_equivalent(frags_D[0], frags_B[0]) and is_ring_isomorphic(frags_D[1], A)) or \
                                        (is_equivalent(frags_D[1], frags_B[0]) and is_ring_isomorphic(frags_D[0], A)) or \
                                        (is_equivalent(frags_D[0], A) and is_ring_isomorphic(frags_D[1], frags_B[0])) or \
                                        (is_equivalent(frags_D[1], A) and is_ring_isomorphic(frags_D[0], frags_B[0])):
                                    is_3_components_reaction = True
                                    return is_3_components_reaction
                            except:
                                # D must be able to be splitted into 2 fragments
                                continue

                elif is_equivalent(frags_B[1], D):
                    # C should be frags_B[0] + A
                    if len(C.graph.edges) != 0:
                        for edge_C in C.graph.edges:
                            bond_C = [(edge_C[0], edge_C[1])]
                            try:
                                frags_C = C.split_molecule_subgraphs(bond_C, allow_reverse=True)
                                if (is_equivalent(frags_C[0], frags_B[0]) and is_ring_isomorphic(frags_C[1], A)) or \
                                        (is_equivalent(frags_C[1], frags_B[0]) and is_ring_isomorphic(frags_C[0], A)) or \
                                        (is_equivalent(frags_C[0], A) and is_ring_isomorphic(frags_C[1], frags_B[0])) or \
                                        (is_equivalent(frags_C[1], A) and is_ring_isomorphic(frags_C[0], frags_B[0])):
                                    is_3_components_reaction = True
                                    return is_3_components_reaction
                            except:
                                # C must be able to be splitted into 2 fragments
                                continue

                elif is_equivalent(frags_B[0], D):
                    # C should be frags_B[1] + A
                    if len(C.graph.edges) != 0:
                        for edge_C in C.graph.edges:
                            bond_C = [(edge_C[0], edge_C[1])]
                            try:
                                frags_C = C.split_molecule_subgraphs(bond_C, allow_reverse=True)
                                if (is_equivalent(frags_C[0], frags_B[1]) and is_ring_isomorphic(frags_C[1], A)) or \
                                        (is_equivalent(frags_C[1], frags_B[1]) and is_ring_isomorphic(frags_C[0], A)) or \
                                        (is_equivalent(frags_C[0], A) and is_ring_isomorphic(frags_C[1], frags_B[1])) or \
                                        (is_equivalent(frags_C[1], A) and is_ring_isomorphic(frags_C[0], frags_B[1])):
                                    is_3_components_reaction = True
                                    return is_3_components_reaction
                            except:
                                # C must be able to be splitted into 2 fragments
                                continue

            except:
                # B must be able to break into 2 fragments to make sure there is one fragment present in C + D
                continue
    is_3_components_reaction = identify_reactions_AB_CD_3_components(mol_graphs2, mol_graphs1)
    return is_3_components_reaction

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
                        print('ABCD each once!')
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
                    print('break A twice, CD once!')
                    return is_reactions_AB_CD

    for item_B in frags_B_two_step:
        for item_C in frags_C_one_step:
            for item_D in frags_D_one_step:
                if check_same_mol_graphs([A] + item_B, item_C + item_D):
                    is_reactions_AB_CD = True
                    print('break B twice, CD once!')
                    return is_reactions_AB_CD

    for item_C in frags_C_two_step:
        for item_A in frags_A_one_step:
            for item_B in frags_B_one_step:
                if check_same_mol_graphs(item_A + item_B, item_C + [D]):
                    is_reactions_AB_CD = True
                    print('break C twice, AB once!')
                    return is_reactions_AB_CD

    for item_D in frags_D_two_step:
        for item_A in frags_A_one_step:
            for item_B in frags_B_one_step:
                if check_same_mol_graphs(item_A + item_B, [C] + item_D):
                    is_reactions_AB_CD = True
                    print('break D twice, AB once!')
                    return is_reactions_AB_CD

    # break two mol two steps (scenario 3)
    for item_A in frags_A_two_step:
        for item_C in frags_C_two_step:
            if check_same_mol_graphs(item_A + [B], item_C + [D]):
                is_reactions_AB_CD = True
                print('break AC twice, BD intact')
                return is_reactions_AB_CD

    for item_A in frags_A_two_step:
        for item_D in frags_D_two_step:
            if check_same_mol_graphs(item_A + [B], [C] + item_D):
                is_reactions_AB_CD = True
                print('break AD twice, BC intact')
                return is_reactions_AB_CD

    for item_B in frags_B_two_step:
        for item_C in frags_C_two_step:
            if check_same_mol_graphs([A] + item_B, item_C + [D]):
                is_reactions_AB_CD = True
                print('break BC twice, AD intact')
                return is_reactions_AB_CD

    for item_B in frags_B_two_step:
        for item_D in frags_D_two_step:
            if check_same_mol_graphs([A] + item_B, [C] + item_D):
                is_reactions_AB_CD = True
                print('break AC twice, BD intact')
                return is_reactions_AB_CD

    return is_reactions_AB_CD

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
                print('A once, C once!')
                return is_reactions_AB_CD

    # A D break once
    for item_A in frags_A_one_step:
        for item_D in frags_D_one_step:
            if check_same_mol_graphs(item_A + [B], [C] + item_D):
                is_reactions_AB_CD = True
                print('A once, D once!')
                return is_reactions_AB_CD

    # B C break once
    for item_B in frags_B_one_step:
        for item_C in frags_C_one_step:
            if check_same_mol_graphs(item_B + [A], item_C + [D]):
                is_reactions_AB_CD = True
                print('B once, C once!')
                return is_reactions_AB_CD

    # B D break once
    for item_B in frags_B_one_step:
        for item_D in frags_D_one_step:
            if check_same_mol_graphs(item_B + [A], [C] + item_D):
                is_reactions_AB_CD = True
                print('B once, D once!')
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
                        print('ABCD each once!')
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
                    print('break A twice, CD once!')
                    return str(is_reactions_AB_CD),one_bond_dict, two_bond_dict

    for item_B in frags_B_two_step:
        for item_C in frags_C_one_step:
            for item_D in frags_D_one_step:
                if check_same_mol_graphs([A] + item_B, item_C + item_D):
                    is_reactions_AB_CD = True
                    print('break B twice, CD once!')
                    return str(is_reactions_AB_CD), one_bond_dict, two_bond_dict

    for item_C in frags_C_two_step:
        for item_A in frags_A_one_step:
            for item_B in frags_B_one_step:
                if check_same_mol_graphs(item_A + item_B, item_C + [D]):
                    is_reactions_AB_CD = True
                    print('break C twice, AB once!')
                    return str(is_reactions_AB_CD), one_bond_dict, two_bond_dict

    for item_D in frags_D_two_step:
        for item_A in frags_A_one_step:
            for item_B in frags_B_one_step:
                if check_same_mol_graphs(item_A + item_B, [C] + item_D):
                    is_reactions_AB_CD = True
                    print('break D twice, AB once!')
                    return str(is_reactions_AB_CD), one_bond_dict, two_bond_dict

    # break two mol two steps (scenario 3)
    for item_A in frags_A_two_step:
        for item_C in frags_C_two_step:
            if check_same_mol_graphs(item_A + [B], item_C + [D]):
                is_reactions_AB_CD = True
                print('break AC twice, BD intact')
                return str(is_reactions_AB_CD), one_bond_dict, two_bond_dict

    for item_A in frags_A_two_step:
        for item_D in frags_D_two_step:
            if check_same_mol_graphs(item_A + [B], [C] + item_D):
                is_reactions_AB_CD = True
                print('break AD twice, BC intact')
                return str(is_reactions_AB_CD), one_bond_dict, two_bond_dict

    for item_B in frags_B_two_step:
        for item_C in frags_C_two_step:
            if check_same_mol_graphs([A] + item_B, item_C + [D]):
                is_reactions_AB_CD = True
                print('break BC twice, AD intact')
                return str(is_reactions_AB_CD), one_bond_dict, two_bond_dict

    for item_B in frags_B_two_step:
        for item_D in frags_D_two_step:
            if check_same_mol_graphs([A] + item_B, [C] + item_D):
                is_reactions_AB_CD = True
                print('break AC twice, BD intact')
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
                        print('ABCD each once!')
                        return str(is_reactions_AB_CD), one_bond_dict

    return str(is_reactions_AB_CD), one_bond_dict


def identify_reactions_AB_CD_old(mol_graphs1, mol_graphs2):
    A = mol_graphs1[0]
    B = mol_graphs1[1]
    C = mol_graphs2[0]
    D = mol_graphs2[1]

    # first scenario
    if len(A.graph.edges) != 0:
        for edge_A in A.graph.edges:
            bond_A = [(edge_A[0], edge_A[1])]
            try:
                frags_A = A.split_molecule_subgraphs(bond_A, allow_reverse=True)
                if len(B.graph.edges) != 0:
                    for edge_B in B.graph.edges:
                        bond_B = [(edge_B[0], edge_B[1])]
                        try:
                            frags_B = B.split_molecule_subgraphs(bond_B, allow_reverse=True)
                            if len(C.graph.edges) != 0:
                                for edge_C in C.graph.edges:
                                    bond_C = [(edge_C[0], edge_C[1])]
                                    try:
                                        frags_C = C.split_molecule_subgraphs(bond_C, allow_reverse=True)
                                        if len(D.graph.edges) != 0:
                                            for edge_D in D.graph.edges:
                                                bond_D = [(edge_D[0], edge_D[1])]
                                                try:
                                                    frags_D = D.split_molecule_subgraphs(bond_D, allow_reverse=True)
                                                    if check_same_mol_graphs(frags_A+frags_B, frags_C+frags_D):
                                                        is_reactions_AB_CD = True
                                                        return  is_reactions_AB_CD
                                                except:
                                                    continue
                                    except:
                                        continue
                        except:
                            continue
            except:
                continue
    # second scenario
    frags_A_two_step = break_two_bonds_in_one_mol(A)
    if frags_A_two_step != []:
        if len(C.graph.edges) != 0:
            for edge_C in C.graph.edges:
                bond_C = [(edge_C[0], edge_C[1])]
                try:
                    frags_C = C.split_molecule_subgraphs(bond_C, allow_reverse=True)
                    if len(D.graph.edges) != 0:
                        for edge_D in D.graph.edges:
                            bond_D = [(edge_D[0], edge_D[1])]
                            try:
                                frags_D = D.split_molecule_subgraphs(bond_D, allow_reverse=True)
                                for item in frags_A_two_step:
                                    if check_same_mol_graphs(item + B, frags_C + frags_D):
                                        is_reactions_AB_CD = True
                                        return is_reactions_AB_CD
                            except:
                                continue
                except:
                    continue
    frags_B_two_step = break_two_bonds_in_one_mol(B)
    if frags_B_two_step != []:
        if len(C.graph.edges) != 0:
            for edge_C in C.graph.edges:
                bond_C = [(edge_C[0], edge_C[1])]
                try:
                    frags_C = C.split_molecule_subgraphs(bond_C, allow_reverse=True)
                    if len(D.graph.edges) != 0:
                        for edge_D in D.graph.edges:
                            bond_D = [(edge_D[0], edge_D[1])]
                            try:
                                frags_D = D.split_molecule_subgraphs(bond_D, allow_reverse=True)
                                for item in frags_B_two_step:
                                    if check_same_mol_graphs(A + item, frags_C + frags_D):
                                        is_reactions_AB_CD = True
                                        return is_reactions_AB_CD
                            except:
                                continue
                except:
                    continue

    # third scenario
    frags_C_two_step = break_two_bonds_in_one_mol(C)

    frags_D_two_step = break_two_bonds_in_one_mol(D)
    pass


def identify_reactions_within_same_stoi(mol_graphs, stoi_dict):
    '''
    Identify reactions that are accessible by <=2 bond breakage <=2 bond formation
    :param mol_graphs: a list of mol graphs
    :param stoi_dict: a dictionary with key index of stoi, a list of mol_pair indices corresponding to the indices in mol_graphs
    :return: A dict of filtered reactions
    '''
    new_stoi_dict = {}
    for key in stoi_dict:
        new_stoi_dict[key] = []
        mol_pairs = stoi_dict[key]
        num_mol_pairs = len(mol_pairs)
        combos = list(combinations(range(num_mol_pairs),2))
        for combo in combos:
            mol_pair1 = mol_pairs[combo[0]]
            mol_pair2 = mol_pairs[combo[1]]
            # not necessarily two molecules, could be one
            mols1 = mol_pair1.split('_')
            mols2 = mol_pair2.split('_')
            if len(mols1) == len(mols2) == 1:
                # only self reaction possible
                mol_graph1 = mol_graphs[int(mols1[0])]
                mol_graph2 = mol_graphs[int(mols2[0])]
                if identify_self_reactions(mol_graph1, mol_graph2):
                    new_stoi_dict[key].append([mols1, mols2])
            else:
                # At least one mol list has 2 mols
                if (len(mols1) == 2 and len(mols2) == 1) or (len(mols1) == 1 and len(mols2) == 2):
                    # A + B -> C or C -> A + B
                    mol_graphs1 = [mol_graphs[int(i)] for i in mols1]
                    mol_graphs2 = [mol_graphs[int(i)] for i in mols2]
                    if len(mols1) == 2:
                        if identify_reactions_AB_C(mol_graphs1, mol_graphs2):
                            new_stoi_dict[key].append([mols1, mols2])
                    elif len(mols2) == 2:
                        if identify_reactions_AB_C(mol_graphs2, mol_graphs1):
                            new_stoi_dict[key].append([mols1, mols2])

                for i, mol in enumerate(mols1):
                    # if they have common molecules, then it's a self-reaction where only one mol is involved.
                    # Check if we can get to the same two fragments by breaking one bond on each of the mols.
                    # Different ring closure also considered
                    if mol in mols2:
                        j = mols2.index(mol)
                        mols1.pop(i)
                        mols2.pop(j)
                        mol_graph1 = mol_graphs[int(mols1[0])]
                        mol_graph2 = mol_graphs[int(mols2[0])]
                        if identify_self_reactions(mol_graph1, mol_graph2):
                            new_stoi_dict[key].append([mols1[0], mols2[0]])

                    # A + B -> C + D
                    else:
                        pass



def identify_reactions(mol_graphs):
    '''

    :param mol_graphs: A list of mol_graphs
    :return: a dictionary with all reactions A+B -> C+D (<= 2 bond breakage, <= 2 bond formation)
    '''
    num_mols = len(mol_graphs)
    all_mol_pair_index = list(combinations_with_replacement(range(num_mols), 2))

    pass

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






