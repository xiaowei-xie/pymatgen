from rdkit import Chem
from pymatgen.analysis import graphs
from pymatgen import Molecule
from pymatgen.analysis.fragmenter import Fragmenter
from monty.json import MSONable
from pymatgen.analysis.graphs import MoleculeGraph, MolGraphSplitError
from pymatgen.analysis.local_env import OpenBabelNN
from pymatgen.io.babel import BabelMolAdaptor
import networkx as nx
import numpy as np
from itertools import combinations_with_replacement, product
import subprocess
import os
import pybel
import openbabel as ob
import copy


class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)

#?
def copy_obmol(obmol):
    obmol_copy = ob.OBMol()
    num_atoms = obmol.NumAtoms()
    num_bonds = obmol.NumBonds()
    for i in range(num_atoms):
        new_atom = obmol_copy.NewAtom()
        new_atom.SetAtomicNum(obmol.GetAtomById(i).GetAtomicNum())
        new_atom.SetIdx(i)
    for i in range(num_bonds):
        bond_orig = obmol.GetBondById(i)
        begin_index = obmol.GetBondById(i).GetBeginAtomIdx() -1
        end_index = obmol.GetBondById(i).GetEndAtomIdx() -1
        new_bond = obmol_copy.NewBond()
        new_bond.SetIdx(i)
        new_bond.SetBegin(obmol_copy.GetAtomById(begin_index))
        new_bond.SetEnd(obmol_copy.GetAtomById(end_index))

        #begin_index = obmol.GetBondById(i).GetBeginAtomIdx() + 1
        #end_index = obmol.GetBondById(i).GetEndAtomIdx() + 1
        #order = obmol.GetBondById(i).GetBondOrder()
        #print(begin_index, end_index, order)
        #bool = obmol_copy.AddBond(bond_orig)
        #print(bool)
    return obmol_copy

def convert_obmol_to_normal(obmol):
    obmol_copy = ob.OBMol()
    num_atoms = obmol.NumAtoms()
    num_bonds = obmol.NumBonds()
    for i in range(num_atoms):
        new_atom = obmol_copy.NewAtom()
        new_atom.SetAtomicNum(obmol.GetAtomById(i).GetAtomicNum())
        new_atom.SetIdx(i)
    for i in range(num_bonds):
        begin_index = obmol.GetBondById(i).GetBeginAtomIdx() - 1
        end_index = obmol.GetBondById(i).GetEndAtomIdx() - 1
        new_bond = obmol_copy.NewBond()
        new_bond.SetIdx(i)
        new_bond.SetBegin(obmol_copy.GetAtomById(begin_index))
        new_bond.SetEnd(obmol_copy.GetAtomById(end_index))

        # begin_index = obmol.GetBondById(i).GetBeginAtomIdx() + 1
        # end_index = obmol.GetBondById(i).GetEndAtomIdx() + 1
        # order = obmol.GetBondById(i).GetBondOrder()
        # print(begin_index, end_index, order)
        # bool = obmol_copy.AddBond(bond_orig)
        # print(bool)
    return obmol_copy


def generate_all_unique_connected_fragments(mol, depth=1, open_rings=True):
    '''

    :param mol: pymatgen mol to fragment
    :return: a list of pymatgen mols and mol graphs each
    '''
    mol_graph = MoleculeGraph.with_local_env_strategy(mol, OpenBabelNN(),
                                                      reorder=False,
                                                      extend_structure=False)
    a = Fragmenter(mol, depth=depth, open_rings=open_rings)
    mols = []
    mol_graphs = []
    for key in a.unique_frag_dict.keys():
        for i in range(len(a.unique_frag_dict[key])):
            mol_graph = a.unique_frag_dict[key][i]
            new_mol_graph = MoleculeGraph.with_local_env_strategy(mol_graph.molecule, OpenBabelNN(),
                                                                  reorder=False,
                                                                  extend_structure=False)
            mol_graphs.append(new_mol_graph)
            mol = a.unique_frag_dict[key][i].molecule
            mols.append(mol)

    connected_mol_graphs = []
    connected_mols = []
    for i in range(len(mol_graphs)):
        G = mol_graphs[i].graph.to_undirected()
        if not nx.is_connected(G):
            print(i)
            indices = [c for c in sorted(nx.connected_components(G), key=len, reverse=True)]
            for j in range(len(indices)):
                index = np.array(list(indices[j]))
                coords = np.array([site.coords for site in mols[i]])[index]
                species = np.array([str(site.specie.symbol) for site in mols[i]])[index]
                new_mol = Molecule(species=species, coords=coords)
                new_mol_graph = MoleculeGraph.with_local_env_strategy(new_mol, OpenBabelNN(),
                                                           reorder=False,
                                                           extend_structure=False)
                if check_in_list(new_mol_graph,connected_mol_graphs):
                    pass
                else:
                    connected_mol_graphs.append(new_mol_graph)
                    connected_mols.append(new_mol)
        else:
            if check_in_list(mol_graphs[i], connected_mol_graphs):
                pass
            else:
                connected_mol_graphs.append(mol_graphs[i])
                connected_mols.append(mols[i])
    return connected_mols, connected_mol_graphs

def get_rdkit_mols(mol_graphs):
    '''
    Generate rdkit mols and smiles from pymatgen mol graphs, only smiles information is used
    :param mol_graphs:
    :return:
    '''
    smiles_list = []
    rmols = []
    for i in range(len(mol_graphs)):
        bb = BabelMolAdaptor.from_molecule_graph(mol_graphs[i])
        pbmol = bb.pybel_mol
        smiles = pbmol.write(str("smi")).split()[0]
        rmol = Chem.MolFromSmiles(smiles,sanitize = False)
        smiles_list.append(smiles)
        rmols.append(rmol)
    return rmols, smiles_list


def check_in_list(mol_graph_test, mol_graph_list):

    for mol_graph in mol_graph_list:
        if mol_graph.molecule.composition.reduced_formula == mol_graph_test.molecule.composition.reduced_formula:
            if mol_graph_test.isomorphic_to(mol_graph):
                return True
    return False

def combine_mols(mol1, mol2, index1, index2):
    '''

    :param mol1: rdkit mol
    :param mol2: rdkit mol
    :param index1: index of the atom on mol1 to be combined
    :param index2: index of the atom on mol2 to be combined
    :return:
    '''
    num_atoms_mol1 = mol1.GetNumAtoms()
    combo = Chem.CombineMols(mol1, mol2)
    edcombo = Chem.EditableMol(combo)
    index2 = index2 + num_atoms_mol1
    edcombo.AddBond(index1, index2, order=Chem.rdchem.BondType.SINGLE)
    mol = edcombo.GetMol()
    smiles_string = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True, allBondsExplicit=True, allHsExplicit=True)
    return smiles_string

def identify_connectable_heavy_atoms(mols):
    '''

    :param mols: a list of rdkit mol object
    :return: a list of list of index of heavy atoms for each mol
    '''
    heavy_atoms_index_list = []
    for i, mol in enumerate(mols):
        heavy_atoms_in_mol = []
        num_atoms = mol.GetNumAtoms()
        for j in range(num_atoms):
            atom = mol.GetAtomWithIdx(j)

            # for O, if (valence < 2) or (valence = 2 but has Li neighbors):
            if atom.GetAtomicNum() == 8:
                if atom.GetTotalValence() < 2:
                    heavy_atoms_in_mol.append(j)
                elif atom.GetTotalValence() == 2:
                    num_neighbors = len(atom.GetNeighbors())
                    if any([atom.GetNeighbors()[k].GetAtomicNum() == 3 for k in range(num_neighbors)]):
                        heavy_atoms_in_mol.append(j)
                else:
                    ValueError("Please check the valence of your original mol!")

            # for C, if valence < 4:
            if atom.GetAtomicNum() == 6:
                if atom.GetTotalValence() < 4:
                    heavy_atoms_in_mol.append(j)

            # for N, if (valence < 3) or (valence ==3 but has Li neighbors):
            if atom.GetAtomicNum() == 7:
                if atom.GetTotalValence() < 3:
                    heavy_atoms_in_mol.append(j)
                elif atom.GetTotalValence() == 3:
                    num_neighbors = len(atom.GetNeighbors())
                    if any([atom.GetNeighbors()[k].GetAtomicNum() == 3 for k in range(num_neighbors)]):
                        heavy_atoms_in_mol.append(j)
                else:
                    ValueError("Please check the valence of your original mol!")

            # for Li, for now only connectable if it's a single atom, and should be set in later functions to only connect to heteroatoms
            if atom.GetAtomicNum() == 3:
                if len(atom.GetNeighbors()) == 0:
                    heavy_atoms_in_mol.append(j)
            # for H, only connectable if it's a single atom, and should be set in later functions to only connect to non-H, non-Li atoms
            if atom.GetAtomicNum() == 1:
                if len(atom.GetNeighbors()) == 0:
                    heavy_atoms_in_mol.append(j)

        heavy_atoms_index_list.append(heavy_atoms_in_mol)
    return heavy_atoms_index_list

def identify_connectable_heavy_atoms_without_sanitize(mols):
    '''

    :param mols: a list of rdkit mol object
    :return: a list of list of index of heavy atoms for each mol
    '''
    heavy_atoms_index_list = []
    for i, mol in enumerate(mols):
        heavy_atoms_in_mol = []
        num_atoms = mol.GetNumAtoms()
        for j in range(num_atoms):
            atom = mol.GetAtomWithIdx(j)

            # for O, if num_neighbors + H count - Li_count < 2: (cannot call total_valence b/c rmol is not sanitized)
            if atom.GetAtomicNum() == 8:
                neighbors = atom.GetNeighbors()
                num_neighbors = len(neighbors)
                Li_count = 0
                H_count = atom.GetNumExplicitHs()
                for k,neighbor in enumerate(neighbors):
                    if neighbor.GetAtomicNum() == 3:
                        Li_count += 1

                if num_neighbors + H_count - Li_count < 2:
                    heavy_atoms_in_mol.append(j)

            # for C, if num_neighbors < 4:
            if atom.GetAtomicNum() == 6:
                neighbors = atom.GetNeighbors()
                num_neighbors = len(neighbors)
                H_count = atom.GetNumExplicitHs()

                if num_neighbors + H_count < 4:
                    heavy_atoms_in_mol.append(j)

            # for N, if num_neighbors + H_count - Li_count < 3
            if atom.GetAtomicNum() == 7:
                neighbors = atom.GetNeighbors()
                num_neighbors = len(neighbors)
                H_count = atom.GetNumExplicitHs()
                Li_count = 0
                for k, neighbor in enumerate(neighbors):
                    if neighbor.GetAtomicNum() == 3:
                        Li_count += 1
                if num_neighbors + H_count - Li_count < 3:
                    heavy_atoms_in_mol.append(j)

            # for Li, for now only connectable if it's a single atom, and should be set in later functions to only connect to heteroatoms
            if atom.GetAtomicNum() == 3:
                if len(atom.GetNeighbors()) == 0 and num_atoms == 1:
                    heavy_atoms_in_mol.append(j)
            # for H, only connectable if it's a single atom, and should be set in later functions to only connect to non-H, non-Li atoms
            if atom.GetAtomicNum() == 1:
                if len(atom.GetNeighbors()) == 0 and num_atoms == 1:
                    heavy_atoms_in_mol.append(j)
            # for F, only connectable if it's a single atom
            if atom.GetAtomicNum() == 9:
                if len(atom.GetNeighbors()) == 0 and num_atoms == 1:
                    heavy_atoms_in_mol.append(j)
            # for 110, if num_neighbors < 5:
            if atom.GetAtomicNum() == 15:
                neighbors = atom.GetNeighbors()
                num_neighbors = len(neighbors)
                H_count = atom.GetNumExplicitHs()

                if num_neighbors + H_count < 5:
                    heavy_atoms_in_mol.append(j)


        heavy_atoms_index_list.append(heavy_atoms_in_mol)
    return heavy_atoms_index_list

def generate_all_combos(mols):
    '''
    generate all combination smiles strings by looping through all molecule pairs(including a mol and itself) and all connectable heavy atoms
    :param mols: a list of rdkit mol object
    :return: a dict of smiles string with key 'mol1_index'+'_'+'mol2_index'+'_'+'atom1_index'+'_'+'atom2_index'.
            Li can only combine with heteroatoms -> update: can combine with carbons now
    '''
    smiles_dict = {}
    heavy_atoms_index_list = identify_connectable_heavy_atoms_without_sanitize(mols)
    num_mols = len(mols)
    all_mol_pair_index = list(combinations_with_replacement(range(num_mols),2))
    for pair_index in all_mol_pair_index:
        pair_key = str(pair_index[0])+'_'+str(pair_index[1])
        mol1 = mols[pair_index[0]]
        mol2 = mols[pair_index[1]]
        heavy_atoms_1 = heavy_atoms_index_list[pair_index[0]]
        heavy_atoms_2 = heavy_atoms_index_list[pair_index[1]]
        if len(heavy_atoms_1) == 0 or len(heavy_atoms_2) == 0:
            pass
        else:
            for i, atom1 in enumerate(heavy_atoms_1):
                for j, atom2 in enumerate(heavy_atoms_2):
                    specie1 = mol1.GetAtomWithIdx(atom1).GetAtomicNum()
                    specie2 = mol2.GetAtomWithIdx(atom2).GetAtomicNum()

                    if specie1 == specie2 == 3:
                        continue
                    elif specie1 == 3 and (specie2 == 1 or specie2 == 15):
                        continue
                    elif specie2 == 3 and (specie1 == 1 or specie2 == 15):
                        continue
                    elif specie1 == specie2 == 15:
                        continue

                    atom_index_key = str(atom1)+'_'+str(atom2)
                    name = pair_key + '_' + atom_index_key
                    print(name)
                    combo_smiles = combine_mols(mol1, mol2, atom1, atom2)
                    smiles_dict[name] = combo_smiles
    return smiles_dict

def generate_all_combo_mol_graphs(mols):
    '''
    Generate all combined mol graphs by looping through all molecule pairs(including a mol and itself) and all connectable heavy atoms, and
    filtering out mol graphs that are isomorphic to the existing mol graphs
    :param mols: rdkit mols
    :return: a list of mol graphs
    '''
    smiles_dict = generate_all_combos(mols)
    new_pbmols = []
    pmols = []
    pmol_graphs = []
    for i, key in enumerate(smiles_dict.keys()):
        new_pbmol = pybel.readstring("smi", smiles_dict[key]).OBMol
        new_pbmol = convert_obmol_to_normal(new_pbmol)
        new_pbmols.append(new_pbmol)
        adaptor = BabelMolAdaptor(new_pbmol)
        adaptor.make3d('uff')
        pmol = adaptor.pymatgen_mol
        pmol_graph = MoleculeGraph.with_local_env_strategy(pmol, OpenBabelNN(),
                                                           reorder=False,
                                                           extend_structure=False)
        if not check_in_list(pmol_graph, pmol_graphs):
            pmols.append(pmol)
            pmol_graphs.append(pmol_graph)
    return pmol_graphs
'''
def check_close_shell_for_one_bond(obmol, bond, missing_valence_list, missing_valence_list_except_Li):
    close_shell = True
    if obmol.NumBonds() == 0:
        if obmol.GetAtomById(0).GetAtomicNum() != 3:
        # Do we consider H-?
            close_shell = False

    begin_index = bond.GetBeginAtomIdx() - 1
    end_index = bond.GetEndAtomIdx() - 1
    begin_missing_valence = missing_valence_list[begin_index]
    end_missing_valence = missing_valence_list[end_index]
    begin_missing_valence_expect_Li = missing_valence_list_except_Li[begin_index]
    end_missing_valence_except_Li = missing_valence_list_except_Li[end_index]
    print(begin_index, end_index, begin_missing_valence, end_missing_valence,begin_missing_valence_expect_Li, end_missing_valence_except_Li)
    if obmol.GetAtomById(begin_index).GetAtomicNum() == 3 or obmol.GetAtomById(end_index).GetAtomicNum() == 3:
        if obmol.NumBonds() == 1 and (begin_missing_valence != end_missing_valence):
            #Li-Li wasn't generated in the first place
            close_shell = False

    elif begin_missing_valence != end_missing_valence:
        # Li should only be connected to heteroatoms, so begin_missing_valence_expect_Li == end_missing_valence_except_Li >= 3 is not considered
        if (begin_missing_valence_expect_Li == end_missing_valence_except_Li == 1) or \
                (begin_missing_valence_expect_Li == end_missing_valence_except_Li == 2):
            bond.SetBondOrder(bond.GetBondOrder() + begin_missing_valence_expect_Li)
            missing_valence_list_except_Li[begin_index] -= begin_missing_valence_expect_Li
            missing_valence_list_except_Li[end_index] -= end_missing_valence_except_Li
        elif (begin_missing_valence_expect_Li == end_missing_valence_except_Li == 0):
            pass
        else:
            # still need to consider the bonds connected to the neighbor atom(i.e. LEDC), if neighbor valence can be changed, it's ok
            for j,begin_neighbor_bond in enumerate(ob.OBAtomBondIter(obmol.GetAtomById(begin_index))):
                begin_neighbor_bond_close_shell, missing_valence_list, missing_valence_list_except_Li = \
                # for the bonds connecting to one atom, first check it valence is fixable by correcting one bond,
                # then check if the missing valence of the central atom is the sum of the missing valences of surrounding atoms
                check_close_shell_for_one_bond(obmol, begin_neighbor_bond,missing_valence_list, missing_valence_list_except_Li)
                #stopped !!

            close_shell = False
    elif (begin_missing_valence == end_missing_valence == 0):
        pass
    elif (begin_missing_valence == end_missing_valence == 1) or (begin_missing_valence == end_missing_valence == 2):
        bond.SetBondOrder(bond.GetBondOrder()+begin_missing_valence)
        missing_valence_list[begin_index] -= begin_missing_valence
        missing_valence_list[end_index] -= end_missing_valence

    else:
        if (begin_missing_valence_expect_Li == end_missing_valence_except_Li == 1):
            bond.SetBondOrder(bond.GetBondOrder() + begin_missing_valence_expect_Li)
            missing_valence_list_except_Li[begin_index] -= 1
            missing_valence_list_except_Li[end_index] -= 1
        elif (begin_missing_valence_expect_Li == end_missing_valence_except_Li == 0):
            pass
        else:
            close_shell = False

    return str(close_shell), missing_valence_list, missing_valence_list_except_Li


def check_close_shell(obmol):
    
    Check if a pbmol has the possibility to be closed shell molecule
    :param obmol: a openbabel mol object
    :return: True/False, modified obmol(only meaningful when it is true)
    
    expected_valence_for_atom = {1:1, 3:1, 6:4, 7:3, 8:2}
    num_atoms = obmol.NumAtoms()
    num_bonds = obmol.NumBonds()
    missing_valence_list = []
    missing_valence_list_except_Li = []
    for i in range(num_atoms):
        atom = obmol.GetAtomById(i)
        atom_number = atom.GetAtomicNum()
        neighbor_index_list = []
        num_of_connected_Li = 0
        for j, bond in enumerate(ob.OBAtomBondIter(atom)):
            # in obmol, the begin and end atom index starts from 1 instead of 0, so have to shift back to get the actual atom index
            begin_index = bond.GetBeginAtomIdx() - 1
            end_index = bond.GetEndAtomIdx() - 1
            if begin_index == i:
                neighbor_index = end_index
            elif end_index == i:
                neighbor_index = begin_index
            if obmol.GetAtomById(neighbor_index).GetAtomicNum() == 3:
                num_of_connected_Li += 1
            neighbor_index_list.append(neighbor_index)
        #missing_valence means (implicit valence - actual valence), where actual valence is (GetHvyValence()+ ExplicitHydrogenCount() - Num of Li connected)
        actual_valence_for_atom = atom.GetHvyValence() + atom.ExplicitHydrogenCount()
        atom_valence_except_Li = atom.GetHvyValence() + atom.ExplicitHydrogenCount() - num_of_connected_Li
        missing_valence_except_Li = expected_valence_for_atom[atom.GetAtomicNum()] - atom_valence_except_Li
        missing_valence_list_except_Li.append(missing_valence_except_Li)
        #expected_valence = expected_valence_for_atom[atom_number]
        missing_valence_for_atom = atom.GetImplicitValence() - actual_valence_for_atom
        missing_valence_list.append(missing_valence_for_atom)
    close_shell = True
    if obmol.NumBonds() == 0:
        if obmol.GetAtomById(0).GetAtomicNum() != 3:
        # Do we consider H-?
            close_shell = False
    for i, bond in enumerate(ob.OBMolBondIter(obmol)):
        if close_shell == True:
            print(i-1, 'passed!')
        begin_index = bond.GetBeginAtomIdx() - 1
        end_index = bond.GetEndAtomIdx() - 1
        begin_missing_valence = missing_valence_list[begin_index]
        end_missing_valence = missing_valence_list[end_index]
        begin_missing_valence_expect_Li = missing_valence_list_except_Li[begin_index]
        end_missing_valence_except_Li = missing_valence_list_except_Li[end_index]
        print(begin_index, end_index, begin_missing_valence, end_missing_valence,begin_missing_valence_expect_Li, end_missing_valence_except_Li)
        if obmol.GetAtomById(begin_index).GetAtomicNum() == 3 or obmol.GetAtomById(end_index).GetAtomicNum() == 3:
            if obmol.NumBonds() == 1 and (begin_missing_valence != end_missing_valence):
                #Li-Li wasn't generated in the first place
                close_shell = False
                break
            else:
                continue
        elif begin_missing_valence != end_missing_valence:
            # Li should only be connected to heteroatoms, so begin_missing_valence_expect_Li == end_missing_valence_except_Li >= 3 is not considered
            if (begin_missing_valence_expect_Li == end_missing_valence_except_Li == 1) or \
                    (begin_missing_valence_expect_Li == end_missing_valence_except_Li == 2):
                bond.SetBondOrder(bond.GetBondOrder() + begin_missing_valence_expect_Li)
                missing_valence_list_except_Li[begin_index] -= begin_missing_valence_expect_Li
                missing_valence_list_except_Li[end_index] -= end_missing_valence_except_Li
                continue
            elif (begin_missing_valence_expect_Li == end_missing_valence_except_Li == 0):
                continue
            else:
                # still need to consider the bonds connected to the neighbor atom(i.e. LEDC), if neighbor valence can be changed, it's ok
                for j,begin_neighbor_bond in enumerate(ob.OBAtomBondIter(obmol.GetAtomById(begin_index))):

                close_shell = False
                break
        elif (begin_missing_valence == end_missing_valence == 0):
            continue
        elif (begin_missing_valence == end_missing_valence == 1) or (begin_missing_valence == end_missing_valence == 2):
            bond.SetBondOrder(bond.GetBondOrder()+begin_missing_valence)
            missing_valence_list[begin_index] -= begin_missing_valence
            missing_valence_list[end_index] -= end_missing_valence
            continue
        else:
            if (begin_missing_valence_expect_Li == end_missing_valence_except_Li == 1):
                bond.SetBondOrder(bond.GetBondOrder() + begin_missing_valence_expect_Li)
                missing_valence_list_except_Li[begin_index] -= 1
                missing_valence_list_except_Li[end_index] -= 1
                continue
            elif (begin_missing_valence_expect_Li == end_missing_valence_except_Li == 0):
                continue
            else:
                close_shell = False
                break
    return str(close_shell), obmol

'''

def missing_valence(obmol):
    expected_valence_for_atom = {1: 1, 3: 1, 6: 4, 7: 3, 8: 2}
    num_atoms = obmol.NumAtoms()
    num_bonds = obmol.NumBonds()
    missing_valence_list = []
    missing_valence_list_except_Li = []
    for i in range(num_atoms):
        atom = obmol.GetAtomById(i)
        atom_number = atom.GetAtomicNum()
        neighbor_index_list = []
        num_of_connected_Li = 0
        bond_order_list = []
        bond_order_excess_total = 0
        for j, bond in enumerate(ob.OBAtomBondIter(atom)):
            # in obmol, the begin and end atom index starts from 1 instead of 0, so have to shift back to get the actual atom index
            begin_index = bond.GetBeginAtomIdx()
            end_index = bond.GetEndAtomIdx()
            bond_order = bond.GetBondOrder()
            bond_order_excess = bond_order - 1
            bond_order_excess_total += bond_order_excess
            bond_order_list.append(bond_order)
            neighbor_index = -1
            if begin_index == i:
                neighbor_index = end_index
            elif end_index == i:
                neighbor_index = begin_index
            if obmol.GetAtomById(neighbor_index).GetAtomicNum() == 3:
                num_of_connected_Li += 1
            neighbor_index_list.append(neighbor_index)
        #missing_valence means (implicit valence - actual valence), where actual valence is (GetHvyValence()+ ExplicitHydrogenCount() - Num of Li connected)
        actual_valence_for_atom = atom.GetHvyValence() + atom.ExplicitHydrogenCount()
        atom_valence_except_Li = atom.GetHvyValence() + atom.ExplicitHydrogenCount() - num_of_connected_Li + bond_order_excess_total
        missing_valence_except_Li = expected_valence_for_atom[atom.GetAtomicNum()] - atom_valence_except_Li
        missing_valence_list_except_Li.append(missing_valence_except_Li)
        #expected_valence = expected_valence_for_atom[atom_number]
        missing_valence_for_atom = atom.GetImplicitValence() - actual_valence_for_atom
        missing_valence_list.append(missing_valence_for_atom)
    missing_valence_array = np.array(missing_valence_list)
    missing_valence_array_except_Li = np.array(missing_valence_list_except_Li)
    return missing_valence_array, missing_valence_array_except_Li



def check_close_shell_for_atom(obmol, idx):
    '''

    :param obmol:
    :param smiles:
    :param idx: atom index
    :return: close_shell(bool),obmols,smiles_list
    '''
    obmols_new = []
    #smiles_new = []
    close_shell = True
    missing_valence_array, missing_valence_array_except_Li = missing_valence(obmol)
    atom = obmol.GetAtomById(idx)
    if missing_valence_array[idx] == 0 or missing_valence_array_except_Li[idx] == 0:
        obmols_new.append(obmol)
        #smiles_new.append(smiles)
    else:
        neighbor_index_list = []
        neighbor_bond_begin_end_index_list = []
        for j, bond in enumerate(ob.OBAtomBondIter(atom)):
            # in obmol, the begin and end atom index starts from 1 instead of 0, so have to shift back to get the actual atom index
            begin_index = bond.GetBeginAtomIdx()
            end_index = bond.GetEndAtomIdx()
            neighbor_bond_begin_end_index_list.append((begin_index, end_index))
            neighbor_index = -1
            if begin_index == idx:
                neighbor_index = end_index
            elif end_index == idx:
                neighbor_index = begin_index
            neighbor_index_list.append(neighbor_index)
        neighbor_index_array = np.array(neighbor_index_list)
        neighbor_missing_valence_array = missing_valence_array[neighbor_index_array]
        neighbor_missing_valence_array_except_Li = missing_valence_array_except_Li[neighbor_index_array]
        transpose = np.transpose(np.vstack([neighbor_missing_valence_array, neighbor_missing_valence_array_except_Li]))
        combinations = list(
            set(list(product(*transpose))))
        for j, combo in enumerate(combinations):
            if np.sum(combo) == missing_valence_array[idx] or np.sum(combo) == missing_valence_array_except_Li[idx]:
                new_obmol = copy_obmol(obmol)
                for k, value in enumerate(combo):
                    if value != 0:
                        new_bond = new_obmol.GetBond(new_obmol.GetAtomById(neighbor_bond_begin_end_index_list[k][0]),
                                                     new_obmol.GetAtomById(neighbor_bond_begin_end_index_list[k][1]))
                        new_bond.SetBondOrder(int(new_bond.GetBondOrder() + value))
                obmols_new.append(new_obmol)
                #new_smiles = pybel.Molecule(new_obmol).write(str("smi")).split()[0]
                #smiles_new.append(new_smiles)

    if obmols_new == []:
        close_shell = False
    return str(close_shell), obmols_new


def check_close_shell_new(obmol):
    '''
    :param obmol:
    :param smiles: smiles string for the whole mol. need to input smiles because there's no way to copy a obmol to ensure the original is not modified when the copy is modified.
    :return: generate all possible close shell obmols by modifying the bonds
    '''
    num_atoms = obmol.NumAtoms()
    final_obmols = {}
    final_obmols[-1] = [obmol]
    #final_smiles[-1] = [smiles]
    for i in range(num_atoms):
        print(i)
        final_obmols[i] = []
        if final_obmols[i-1] != []:
            print('true')
            for k in range(len(final_obmols[i-1])):
                close_shell, obmols_new = check_close_shell_for_atom(final_obmols[i-1][k],i)
                if close_shell == 'True':
                    final_obmols[i] += obmols_new
                    #final_smiles[i] += smiles_new
    # remove duplicates
    #final_final_obmols, final_final_smiles = [], []
    #if len(final_smiles[num_atoms-1]) != 0:
    #    for i in range(len(final_smiles[num_atoms-1])):
    #        if final_smiles[num_atoms-1][i] not in final_smiles[num_atoms-1][:i]:
    #            final_final_smiles.append(final_smiles[num_atoms-1][i])
    #            final_final_obmols.append(final_obmols[num_atoms-1][i])

    return final_obmols[-1]

def generate_mols_for_close_shell(obmols):
    pmols = []
    for i, obmol in enumerate(obmols):
        adaptor = BabelMolAdaptor(obmol)
        adaptor.make3d('uff')
        pmol = adaptor.pymatgen_mol
        pmols.append(pmol)
    return pmols

def visualize_obmol(obmol):
    for i in range(obmol.NumAtoms()):
        print(i, obmol.GetAtomById(i).GetAtomicNum())
    for i in range(obmol.NumBonds()):
        begin_index = obmol.GetBondById(i).GetBeginAtom().GetIdx()
        end_index = obmol.GetBondById(i).GetEndAtom().GetIdx()
        begin_atom = obmol.GetBondById(i).GetBeginAtom().GetAtomicNum()
        end_atom = obmol.GetBondById(i).GetEndAtom().GetAtomicNum()
        order = obmol.GetBondById(i).GetBondOrder()
        print(i, begin_index, end_index,begin_atom, end_atom,order)
    return

def visualize_rmol(rmol):
    num_heavy_atoms = rmol.GetNumAtoms()
    num_bonds = rmol.GetNumBonds()
    for i in range(num_heavy_atoms):
        atom = rmol.GetAtomWithIdx(i)
        atomic_number = atom.GetAtomicNum()
        print(i, atomic_number)
    for i in range(num_bonds):
        bond = rmol.GetBondWithIdx(i)
        begin_index = bond.GetBeginAtomIdx()
        end_index = bond.GetEndAtomIdx()
        begin_atom = bond.GetBeginAtom().GetAtomicNum()
        end_atom = bond.GetEndAtom().GetAtomicNum()
        bond_order = bond.GetBondType()
        print(i, begin_index, end_index, begin_atom,end_atom, bond_order)

    return

def missing_valence_for_rmol(rmol):
    expected_valence_for_atom = {1: 1, 3: 1, 6: 4, 7: 3, 8: 2, 9:1}
    # Only heavy atoms in rmol
    num_atoms = rmol.GetNumAtoms()
    num_bonds = rmol.GetNumBonds()
    missing_valence_list = []
    missing_valence_list_except_Li = []
    for i in range(num_atoms):
        atom = rmol.GetAtomWithIdx(i)
        neighbor_index_list = []
        num_of_connected_Li = 0
        bond_order_list = []
        bond_order_excess_total = 0
        neighbors = atom.GetNeighbors()
        num_neighbors = len(neighbors)
        num_explicit_Hs = atom.GetNumExplicitHs()
        for j in range(len(neighbors)):
            neighbor_index = neighbors[j].GetIdx()
            neighbor_index_list.append(neighbor_index)
            bond_order = rmol.GetBondBetweenAtoms(i,neighbor_index).GetBondType().bit_length()
            bond_order_excess = bond_order - 1
            bond_order_excess_total += bond_order_excess
            bond_order_list.append(bond_order)
            if rmol.GetAtomWithIdx(neighbor_index).GetAtomicNum() == 3:
                num_of_connected_Li += 1
        actual_valence_for_atom = num_neighbors + bond_order_excess_total + num_explicit_Hs
        atom_valence_except_Li = num_neighbors + bond_order_excess_total - num_of_connected_Li + num_explicit_Hs
        missing_valence_for_atom = None
        missing_valence_except_Li = None
        # For phosphorus, allowing valence to be 3 or 5
        if atom.GetAtomicNum() == 15:
            if actual_valence_for_atom <= 3:
                missing_valence_for_atom = 3 - actual_valence_for_atom
                missing_valence_except_Li = 3 - atom_valence_except_Li
            elif actual_valence_for_atom >3 :
                missing_valence_for_atom = 5 - actual_valence_for_atom
                missing_valence_except_Li = 5 - atom_valence_except_Li
        # For carbon, C-Li bond would be considered as valence bond, so missing_valence_except_Li would not change
        elif atom.GetAtomicNum() == 6:
            missing_valence_for_atom = 4 - actual_valence_for_atom
            missing_valence_except_Li = 4 - actual_valence_for_atom
        else:
            missing_valence_for_atom = expected_valence_for_atom[atom.GetAtomicNum()] - actual_valence_for_atom
            missing_valence_except_Li = expected_valence_for_atom[atom.GetAtomicNum()] - atom_valence_except_Li
        if missing_valence_for_atom < 0:
            missing_valence_for_atom = 0

        missing_valence_list.append(missing_valence_for_atom)
        missing_valence_list_except_Li.append(missing_valence_except_Li)
    missing_valence_array = np.array(missing_valence_list)
    missing_valence_array_except_Li = np.array(missing_valence_list_except_Li)
    return missing_valence_array, missing_valence_array_except_Li

def check_close_shell_for_atom_rmol(rmol, idx):
    '''

    :param obmol:
    :param smiles:
    :param idx: atom index
    :return: close_shell(bool),obmols,smiles_list
    '''
    bond_types = {1: Chem.rdchem.BondType.SINGLE, 2: Chem.rdchem.BondType.DOUBLE, 3: Chem.rdchem.BondType.TRIPLE}
    smiles_for_rmol = Chem.MolToSmiles(rmol, isomericSmiles=True, canonical=True, allBondsExplicit=True, allHsExplicit=True)
    rmols_new = []
    smiles_new = []
    close_shell = True
    missing_valence_array, missing_valence_array_except_Li = missing_valence_for_rmol(rmol)
    atom = rmol.GetAtomWithIdx(idx)
    if missing_valence_array[idx] == 0 or missing_valence_array_except_Li[idx] == 0:
        rmols_new.append(rmol)
        smiles_new.append(smiles_for_rmol)
    else:
        neighbor_index_list = []
        neighbors = atom.GetNeighbors()
        for j in range(len(neighbors)):
            neighbor_index = neighbors[j].GetIdx()
            neighbor_index_list.append(neighbor_index)
        neighbor_index_array = np.array(neighbor_index_list)
        neighbor_missing_valence_array = missing_valence_array[neighbor_index_array]
        neighbor_missing_valence_array_except_Li = missing_valence_array_except_Li[neighbor_index_array]

        transpose = np.transpose(np.vstack([neighbor_missing_valence_array, neighbor_missing_valence_array_except_Li]))
        combinations = list(set(list(product(*transpose))))
        for j, combo in enumerate(combinations):
            if np.sum(combo) == missing_valence_array[idx] or np.sum(combo) == missing_valence_array_except_Li[idx]:
                new_rmol = copy.deepcopy(rmol)
                for k, value in enumerate(combo):
                    if value != 0:
                        new_bond = new_rmol.GetBondBetweenAtoms(idx,neighbor_index_list[k])
                        new_bond_order = new_bond.GetBondType().bit_length() + value
                        new_bond.SetBondType(bond_types[new_bond_order])
                rmols_new.append(new_rmol)

    if rmols_new == []:
        close_shell = False
    else:
        for i, mol in enumerate(rmols_new):
            smiles = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True, allBondsExplicit=True, allHsExplicit=True)
            smiles_new.append(smiles)
    return str(close_shell), rmols_new, smiles_new

def check_close_shell_for_rmol(rmol):
    '''
    :param obmol:
    :param smiles: smiles string for the whole mol. need to input smiles because there's no way to copy a obmol to ensure the original is not modified when the copy is modified.
    :return: generate all possible close shell obmols by modifying the bonds
    '''
    num_atoms = rmol.GetNumAtoms()
    smiles_for_rmol = Chem.MolToSmiles(rmol, isomericSmiles=True, canonical=True, allBondsExplicit=True,
                                       allHsExplicit=True)
    if num_atoms == 1:
        if rmol.GetAtomWithIdx(0).GetAtomicNum()==3:
            return [rmol], [smiles_for_rmol]
        else:
            return [],[]
    final_rmols, final_smiles = {}, {}
    final_rmols[-1] = [rmol]
    final_smiles[-1] = [smiles_for_rmol]
    for i in range(num_atoms):
        print(i)
        final_rmols[i] = []
        final_smiles[i] = []
        if final_rmols[i-1] != []:
            print('true')
            for k in range(len(final_rmols[i-1])):
                close_shell, rmols_new, smiles_new = check_close_shell_for_atom_rmol(final_rmols[i-1][k],i)
                if close_shell == 'True':
                    final_rmols[i] += rmols_new
                    final_smiles[i] += smiles_new
    # remove duplicates
    final_final_rmols, final_final_smiles = [], []
    if len(final_smiles[num_atoms-1]) != 0:
        for i in range(len(final_smiles[num_atoms-1])):
            if final_smiles[num_atoms-1][i] not in final_smiles[num_atoms-1][:i]:
                final_final_smiles.append(final_smiles[num_atoms-1][i])
                final_final_rmols.append(final_rmols[num_atoms-1][i])

    return final_final_rmols, final_final_smiles

def compute_lacking_valence_for_rmol(rmol):
    '''
    compute the least possible lacking valence for a rmol. Change bond orders if possible.
    :param rmol:
    :return:
    '''
    pass

def generate_pmols_from_smiles(smiles_list):
    pmols = []
    for i, smiles in enumerate(smiles_list):
        new_obmol = pybel.readstring("smi", smiles).OBMol
        adaptor = BabelMolAdaptor(new_obmol)
        adaptor.make3d('uff')
        pmol = adaptor.pymatgen_mol
        pmols.append(pmol)
    return pmols

def recombine_between_mol_graphs(mol_graphs):
    unique_fragments_after_recomb = mol_graphs.copy()
    unique_obmols_after_recomb = []
    unique_rmols_after_recomb = []
    unique_smiles_after_recomb = []
    for i, mol_graph in enumerate(unique_fragments_after_recomb):
        bb = BabelMolAdaptor.from_molecule_graph(mol_graph)
        pbmol = bb.pybel_mol
        # had to do this ,otherwise some obmol.GetBondById(i) would be empty. Why?
        smiles = pbmol.write(str("smi")).split()[0]
        rmol = Chem.MolFromSmiles(smiles)
        #unique_rmols_after_recomb.append(rmol)
        smiles_string = Chem.MolToSmiles(rmol, isomericSmiles=True, canonical=True, allBondsExplicit=True,
                                         allHsExplicit=True)
        # Phosphorus will be force add Hs if not satisfy valence...
        if '[PH]' in smiles_string:
            smiles_string = smiles_string.replace('[PH]','[P]')
        unique_smiles_after_recomb.append(smiles_string)
        obmol = pybel.readstring("smi", smiles_string).OBMol
        rmol = Chem.MolFromSmiles(smiles_string, sanitize=False)
        # obmol = convert_obmol_to_normal(obmol)
        unique_rmols_after_recomb.append(rmol)
        unique_obmols_after_recomb.append(obmol)

    # rmols, smiles_list = get_rdkit_mols(unique_rmols_after_recomb)

    smiles_dict = generate_all_combos(unique_rmols_after_recomb)
    recomb_dict = {}
    for i, key in enumerate(smiles_dict.keys()):
        recomb_dict[key] = {'reactant': [], 'product': []}
        indexs = key.split('_')
        mol1_index = indexs[0]
        mol2_index = indexs[1]
        new_pbmol = pybel.readstring("smi", smiles_dict[key]).OBMol
        # new_pbmol = convert_obmol_to_normal(new_pbmol)
        # new_rmol = Chem.MolFromSmiles(smiles_dict[key])
        new_rmol = Chem.MolFromSmiles(smiles_dict[key], sanitize=False)
        adaptor = BabelMolAdaptor(new_pbmol)
        adaptor.make3d('uff')
        pmol = adaptor.pymatgen_mol
        try:
            pmol_graph = MoleculeGraph.with_local_env_strategy(pmol, OpenBabelNN(),
                                                               reorder=False,
                                                               extend_structure=False)
            G = pmol_graph.graph.to_undirected()
            if not nx.is_connected(G):
                print(key)
                for m in range(len(pmol_graph.molecule.sites)):
                    if pmol_graph.get_connected_sites(m) == []:
                        if pmol.sites[m].specie.name == 'Li':
                            min_dist = 100.0
                            min_atom_index = None
                            for n in range(len(pmol.sites)):
                                if pmol.sites[n].specie.name == 'O':
                                    dist = np.linalg.norm(pmol.sites[m].coords - pmol.sites[n].coords)
                                    if dist < min_dist:
                                        min_dist = dist
                                        min_atom_index = n
                            pmol_graph.add_edge(m, min_atom_index, weight=1.0, warn_duplicates=False)
            # failed when combining 2 O-O-H ?
            found = False
            fragment_name = None
            for j, unique_fragment in enumerate(unique_fragments_after_recomb):
                if unique_fragment.isomorphic_to(pmol_graph):
                    found = True
                    fragment_name = str(j)
                    break
            if not found:
                fragment_name = str(len(unique_fragments_after_recomb))
                unique_fragments_after_recomb.append(pmol_graph)
                unique_obmols_after_recomb.append(
                    new_pbmol)  # cannot convert mol graph to obmol again(some obmol.GetBondById(i) would be empty. Why?)
                unique_smiles_after_recomb.append(smiles_dict[key])
                unique_rmols_after_recomb.append(new_rmol)
            recomb_dict[key]['Fstant'] = [str(mol1_index), str(mol2_index)]
            recomb_dict[key]['product'] = [fragment_name]
        except:
            pass

    return unique_fragments_after_recomb, unique_rmols_after_recomb, unique_smiles_after_recomb, recomb_dict

def recombine_between_mol_graphs_2(mol_graphs, rmols, smiles):
    '''
    Use for cases where rmols and smiles are known
    :param mol_graphs:
    :param rmols:
    :param smiles:
    :return:
    '''
    unique_fragments_after_recomb = mol_graphs.copy()
    unique_rmols_after_recomb = rmols.copy()
    unique_smiles_after_recomb = smiles.copy()

    smiles_dict = generate_all_combos(unique_rmols_after_recomb)
    recomb_dict = {}
    for i, key in enumerate(smiles_dict.keys()):
        recomb_dict[key] = {'reactant': [], 'product': []}
        indexs = key.split('_')
        mol1_index = indexs[0]
        mol2_index = indexs[1]
        new_pbmol = pybel.readstring("smi", smiles_dict[key]).OBMol
        # new_pbmol = convert_obmol_to_normal(new_pbmol)
        # new_rmol = Chem.MolFromSmiles(smiles_dict[key])
        new_rmol = Chem.MolFromSmiles(smiles_dict[key], sanitize=False)
        adaptor = BabelMolAdaptor(new_pbmol)
        adaptor.make3d('uff')
        pmol = adaptor.pymatgen_mol
        try:
            pmol_graph = MoleculeGraph.with_local_env_strategy(pmol, OpenBabelNN(),
                                                               reorder=False,
                                                               extend_structure=False)
            # failed when combining 2 O-O-H ?
            found = False
            fragment_name = None
            for j, unique_fragment in enumerate(unique_fragments_after_recomb):
                if unique_fragment.isomorphic_to(pmol_graph):
                    found = True
                    fragment_name = str(j)
                    break
            if not found:
                fragment_name = str(len(unique_fragments_after_recomb))
                unique_fragments_after_recomb.append(pmol_graph)
                # cannot convert mol graph to obmol again(some obmol.GetBondById(i) would be empty. Why?)
                unique_smiles_after_recomb.append(smiles_dict[key])
                unique_rmols_after_recomb.append(new_rmol)
            recomb_dict[key]['reactant'] = [str(mol1_index), str(mol2_index)]
            recomb_dict[key]['product'] = [fragment_name]
        except:
            pass

    return unique_fragments_after_recomb, unique_rmols_after_recomb, unique_smiles_after_recomb, recomb_dict

if __name__ == '__main__':


    LiEC_neutral = Molecule.from_file('/Users/xiaowei_xie/PycharmProjects/electrolyte/fragmentation/LiEC_neutral.xyz')
    connected_mols, connected_mol_graphs = generate_all_unique_connected_fragments(LiEC_neutral, depth=1,
                                                                                   open_rings=True)

    rmols, smiles_list = get_rdkit_mols(connected_mol_graphs)

    smiles_dict = generate_all_combos(rmols)
    #path = '/Users/xiaowei_xie/PycharmProjects/electrolyte/bond_order_change/recomb_depth_2_pymatgen/'
    new_pbmols = []
    pmols = []
    pmol_graphs = []
    for i, key in enumerate(smiles_dict.keys()):
        new_pbmol = pybel.readstring("smi", smiles_dict[key]).OBMol
        new_pbmols.append(new_pbmol)
        adaptor = BabelMolAdaptor(new_pbmol)
        adaptor.make3d('uff')
        pmol = adaptor.pymatgen_mol
        pmol_graph = MoleculeGraph.with_local_env_strategy(pmol, OpenBabelNN(),
                                                           reorder=False,
                                                           extend_structure=False)
        if not check_in_list(pmol_graph, pmol_graphs):
            pmols.append(pmol)
            pmol_graphs.append(pmol_graph)
            #pmol.to(fmt='xyz', filename=path + key + '.xyz')


    for i in range(obmol.NumAtoms()):
        print(i, obmol.GetAtomById(i).GetAtomicNum())

    for i in range(obmol.NumBonds()):
        begin_index = obmol.GetBondById(i).GetBeginAtom().GetIdx()
        end_index = obmol.GetBondById(i).GetEndAtom().GetIdx()
        begin_atom = obmol.GetBondById(i).GetBeginAtom().GetAtomicNum()
        end_atom = obmol.GetBondById(i).GetEndAtom().GetAtomicNum()
        order = obmol.GetBondById(i).GetBondOrder()
        print(i, begin_index, end_index, order, begin_atom, end_atom)

    for i in range(obmol_copy.NumBonds()):
        begin_index = obmol_copy.GetBondById(i).GetBeginAtom().GetIdx()
        end_index = obmol_copy.GetBondById(i).GetEndAtom().GetIdx()
        begin_atom = obmol_copy.GetBondById(i).GetBeginAtom().GetAtomicNum()
        end_atom = obmol_copy.GetBondById(i).GetEndAtom().GetAtomicNum()
        order = obmol_copy.GetBondById(i).GetBondOrder()
        print(i, begin_index, end_index, order, begin_atom, end_atom)

    for i in range(new_obmol.NumBonds()):
        begin_index = new_obmol.GetBondById(i).GetBeginAtom().GetIdx()
        end_index = new_obmol.GetBondById(i).GetEndAtom().GetIdx()
        begin_atom = new_obmol.GetBondById(i).GetBeginAtom().GetAtomicNum()
        end_atom = new_obmol.GetBondById(i).GetEndAtom().GetAtomicNum()
        order = new_obmol.GetBondById(i).GetBondOrder()
        print(i, begin_index, end_index, order, begin_atom, end_atom)