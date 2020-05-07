# coding: utf-8
# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.

import logging
import numpy as np
from pymatgen.analysis.graphs import MoleculeGraph, MolGraphSplitError
from pymatgen.analysis.local_env import OpenBabelNN
from pymatgen.io.babel import BabelMolAdaptor
from pymatgen import Molecule
from pymatgen.analysis.fragmenter import Fragmenter
from pymatgen.analysis.fragmenter import metal_edge_extender
from monty.serialization import dumpfn, loadfn
import os
import copy
from itertools import repeat,product,combinations_with_replacement
from rdkit import Chem
import networkx as nx
from networkx.readwrite import json_graph
from rdkit.Chem import AllChem
from atomate.qchem.database import QChemCalcDb
from pymatgen.entries.mol_entry import MoleculeEntry
from pymatgen.analysis.reaction_network.utils import BabelMolAdaptor2, MoleculeWrapper

__author__ = "Xiaowei Xie"
__copyright__ = "Copyright 2020, The Materials Project"
__version__ = "1.0"
__maintainer__ = "Xiaowei Xie"
__email__ = "xxw940407@icloud.com"
__status__ = "Alpha"
__date__ = "05/01/19"

logger = logging.getLogger(__name__)

class Fragment_Recombination:
    """
    Class for fragment recombination.

    """
    def __init__(self, mol_graphs):
        '''
        Initialize with the mol_graphs to recombine.
        :param mol_graphs: [MoleculeGraph]
        '''
        self.mol_graphs = mol_graphs
        return

    def query_database(self,db_file="/Users/xiaoweixie/Desktop/sam_db.json", save=False, entries_name="smd_target_entries_test"):
        mmdb = QChemCalcDb.from_db_file(db_file, admin=True)
        self.target_entries = list(mmdb.collection.find({"environment": "smd_18.5,1.415,0.00,0.735,20.2,0.00,0.00"}))[:5]
        print(len(self.target_entries), "production entries")
        if save:
            dumpfn(self.target_entries,entries_name+".json")
        return

    def get_optimized_structures(self,energy_dict_name='free_energy_dict',
                                 opt_to_orig_dict_name='opt_to_orig_index_dict',
                                 load_entries=False, entries_name="smd_target_entries"):
        '''
        For getting optimized structures that are isomorphic to provided mol graphs in the mongodb database.
        Need to call this before recombination.
        :return: self.opt_mol_graphs
        '''
        # Make a dict that provides the map between optimized structures in self.opt_mol_graphs and
        # original mol graphs in self.mol_graphs. {opt_mol_graph index: mol_graph index}.
        self.opt_to_orig_keys = {}
        self.opt_mol_graphs = []
        info_dict = {}
        for i in range(len(self.mol_graphs)):
            info_dict[i] = {}
            info_dict[i][1] = {"index":None, "free_energy":1e8}
            info_dict[i][-1] = {"index": None, "free_energy": 1e8}
            info_dict[i][0] = {"index": None, "free_energy": 1e8}
        if load_entries:
            self.target_entries = loadfn(entries_name+".json")
            for i, entry in enumerate(self.target_entries):
                for j, mol_graph in enumerate(self.mol_graphs):
                    if "mol_graph" in entry:
                        mol_entry = MoleculeEntry(molecule=entry["molecule"],
                                                  energy=entry["energy_Ha"],
                                                  mol_doc={"mol_graph": entry["mol_graph"],
                                                           "enthalpy_kcal/mol": entry["enthalpy_kcal/mol"],
                                                           "entropy_cal/molK": entry["entropy_cal/molK"],
                                                           "task_id": entry["task_id"]})
                        if mol_entry.molecule.composition.alphabetical_formula == mol_graph.molecule.composition.alphabetical_formula:
                            mol_graph_in_db = mol_entry.mol_graph
                            total_charge = mol_entry.charge
                            if mol_graph_in_db.isomorphic_to(mol_graph):
                                free_energy = mol_entry.free_energy
                                if free_energy < info_dict[j][total_charge]["free_energy"]:
                                    info_dict[j][total_charge]["free_energy"] = free_energy
                                    info_dict[j][total_charge]["index"] = i
        else:
            for i, entry in enumerate(self.target_entries):
                for j, mol_graph in enumerate(self.mol_graphs):
                    if "mol_graph" in entry:
                        mol_entry = MoleculeEntry(molecule=Molecule.from_dict(entry["molecule"]),
                                                  energy=entry["energy_Ha"],
                                                  mol_doc={"mol_graph": MoleculeGraph.from_dict(entry["mol_graph"]),
                                                           "enthalpy_kcal/mol": entry["enthalpy_kcal/mol"],
                                                           "entropy_cal/molK": entry["entropy_cal/molK"],
                                                           "task_id": entry["task_id"]})
                        if mol_entry.molecule.composition.alphabetical_formula == mol_graph.molecule.composition.alphabetical_formula:
                            mol_graph_in_db = mol_entry.mol_graph
                            total_charge = mol_entry.charge
                            if mol_graph_in_db.isomorphic_to(mol_graph):
                                free_energy = mol_entry.free_energy
                                if free_energy < info_dict[j][total_charge]["free_energy"]:
                                    info_dict[j][total_charge]["free_energy"] = free_energy
                                    info_dict[j][total_charge]["index"] = i
        total_charges = [1,0,-1]
        self.free_energy_dict = {}
        # keys of self.free_energy_dict correspond to indices in self.opt_mol_graphs
        if load_entries:
            for key in info_dict.keys():
                for charge in total_charges:
                    if info_dict[key][charge]["index"] != None:
                        index = info_dict[key][charge]["index"]
                        entry = self.target_entries[index]
                        mol_entry = MoleculeEntry(molecule=entry["molecule"],
                                                  energy=entry["energy_Ha"],
                                                  mol_doc={"mol_graph": entry["mol_graph"],
                                                           "enthalpy_kcal/mol": entry["enthalpy_kcal/mol"],
                                                           "entropy_cal/molK": entry["entropy_cal/molK"],
                                                           "task_id": entry["task_id"]})
                        opt_mol_graph = mol_entry.mol_graph
                        self.free_energy_dict[len(self.opt_mol_graphs)] = mol_entry.free_energy
                        self.opt_to_orig_keys[len(self.opt_mol_graphs)] = key
                        self.opt_mol_graphs.append(opt_mol_graph)
        else:
            for key in info_dict.keys():
                for charge in total_charges:
                    if info_dict[key][charge]["index"] != None:
                        index = info_dict[key][charge]["index"]
                        entry = self.target_entries[index]
                        mol_entry = MoleculeEntry(molecule=Molecule.from_dict(entry["molecule"]),
                                                  energy=entry["energy_Ha"],
                                                  mol_doc={"mol_graph": MoleculeGraph.from_dict(entry["mol_graph"]),
                                                           "enthalpy_kcal/mol": entry["enthalpy_kcal/mol"],
                                                           "entropy_cal/molK": entry["entropy_cal/molK"],
                                                           "task_id": entry["task_id"]})
                        opt_mol_graph = mol_entry.mol_graph
                        self.free_energy_dict[len(self.opt_mol_graphs)] = mol_entry.free_energy
                        self.opt_to_orig_keys[len(self.opt_mol_graphs)] = key
                        self.opt_mol_graphs.append(opt_mol_graph)
        dumpfn(self.free_energy_dict, energy_dict_name+".json")
        dumpfn(self.opt_to_orig_keys, opt_to_orig_dict_name + ".json")
        return

    def remove_Li_bonds(self):
        '''
        Only for recombination through rdkit.
        TODO: Still not working. rdkit does not allow extra valence for other elements either.
        If Li is bonded to more than one atom, need to remove extra bonds b/c this will cause a problem for generating 3d structure from uff.
        '''
        mol_graphs_new = []
        for i, mol_graph in enumerate(self.mol_graphs):
            Li_exist = False
            Li_indices = []
            for j in range(len(mol_graph.graph.nodes)):
                if mol_graph.graph.nodes[j]['specie'] == 'Li':
                    Li_exist = True
                    Li_indices.append(j)
            if not Li_exist:
                mol_graphs_new.append(mol_graph)
            else:
                mol_graph_copy = copy.deepcopy(mol_graph)
                for li_ind in Li_indices:
                    connected_sites = mol_graph.get_connected_sites(li_ind)
                    if len(connected_sites) > 1:
                        min_dist = 1e8
                        for k, site in enumerate(connected_sites):
                            if site.dist < min_dist:
                                min_dist = site.dist
                                min_dist_site_index = k
                        for m, site in enumerate(connected_sites):
                            if m != min_dist_site_index:
                                mol_graph_copy.break_edge(li_ind, site.index, allow_reverse=True)
                mol_graphs_new.append(mol_graph_copy)
        self.mol_graphs = mol_graphs_new

        return

    def get_rmol(self, mol_graph):

        m = Chem.Mol()
        em = Chem.RWMol(m)
        for site in mol_graph.molecule:
            atomno = site.specie.Z
            em.AddAtom(Chem.Atom(atomno))

        for edge in mol_graph.graph.edges.data():
            em.AddBond(edge[0], edge[1], Chem.BondType.SINGLE)

        rmol = em.GetMol()

        return rmol

    def get_structure(self, mol_graph):
        '''
        Convert a mol graph to a schrodinger structure, with the 3d coords information.
        :param mol_graph: MoleculeGraph
        :return: [schrodinger.structure]
        '''
        from schrodinger import structure
        struct = structure.create_new_structure(num_atoms=0)
        for site in mol_graph.molecule:
            symbol = site.specie.name
            coords = site.coords
            struct.addAtom(symbol,coords[0],coords[1],coords[2])

        for edge in mol_graph.graph.edges.data():
            struct.addBond(edge[0]+1,edge[1]+1,1)

        return struct

    def get_combined_rmol(self, frag1, frag2, index1, index2, gen_3d=True):
        '''
        Make a direct recombination of two mol_graphs and add a bond between atom index1 in frag1 and atom index2 in frag2.
        Output a rdkit Mol.
        :param frag1 (MoleculeGraph)
        :param frag2 (MoleculeGraph)
        :return: Mol(rdkit Molecule)
        '''
        m = Chem.Mol()
        em = Chem.RWMol(m)
        for site in frag1.molecule:
            atomno = site.specie.Z
            em.AddAtom(Chem.Atom(atomno))
        for site in frag2.molecule:
            atomno = site.specie.Z
            em.AddAtom(Chem.Atom(atomno))

        for edge in frag1.graph.edges.data():
            em.AddBond(edge[0], edge[1], Chem.BondType.SINGLE)
        for edge in frag2.graph.edges.data():
            em.AddBond(edge[0]+len(frag1.molecule), edge[1]+len(frag1.molecule), Chem.BondType.SINGLE)
        em.AddBond(index1, index2+len(frag1.molecule),Chem.BondType.SINGLE)
        rmol = em.GetMol()

        if gen_3d:
            # Li cannot have valence 2 for uff opt. UpdatePropertyCache() will give an error.
            rmol.UpdatePropertyCache()
            Chem.GetSymmSSSR(rmol)
            rmol.GetRingInfo().NumRings()
            AllChem.EmbedMolecule(rmol)
            AllChem.UFFOptimizeMolecule(rmol)
            print(Chem.MolToMolBlock(rmol))

        return rmol

    def get_combined_schrodinger_structure(self, frag1, frag2, index1, index2, gen_3d=True):
        '''
        Make a direct recombination of two mol_graphs and add a bond between atom index1 in frag1 and atom index2 in frag2.
        Output a rdkit Mol.
        :param frag1 (MoleculeGraph)
        :param frag2 (MoleculeGraph)
        :return: Structure (schrodinger.structure)
        '''

        from schrodinger import structure

        struct = structure.create_new_structure(num_atoms=0)
        for site in frag1.molecule:
            symbol = site.specie.name
            struct.addAtom(symbol,0,0,0)
        for site in frag2.molecule:
            symbol = site.specie.name
            struct.addAtom(symbol, 0, 0, 0)
        for edge in frag1.graph.edges.data():
            struct.addBond(edge[0]+1,edge[1]+1,1)
        for edge in frag2.graph.edges.data():
            struct.addBond(edge[0]+1+len(frag1.molecule),edge[1]+1+len(frag1.molecule),1)
        struct.addBond(index1+1, index2 + 1 + len(frag1.molecule), 1)

        if gen_3d:
            from schrodinger.infra import fast3d
            fast3d_volumizer = fast3d.Volumizer()
            fast3d_volumizer.run(struct, False, False)

        return struct

    def build_mol_graph_from_two_fragments_through_rdkit(self, frag1, frag2, index1, index2, gen_3d=True):
        '''
        Build a recombined mol graph given two fragment mol graphs and the atom indices to recombine.
        This is through rdkit to create 3d structure.
        TODO: Generate 3d not working if Li has valence >= 2, O has valence >= 3 etc.

        :param frag1 (MoleculeGraph): fragment1
        :param frag2 (MoleculeGraph): fragment2
        :param index1 (int): atom index in fragment1
        :param index2 (int): atom index in fragment2
        :param gen_3d (bool): whether to generate 3d structure
        :return: (MoleculeGraph) recombined mol graph
        '''

        rmol = self.get_combined_rmol(frag1, frag2, index1, index2, gen_3d)
        graph = nx.MultiDiGraph(edge_weight_name="bond_length",
                                edge_weight_units="Å",
                                name="bonds")

        graph.add_nodes_from(range(len(frag1.molecule)))
        graph.add_nodes_from(range(len(frag1.molecule), len(frag1.molecule) + len(frag2.molecule)))
        for edge in frag1.graph.edges.data():
            graph.add_edge(edge[0], edge[1], **edge[2])
        for edge in frag2.graph.edges.data():
            graph.add_edge(edge[0] + len(frag1.molecule), edge[1] + len(frag1.molecule), **edge[2])
        graph.add_edge(index1, index2 + len(frag1.molecule), **{'weight': 1})

        species = {}
        for node in range(len(frag1.molecule)):
            specie = frag1.molecule[node].specie.symbol
            species[node] = specie
        for node in range(len(frag2.molecule)):
            specie = frag2.molecule[node].specie.symbol
            species[node + len(frag1.molecule)] = specie

        properties = {}
        for node in range(len(frag1.molecule)):
            prop = frag1.molecule[node].properties
            properties[node] = prop
        for node in range(len(frag2.molecule)):
            prop = frag2.molecule[node].properties
            properties[node + len(frag1.molecule)] = prop

        coords = {}
        for i in range(len(graph.nodes)):
            pos = rmol.GetConformer().GetAtomPosition(i)
            coord = np.array([pos.x, pos.y, pos.z])
            coords[i] = coord

        nx.set_node_attributes(graph, species, "specie")
        nx.set_node_attributes(graph, coords, 'coords')
        nx.set_node_attributes(graph, properties, "properties")

        graph_data = json_graph.adjacency_data(graph)

        new_mol = Molecule(species=species, coords=coords)

        return MoleculeGraph(new_mol, graph_data=graph_data)

    def build_mol_graph_from_two_fragments_through_schrodinger(self, frag1, frag2, index1, index2, gen_3d=True):
        '''
        Build a recombined mol graph given two fragment mol graphs and the atom indices to recombine.
        This is through schrodinger to create 3d structure.
        :param frag1 (MoleculeGraph): fragment1
        :param frag2 (MoleculeGraph): fragment2
        :param index1 (int): atom index in fragment1
        :param index2 (int): atom index in fragment2
        :param gen_3d (bool): whether to generate 3d structure
        :return: (MoleculeGraph) recombined mol graph, structure(schrodinger.Structure)
        '''

        structure = self.get_combined_schrodinger_structure(frag1, frag2, index1, index2, gen_3d)
        graph = nx.MultiDiGraph(edge_weight_name="bond_length",
                                edge_weight_units="Å",
                                name="bonds")

        graph.add_nodes_from(range(len(frag1.molecule)))
        graph.add_nodes_from(range(len(frag1.molecule), len(frag1.molecule) + len(frag2.molecule)))
        for edge in frag1.graph.edges.data():
            graph.add_edge(edge[0], edge[1], **edge[2])
        for edge in frag2.graph.edges.data():
            graph.add_edge(edge[0] + len(frag1.molecule), edge[1] + len(frag1.molecule), **edge[2])
        graph.add_edge(index1, index2 + len(frag1.molecule), **{'weight': 1})

        species = {}
        for node in range(len(frag1.molecule)):
            specie = frag1.molecule[node].specie.symbol
            species[node] = specie
        for node in range(len(frag2.molecule)):
            specie = frag2.molecule[node].specie.symbol
            species[node + len(frag1.molecule)] = specie

        properties = {}
        for node in range(len(frag1.molecule)):
            prop = frag1.molecule[node].properties
            properties[node] = prop
        for node in range(len(frag2.molecule)):
            prop = frag2.molecule[node].properties
            properties[node + len(frag1.molecule)] = prop

        coords = {}
        for i in range(len(graph.nodes)):
            atom = structure.atom[i+1]
            coord = np.array([atom.x, atom.y, atom.z])
            coords[i] = coord

        nx.set_node_attributes(graph, species, "specie")
        nx.set_node_attributes(graph, coords, 'coords')
        nx.set_node_attributes(graph, properties, "properties")

        graph_data = json_graph.adjacency_data(graph)

        new_mol = Molecule(species=species, coords=coords)

        return MoleculeGraph(new_mol, graph_data=graph_data), structure

    def identify_connectable_heavy_atoms_for_rdkit(self,mols):
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
                    for k, neighbor in enumerate(neighbors):
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

                # for Li, for now only connectable if it's a single atom
                if atom.GetAtomicNum() == 3:
                    if len(atom.GetNeighbors()) == 0 and num_atoms == 1:
                        heavy_atoms_in_mol.append(j)
                # for H, only connectable if it's a single atom
                if atom.GetAtomicNum() == 1:
                    if len(atom.GetNeighbors()) == 0 and num_atoms == 1:
                        heavy_atoms_in_mol.append(j)
                # for F, only connectable if it's a single atom
                if atom.GetAtomicNum() == 9:
                    if len(atom.GetNeighbors()) == 0 and num_atoms == 1:
                        heavy_atoms_in_mol.append(j)
                # for P, if num_neighbors < 5:
                if atom.GetAtomicNum() == 15:
                    neighbors = atom.GetNeighbors()
                    num_neighbors = len(neighbors)
                    H_count = atom.GetNumExplicitHs()

                    if num_neighbors + H_count < 5:
                        heavy_atoms_in_mol.append(j)

            heavy_atoms_index_list.append(heavy_atoms_in_mol)

        return heavy_atoms_index_list

    def identify_connectable_heavy_atoms_for_pymatgen(self,mol_graphs):
        '''

        :param mol_graphs: [MoleculeGraph]
        :return: a list of list of index of heavy atoms for each mol
        '''
        heavy_atoms_index_list = []
        for i, mol_graph in enumerate(mol_graphs):
            heavy_atoms_in_mol = []
            num_atoms = len(mol_graph.molecule)
            for j in range(num_atoms):
                connected_sites = mol_graph.get_connected_sites(j)
                num_connected_sites = len(connected_sites)
                atomic_number = mol_graph.molecule[j].specie.Z

                # for O, if num_neighbors - Li_count < 2
                if atomic_number == 8:
                    Li_count = 0
                    for k, site in enumerate(connected_sites):
                        if site.site.specie.Z == 3:
                            Li_count += 1
                    if num_connected_sites - Li_count < 2:
                        heavy_atoms_in_mol.append(j)

                # for C, if num_neighbors < 4:
                elif atomic_number == 6:
                    if num_connected_sites < 4:
                        heavy_atoms_in_mol.append(j)

                # for N, if num_neighbors - Li_count < 3
                elif atomic_number == 7:
                    Li_count = 0
                    for k, site in enumerate(connected_sites):
                        if site.site.specie.Z == 3:
                            Li_count += 1
                    if num_connected_sites - Li_count < 3:
                        heavy_atoms_in_mol.append(j)

                # for Li, for now only connectable if it's a single atom
                if atomic_number == 3:
                    if num_connected_sites == 0 and num_atoms == 1:
                        heavy_atoms_in_mol.append(j)

                # for H, only connectable if it's a single atom
                if atomic_number == 1:
                    if num_connected_sites == 0 and num_atoms == 1:
                        heavy_atoms_in_mol.append(j)

                # for F, only connectable if it's a single atom
                if atomic_number == 9:
                    if num_connected_sites == 0 and num_atoms == 1:
                        heavy_atoms_in_mol.append(j)

                # for P, if num_neighbors < 5:
                if atomic_number == 15:
                    if num_connected_sites < 5:
                        heavy_atoms_in_mol.append(j)

            heavy_atoms_index_list.append(heavy_atoms_in_mol)

        return heavy_atoms_index_list

    def generate_all_combinations_for_rdkit(self,mols):
        '''
        Generate all combination of molecule/atom indices that can participate in recombination,
         by looping through all molecule pairs(including a mol and itself) and all connectable heavy atoms in each molecule.
        :param mols: a list of rdkit mol object
        :return: list of string [ 'mol1_index'+'_'+'mol2_index'+'_'+'atom1_index'+'_'+'atom2_index'].
        '''
        final_list = []
        heavy_atoms_index_list = self.identify_connectable_heavy_atoms_for_rdkit(mols)
        num_mols = len(mols)
        all_mol_pair_index = list(combinations_with_replacement(range(num_mols), 2))
        for pair_index in all_mol_pair_index:
            pair_key = str(pair_index[0]) + '_' + str(pair_index[1])
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
                        # Not allowing Li to combine with Li,P. Li and H are connectable only if they are both single atoms.
                        # This was set in self.identify_connectable_heavy_atoms_without_sanitize function.
                        if specie1 == specie2 == 3:
                            continue
                        elif specie1 == 3 and specie2 == 15:
                            continue
                        elif specie2 == 3 and specie1 == 15:
                            continue
                        elif specie1 == specie2 == 15:
                            continue

                        atom_index_key = str(atom1) + '_' + str(atom2)
                        name = pair_key + '_' + atom_index_key
                        final_list.append(name)

        return final_list

    def generate_all_combinations_for_pymatgen(self,mol_graphs):
        '''
        Generate all combination of molecule/atom indices that can participate in recombination,
         by looping through all molecule pairs(including a mol and itself) and all connectable heavy atoms in each molecule.
        :param mol_graphs: [MoleculeGraph]
        :return: list of string [ 'mol1_index'+'_'+'mol2_index'+'_'+'atom1_index'+'_'+'atom2_index'].
        '''
        final_list = []
        heavy_atoms_index_list = self.identify_connectable_heavy_atoms_for_pymatgen(mol_graphs)
        num_mols = len(mol_graphs)
        all_mol_pair_index = list(combinations_with_replacement(range(num_mols), 2))
        for pair_index in all_mol_pair_index:
            pair_key = str(pair_index[0]) + '_' + str(pair_index[1])
            mol_graph1 = mol_graphs[pair_index[0]]
            mol_graph2 = mol_graphs[pair_index[1]]
            heavy_atoms_1 = heavy_atoms_index_list[pair_index[0]]
            heavy_atoms_2 = heavy_atoms_index_list[pair_index[1]]
            if len(heavy_atoms_1) == 0 or len(heavy_atoms_2) == 0:
                pass
            else:
                for i, atom1 in enumerate(heavy_atoms_1):
                    for j, atom2 in enumerate(heavy_atoms_2):
                        specie1 = mol_graph1.molecule[atom1].specie.Z
                        specie2 = mol_graph2.molecule[atom2].specie.Z
                        # Not allowing Li to combine with Li,P. Li and H are connectable only if they are both single atoms.
                        # This was set in self.identify_connectable_heavy_atoms_without_sanitize function.
                        if specie1 == specie2 == 3:
                            continue
                        elif specie1 == 3 and specie2 == 15:
                            continue
                        elif specie2 == 3 and specie1 == 15:
                            continue
                        elif specie1 == specie2 == 15:
                            continue

                        atom_index_key = str(atom1) + '_' + str(atom2)
                        name = pair_key + '_' + atom_index_key
                        final_list.append(name)

        return final_list

    def recombine_between_mol_graphs_for_rdkit(self):
        '''
        Generate all possible recombined mol_graphs from a list of mol_graphs.
        :param mol_graphs: [MoleculeGraph]
        :return: recomb_mol_graphs: [MoleculeGraph],
        recomb_dict: {'mol1_index'+'_'+'mol2_index'+'_'+'atom1_index'+'_'+'atom2_index': mol_graph index in the previous list}.
        '''

        rmols = [self.get_rmol(mol_graph) for mol_graph in self.mol_graphs]
        keys = self.generate_all_combinations_for_rdkit(rmols)

        self.recomb_mol_graphs = []
        self.recomb_dict = {}
        for key in keys:
            print(key)
            inds = key.split('_')
            mol_ind1, mol_ind2, atom_ind1, atom_ind2 = int(inds[0]), int(inds[1]), int(inds[2]), int(inds[3])
            recomb_mol_graph = self.build_mol_graph_from_two_fragments_through_rdkit(
                self.mol_graphs[mol_ind1], self.mol_graphs[mol_ind2], atom_ind1, atom_ind2)
            for i, mol_graph in enumerate(self.recomb_mol_graphs):
                if (mol_graph.molecule.composition.alphabetical_formula == recomb_mol_graph.molecule.composition.alphabetical_formula) and \
                    mol_graph.isomorphic_to(recomb_mol_graph):
                    self.recomb_dict[key] = i
                else:
                    self.recomb_dict[key] = len(self.recomb_mol_graphs)
                    self.recomb_mol_graphs.append(recomb_mol_graph)
        dumpfn(self.recomb_dict,'recomb_dict.json')

        return

    def recombine_between_mol_graphs_through_schrodinger(self):
        '''
        Generate all possible recombined mol_graphs from a list of mol_graphs through Schrodinger (to generate 3d structures).
        :param mol_graphs: [MoleculeGraph]
        :return: recomb_mol_graphs: [MoleculeGraph],
        recomb_dict: {'mol1_index'+'_'+'mol2_index'+'_'+'atom1_index'+'_'+'atom2_index': mol_graph index in the previous list}.
        '''
        self.opt_structs = [self.get_structure(mol_graph) for mol_graph in self.opt_mol_graphs]
        assert len(self.opt_structs) == len(self.opt_mol_graphs)

        keys = self.generate_all_combinations_for_pymatgen(self.mol_graphs)

        self.recomb_mol_graphs = []
        self.recomb_structs = []
        self.recomb_dict = {}
        for key in keys:
            print(key)
            inds = key.split('_')
            mol_ind1, mol_ind2, atom_ind1, atom_ind2 = int(inds[0]), int(inds[1]), int(inds[2]), int(inds[3])
            recomb_mol_graph, recomb_struct = self.build_mol_graph_from_two_fragments_through_schrodinger(
                self.mol_graphs[mol_ind1], self.mol_graphs[mol_ind2], atom_ind1, atom_ind2, True)
            found = False
            #recomb_struct.write(os.path.join(sdf_path, key+'.sdf'))
            for i, mol_graph in enumerate(self.recomb_mol_graphs):
                if (mol_graph.molecule.composition.alphabetical_formula == recomb_mol_graph.molecule.composition.alphabetical_formula) and \
                    mol_graph.isomorphic_to(recomb_mol_graph):
                    self.recomb_dict[key] = i + len(self.opt_mol_graphs)
                    found = True
            if not found:
                self.recomb_dict[key] = len(self.recomb_mol_graphs) + len(self.opt_mol_graphs)
                self.recomb_mol_graphs.append(recomb_mol_graph)
                self.recomb_structs.append(recomb_struct)
        #dumpfn(self.recomb_dict,recomb_name+'.json')
        assert len(self.recomb_structs) == len(self.recomb_mol_graphs)

        return

    def generate_files_for_BDE_prediction(self,sdf_name='recomb_mols', charge_file_name='total_charges',
                                          reaction_file_name='reactions', recomb_dict_name='recomb_dict'):
        '''
        This is for integrating with Mingjian's model for predicting bond dissociation energies.
        Each recombined structure has to be repeated 3 times for different charge states.
        Write sdf file for all structures in self.total_structs.
        :return:
        '''
        import csv
        #from schrodinger import structure

        self.total_mol_graphs = self.opt_mol_graphs + self.recomb_mol_graphs + self.recomb_mol_graphs + self.recomb_mol_graphs
        self.total_sdf_string = ""
        for i, mol_graph in enumerate(self.total_mol_graphs):
            MW = MoleculeWrapper()
            MW.mol_graph = mol_graph
            MW.pymatgen_mol = mol_graph.molecule
            sdf_string = MW.write(message="index: "+str(i))
            self.total_sdf_string += sdf_string
        f = open(sdf_name+'.sdf', "w")
        f.write(self.total_sdf_string)
        f.close()

        #self.total_structs = self.opt_structs + self.recomb_structs + self.recomb_structs + self.recomb_structs
        self.total_charges = []
        for i, mol_graph in enumerate(self.opt_mol_graphs):
            charge = mol_graph.molecule.charge
            self.total_charges.append(charge)
        self.total_charges += [1] * len(self.recomb_structs)
        self.total_charges += [0] * len(self.recomb_structs)
        self.total_charges += [-1] * len(self.recomb_structs)
        assert len(self.total_charges)  == len(self.total_mol_graphs) # == len(self.total_structs)
        # with structure.StructureWriter(sdf_name + ".sdf") as writer:
        #     for st in self.total_structs:
        #         writer.append(st)

        f = open(charge_file_name, "w")
        for charge in self.total_charges:
            f.write(str(charge)+"\n")
        f.close()

        self.all_reactions = []
        self.new_recomb_dict = {}
        with open(reaction_file_name+".csv", "w") as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            filewriter.writerow(
                ["reactant", "fragment1", "fragment2"])
            for key in self.recomb_dict.keys():
                mol_ind1, mol_ind2, atom_ind1, atom_ind2 = key.split('_')
                for i, mol_graph1 in enumerate(self.opt_mol_graphs):
                    if self.opt_to_orig_keys[i] == int(mol_ind1):
                        charge1 = mol_graph1.molecule.charge
                        for j, mol_graph2 in enumerate(self.opt_mol_graphs):
                            if self.opt_to_orig_keys[j] == int(mol_ind2):
                                charge2 = mol_graph2.molecule.charge
                                fragments = sorted([i,j])
                                if charge1 + charge2 in [-1,0,1]:
                                    if charge1 + charge2 == 1:
                                        reactant = self.recomb_dict[key]
                                    elif charge1 + charge2 == 0:
                                        reactant = self.recomb_dict[key] + len(self.recomb_structs)
                                    elif charge1 + charge2 == -1:
                                        reactant = self.recomb_dict[key] + 2*len(self.recomb_structs)
                                    reactant_fragments = [reactant] + fragments
                                    if reactant_fragments not in self.all_reactions:
                                        self.all_reactions.append(reactant_fragments)
                                        filewriter.writerow(reactant_fragments)
                                        self.new_recomb_dict[str(i)+"_"+str(j)+"_"+atom_ind1+"_"+atom_ind2] = reactant
                                else:
                                    continue
        print('Total number of reactions:',len(self.all_reactions))
        dumpfn(self.new_recomb_dict, recomb_dict_name+'.json')
        return

    def to_xyz(self, recomb_path='recomb_mols'):
        if not os.path.isdir(recomb_path):
            os.mkdir(recomb_path)
        for i, mol_graph in enumerate(self.total_mol_graphs):
            mol_graph.molecule.to('xyz',os.path.join(recomb_path, str(i)+'.xyz'))


    def visualize_obmol(self,obmol):
        for i in range(obmol.NumAtoms()):
            print(i, obmol.GetAtomById(i).GetAtomicNum())
        for i in range(obmol.NumBonds()):
            begin_index = obmol.GetBondById(i).GetBeginAtom().GetIdx()
            end_index = obmol.GetBondById(i).GetEndAtom().GetIdx()
            begin_atom = obmol.GetBondById(i).GetBeginAtom().GetAtomicNum()
            end_atom = obmol.GetBondById(i).GetEndAtom().GetAtomicNum()
            order = obmol.GetBondById(i).GetBondOrder()
            print(i, begin_index, end_index, begin_atom, end_atom, order)
        return

    def visualize_rmol(self,rmol):
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
            print(i, begin_index, end_index, begin_atom, end_atom, bond_order)

        return


if __name__== '__main__':
    LiEC_neutral = Molecule.from_file('/Users/xiaoweixie/PycharmProjects/electrolyte/fragmentation/LiEC_neutral.xyz')
    LiEC_neutral_graph = MoleculeGraph.with_local_env_strategy(LiEC_neutral, OpenBabelNN(),
                                                               reorder=False,
                                                               extend_structure=False)
    LiEC_neutral_graph_extender = metal_edge_extender(LiEC_neutral_graph)

    Li = Molecule.from_file('/Users/xiaoweixie/PycharmProjects/electrolyte/reaction_mechanism_final_3/Li.xyz')
    Li_graph = MoleculeGraph.with_local_env_strategy(Li, OpenBabelNN(),
                                                     reorder=False,
                                                     extend_structure=False)

    hydrogen = Molecule.from_file('/Users/xiaoweixie/PycharmProjects/electrolyte/reaction_mechanism_final_3/H.xyz')
    hydrogen_graph = MoleculeGraph.with_local_env_strategy(hydrogen, OpenBabelNN(),
                                                           reorder=False,
                                                           extend_structure=False)

    LPF6 = Molecule.from_file('/Users/xiaoweixie/PycharmProjects/electrolyte/PF6/LPF6.xyz')
    LPF6_graph = MoleculeGraph.with_local_env_strategy(LPF6, OpenBabelNN(),
                                                       reorder=False,
                                                       extend_structure=False)

    water = Molecule.from_file('/Users/xiaoweixie/PycharmProjects/electrolyte/PF6/water.xyz')
    water_graph = MoleculeGraph.with_local_env_strategy(water, OpenBabelNN(),
                                                        reorder=False,
                                                        extend_structure=False)

    fragments = [LiEC_neutral_graph, LiEC_neutral_graph_extender, LPF6_graph, water_graph]

    fragmenter = Fragmenter(LiEC_neutral, depth=2,open_rings=True, use_metal_edge_extender=False)
    for i, key in enumerate(fragmenter.unique_frag_dict.keys()):
        mol_graph = fragmenter.unique_frag_dict[key][0]
        fragments.append(mol_graph)

    fragmenter = Fragmenter(LPF6, depth=2)
    for i, key in enumerate(fragmenter.unique_frag_dict.keys()):
        mol_graph = fragmenter.unique_frag_dict[key][0]
        fragments.append(mol_graph)

    fragmenter = Fragmenter(water, depth=1)
    for i, key in enumerate(fragmenter.unique_frag_dict.keys()):
        mol_graph = fragmenter.unique_frag_dict[key][0]
        fragments.append(mol_graph)

    unique_frags = []
    for frag in fragments:
        found = False
        for existing_frag in unique_frags:
            if frag.isomorphic_to(existing_frag):
                found = True
        if not found:
            unique_frags.append(frag)
    # for i,mol_graph in enumerate(unique_frags):
    #     mol_graph.molecule.to('xyz','/Users/xiaoweixie/pymatgen/pymatgen/analysis/reaction_network/recombination/mgcf/orig_frags/'+str(i)+'.xyz')

    FR = Fragment_Recombination(unique_frags)
    #FR.query_database(save=True)
    '''
    # FR.remove_Li_bonds()
    # FR.recombine_between_mol_graphs()
    # FR.to_xyz()
    rmol = FR.get_combined_rmol(unique_frags[0],unique_frags[0],1,1,False)

    # rmol = FR.get_rmol(unique_frags[0])
    #
    # w = Chem.SDWriter('/Users/xiaoweixie/pymatgen/pymatgen/analysis/reaction_network/recombination/test_mols/recomb_0_0_1_1.sdf')
    # w.write(rmol)
    # w.flush()
    #
    suppl = Chem.SDMolSupplier('/Users/xiaoweixie/pymatgen/pymatgen/analysis/reaction_network/recombination/test_mols/output_mono.sdf',removeHs=False)
    rmol = suppl[0]
    AllChem.EmbedMolecule(rmol)
    AllChem.UFFOptimizeMolecule(rmol)
    #
    w = Chem.SDWriter('/Users/xiaoweixie/pymatgen/pymatgen/analysis/reaction_network/recombination/test_mols/output_mono_uff.sdf')
    w.write(rmol)
    w.flush()'''

    LiH = Molecule.from_file('/Users/xiaoweixie/Desktop/Sam_production/xyzs/LiH.xyz')
    LiH_graph = MoleculeGraph.with_local_env_strategy(
        LiH,
        OpenBabelNN(),
        reorder=False,
        extend_structure=False)


    path = '/Users/xiaoweixie/pymatgen/pymatgen/analysis/reaction_network/recombination/mgcf/test1/recomb_mols/'
    for i in range(5258):
        mol = Molecule.from_file(path+str(i)+'.xyz')
        mol_graph = MoleculeGraph.with_local_env_strategy(
        mol,
        OpenBabelNN(),
        reorder=False,
        extend_structure=False)
        if mol_graph.molecule.composition.alphabetical_formula == LiH.composition.alphabetical_formula:
            if mol_graph.isomorphic_to(LiH_graph):
                print(i)
