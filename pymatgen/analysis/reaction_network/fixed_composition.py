# coding: utf-8
# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.

import logging
import numpy as np
from pymatgen.analysis.graphs import MoleculeGraph, MolGraphSplitError
from pymatgen.analysis.fragmenter import open_ring
from pymatgen.analysis.local_env import OpenBabelNN
from pymatgen import Molecule
from pymatgen.analysis.fragmenter import metal_edge_extender
import networkx as nx
from pymatgen.analysis.reaction_network.fragment_recombination import Fragment_Recombination
from pymatgen.entries.mol_entry import MoleculeEntry
from atomate.qchem.database import QChemCalcDb
from monty.serialization import dumpfn, loadfn
from graphviz import Digraph
from itertools import combinations_with_replacement, product
import copy
from networkx.readwrite import json_graph
import os


__author__ = "Xiaowei Xie"
__copyright__ = "Copyright 2019, The Materials Project"
__version__ = "1.0"
__maintainer__ = "Xiaowei Xie"
__email__ = "xxw940407@icloud.com"
__status__ = "Alpha"
__date__ = "05/09/20"

logger = logging.getLogger(__name__)

class FixedCompositionNetwork:
    """
    Class for creating a fixed composition reaction network. Species to include comes from fragmentation and recombination.
    Important species for a system are identified by
    (1) Choosing a relevant chemical composition, which is specified by the user. composition = number of elements, e.g. 10 carbon + 8 oxygen + 1 lithium.
    (2) Searching through all the species in the fragmentation-recombination range, further filtered by existence of those species in the database.
    (3) Finding the n lowest energy ensemble of molecules that matches the chemical composition. ensemble of molecules = a mixture of molecules, e.g. 2EC + 2Li + 2H2O.
    (4) Chasing back the resulting ensemble of molecules to user-specified starting materials using the stored bond-breaking/forming relationship, to make sure they
        are accessible from bond-breaking/forming reactions. The total number of bonds allowed to break/form is not limited, however each molecule is only allowed to
        break/form one bond or stay unchanged in each step. Below is an example:
        If we want to achieve A + B, and we know that A can be created from C - D, and B can be created from C + D,
        then we can trace {A:1,B:1} -> [{C:2}, {A:1, C:1, D:1}] in one step. Note that {C:1, D:-1, B:1} is not allowed because the number of molecule D is negative.
        We iterate this process until the starting material is hit. A reaction network can be created from this process.
    All the intermediates/starting materials/products can be considered important in this chemical space.

    Args:
        mol_graphs (List[MoleculeGraph]) : Molecule graphs to fragment
        fragmentation_depth (List[int]) : fragmentation depth for each mol graph.
        use_metal_edge_extender (bool) : whether to use metal_edge_extender for the mol graphs before fragmentation
        electron_free_energy (float): The Gibbs free energy of an electron.
            Defaults to -2.15 eV, the value at which the LiEC SEI forms.

    """
    def __init__(self, mol_graphs, fragmentation_depth, electron_free_energy=-2.15):

        self.mol_graphs = mol_graphs
        self.fragmentation_depth = fragmentation_depth
        self.electron_free_energy = electron_free_energy
        self.unique_fragments = []
        return

    def open_ring(self,mol_graph, bond):
        '''
        Function to actually open a ring using Schrodinger's generate 3d method.
        Create a schrodinger structure with the same connectivity but only removing the target bond, and generate the 3d structure from force field.
        :param mol_graph: MoleculeGraph
        :param bond: tuple(int,int)
        :return: ring opened MoleculeGraph
        '''

        from schrodinger import structure
        from schrodinger.infra import fast3d
        struct = structure.create_new_structure(num_atoms=0)
        for site in mol_graph.molecule:
            symbol = site.specie.name
            struct.addAtom(symbol, 0, 0, 0)

        graph = nx.MultiDiGraph(edge_weight_name="bond_length",
                                edge_weight_units="Å",
                                name="bonds")
        graph.add_nodes_from(range(len(mol_graph.molecule)))

        for edge in mol_graph.graph.edges.data():
            if not (edge[0] == bond[0] and edge[1] == bond[1]):
                struct.addBond(edge[0] + 1, edge[1] + 1, 1)
                graph.add_edge(edge[0], edge[1], **edge[2])
                fast3d_volumizer = fast3d.Volumizer()
                fast3d_volumizer.run(struct, False, False)

        species = {}
        for node in range(len(mol_graph.molecule)):
            specie = mol_graph.molecule[node].specie.symbol
            species[node] = specie

        properties = {}
        for node in range(len(mol_graph.molecule)):
            prop = mol_graph.molecule[node].properties
            properties[node] = prop

        coords = {}
        for i in range(len(graph.nodes)):
            atom = struct.atom[i+1]
            coord = np.array([atom.x, atom.y, atom.z])
            coords[i] = coord

        nx.set_node_attributes(graph, species, "specie")
        nx.set_node_attributes(graph, coords, 'coords')
        nx.set_node_attributes(graph, properties, "properties")

        graph_data = json_graph.adjacency_data(graph)

        new_mol = Molecule(species=species, coords=coords)

        return MoleculeGraph(new_mol, graph_data=graph_data)

    def _fragment_one_level(self,mol_graph):
        '''
        Perform one-step fragmentation for a mol_graph. Resulting mol graphs have to be connected graphs.
        Fragments that are not connected by removing Li will be removed.
        TODO: For Li coordinating to 2 F in PF6 type molecules, schrodinger cannot produce reasonable 3d structures.
              Removing for now.
        :param mol_graph (MoleculeGraph):
        :return: List of fragments. [[2,3],[4,5]]: This mol graph can be broken into 2+3 and 4+5
                Numbers in the list are indices in unique_fragments here. Indices need to be updated later on.

        '''
        def is_connected(mol_graph):
            is_connected = False
            if len(mol_graph.molecule) > 1:
                if nx.is_weakly_connected(mol_graph.graph):
                    is_connected = True
            else:
                is_connected = True
            return is_connected

        fragmentation_list = []
        unique_fragments = []
        for edge in mol_graph.graph.edges:
            bond = [(edge[0], edge[1])]
            try:
                fragments = mol_graph.split_molecule_subgraphs(bond, allow_reverse=True)
                fragments_name = []
                if all(is_connected(fragment) for fragment in fragments):
                    for fragment in fragments:
                        found = False
                        for j, unique_fragment in enumerate(unique_fragments):
                            if unique_fragment.isomorphic_to(fragment):
                                found = True
                                fragment_name = j
                                break
                        if not found:
                            fragment_name = len(unique_fragments)
                            unique_fragments.append(fragment)
                        fragments_name.append(fragment_name)
                    fragmentation_list.append(fragments_name)

            except MolGraphSplitError:
                fragment = self.open_ring(mol_graph, bond[0])
                if is_connected(fragment):
                    # Check if the fragment mol graph is still connected if Li is removed. If not connected, it should be removed.
                    is_connected_after_removing_li = True
                    for i, site in enumerate(fragment.molecule):
                        if site.specie.name == "Li":
                            fragment_copy = copy.deepcopy(fragment)
                            fragment_copy.remove_nodes([i])
                            if not is_connected(fragment_copy):
                                is_connected_after_removing_li = False
                    two_Li_F_bonds = False
                    for i, site in enumerate(fragment.molecule):
                        if site.specie.name == "Li":
                            if len(fragment.get_connected_sites(i)) == 2 and all(site.site.specie.name == 'F' for site in fragment.get_connected_sites(0)):
                                two_Li_F_bonds = True

                    if is_connected_after_removing_li and not two_Li_F_bonds:
                        found = False
                        for j, unique_fragment in enumerate(unique_fragments):
                            if unique_fragment.isomorphic_to(fragment):
                                found = True
                                fragment_name = j
                                break
                        if not found:
                            fragment_name = len(unique_fragments)
                            unique_fragments.append(fragment)
                        fragmentation_list.append([fragment_name])

        unique_fragmentation_list = []
        for item in fragmentation_list:
            if item not in unique_fragmentation_list:
                unique_fragmentation_list.append(item)
        return unique_fragmentation_list, unique_fragments

    def fragmentation(self):
        '''
        Generate all the unique fragments (including the starting molgraphs themselves) from fragmenting
        self.mol_graphs according to the depth specified in self.fragmentation_depth.
        :return: self.unique_fragments_new (List[MoleculeGraph])
                self.fragmentation_dict_new : Dictionary that maps the starting molgraph index to fragments indices.
                                              Here the indices are the indices in self.unique_fragments_new.
        '''
        self.fragmentation_dict = {}
        for i,mol_graph in enumerate(self.mol_graphs):
            previous_length = len(self.unique_fragments)
            # Need to check uniqueness of self.mol_graphs[i] later.
            self.unique_fragments += [self.mol_graphs[i]]

            for j in range(self.fragmentation_depth[i]):
                print('fragmenting fragment:',i,' depth:',j+1, flush=True)
                self.new_fragments = []
                for p, mol_graph_from_i in enumerate(self.unique_fragments):
                    if p >= previous_length:
                        if p not in self.fragmentation_dict.keys():
                            self.fragmentation_dict[p] = []
                        frag_list, frags = self._fragment_one_level(mol_graph_from_i)
                        #print('length of frags:',len(frags))
                        # dict for saving index in frags to index in self.unique_fragments
                        old_to_new_index_dict = {}
                        for k,frag in enumerate(frags):
                            found = False
                            for m, old_frag in enumerate(self.unique_fragments+self.new_fragments):
                                if frag.molecule.composition.alphabetical_formula == old_frag.molecule.composition.alphabetical_formula \
                                        and frag.isomorphic_to(old_frag):
                                    found = True
                                    old_to_new_index_dict[k] = m
                            if not found:
                                old_to_new_index_dict[k] = len(self.unique_fragments+self.new_fragments)
                                self.new_fragments.append(frag)
                        #print('frag list:',frag_list)
                        #print('old_to_new_index_dict:',old_to_new_index_dict)
                        for item in frag_list:
                            new_item = [old_to_new_index_dict[q] for q in item]
                            new_item.sort()
                            if new_item not in self.fragmentation_dict[p]:
                                self.fragmentation_dict[p].append(new_item)
                self.unique_fragments += self.new_fragments

        # Need to check the uniqueness of the molgraphs in self.unique_graphs again.
        # Because self.mol_graphs[i] might be reached from fragmentation of previous mol_graphs.
        self.unique_fragments_new = []
        self.fragmentation_dict_new = {}
        # Keep track of the index from self.unique_fragments to self.unique_fragments_new
        self.old_to_new_index_dict = {}
        for i, frag in enumerate(self.unique_fragments):
            found = False
            for j, new_frag in enumerate(self.unique_fragments_new):
                if frag.molecule.composition.alphabetical_formula == new_frag.molecule.composition.alphabetical_formula \
                        and frag.isomorphic_to(new_frag):
                    found = True
                    self.old_to_new_index_dict[i] = j
            if not found:
                self.old_to_new_index_dict[i] = len(self.unique_fragments_new)
                self.unique_fragments_new.append(frag)
        for key in self.fragmentation_dict.keys():
            print('key:',key, flush=True)
            self.fragmentation_dict_new[self.old_to_new_index_dict[key]] = []
            for item in self.fragmentation_dict[key]:
                new_item = [self.old_to_new_index_dict[p] for p in item]
                new_item.sort()
                if new_item not in self.fragmentation_dict_new[self.old_to_new_index_dict[key]]:
                    self.fragmentation_dict_new[self.old_to_new_index_dict[key]].append(new_item)

        dumpfn(self.fragmentation_dict_new, 'fragmentation_dict_new.json')
        dumpfn(self.unique_fragments_new, 'unique_fragments_new.json')
        self.to_xyz(self.unique_fragments_new, path='fragmentation_mols' )
        return self.unique_fragments_new, self.fragmentation_dict_new

    def query_database(self, db_file="/Users/xiaoweixie/Desktop/sam_db.json", save=False,
                       entries_name="smd_target_entries"):
        mmdb = QChemCalcDb.from_db_file(db_file, admin=True)
        self.target_entries = list(mmdb.collection.find({"environment": "smd_18.5,1.415,0.00,0.735,20.2,0.00,0.00"}))
        print(len(self.target_entries), "production entries", flush=True)
        if save:
            dumpfn(self.target_entries, entries_name + ".json")
        return

    def get_optimized_structures(self,energy_dict_name='free_energy_dict',
                                 opt_to_orig_dict_name='opt_to_orig_index_dict',
                                 load_entries=False, entries_name=["smd_target_entries"]):
        '''
        Different from the "get_optimized_structures" function from the FragmentRecombination class.
        Need to call this function after recombination. So call self.recombination() first.
        For getting optimized structures that are isomorphic to provided mol graphs in the mongodb database.
        Need to call this before recombination.
        If not load entries, must run self.query_database first. entries_name variable should be the same as that in self.query_database.
        :return: self.opt_entries_list (List[MoleculeEntry])
        '''
        # Make a dict that provides the map between optimized structures in self.opt_entries_list and
        # original mol graphs in self.total_mol_graphs_no_opt. {self.opt_entries_list index: mol_graph index}.

        self.opt_entries = {}
        self.opt_species_w_charge = []
        info_dict = {}
        print('Number of mol graphs:', len(self.total_mol_graphs_no_opt), flush=True)
        for i in range(len(self.total_mol_graphs_no_opt)):
            info_dict[i] = {}
            info_dict[i][1] = {"index":None, "free_energy":1e8}
            info_dict[i][-1] = {"index": None, "free_energy": 1e8}
            info_dict[i][0] = {"index": None, "free_energy": 1e8}
        if load_entries:
            self.target_entries = []
            for name in entries_name:
                self.target_entries += loadfn(name + ".json")
            for i, entry in enumerate(self.target_entries):
                for j, mol_graph in enumerate(self.total_mol_graphs_no_opt):
                    if "mol_graph" in entry:
                        if "task_id" in entry.keys():
                            task_id = entry["task_id"]
                        else:
                            task_id = 10000
                        mol_entry = MoleculeEntry(molecule=entry["molecule"],
                                                  energy=entry["energy_Ha"],
                                                  mol_doc={"mol_graph": entry["mol_graph"],
                                                           "enthalpy_kcal/mol": entry["enthalpy_kcal/mol"],
                                                           "entropy_cal/molK": entry["entropy_cal/molK"],
                                                           "task_id": task_id})
                        if mol_entry.molecule.composition.alphabetical_formula == mol_graph.molecule.composition.alphabetical_formula:
                            mol_graph_in_db = mol_entry.mol_graph
                            total_charge = mol_entry.charge
                            if mol_graph_in_db.isomorphic_to(mol_graph):
                                free_energy = mol_entry.free_energy
                                if free_energy < info_dict[j][total_charge]["free_energy"]:
                                    info_dict[j][total_charge]["free_energy"] = free_energy
                                    info_dict[j][total_charge]["index"] = i
                    else:
                        if "entropy_cal/molK" in entry.keys() and "enthalpy_kcal/mol" in entry.keys():
                            if "task_id" in entry.keys():
                                task_id = entry["task_id"]
                            else:
                                task_id = 10000
                            mol_entry = MoleculeEntry(molecule=entry["molecule"],
                                                      energy=entry["energy_Ha"],
                                                      mol_doc={"enthalpy_kcal/mol": entry["enthalpy_kcal/mol"],
                                                               "entropy_cal/molK": entry["entropy_cal/molK"],
                                                               "task_id": task_id})
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
                for j, mol_graph in enumerate(self.total_mol_graphs_no_opt):
                    if "mol_graph" in entry:
                        if "task_id" in entry.keys():
                            task_id = entry["task_id"]
                        else:
                            task_id = 10000
                        mol_entry = MoleculeEntry(molecule=entry["molecule"],
                                                  energy=entry["energy_Ha"],
                                                  mol_doc={"mol_graph": entry["mol_graph"],
                                                           "enthalpy_kcal/mol": entry["enthalpy_kcal/mol"],
                                                           "entropy_cal/molK": entry["entropy_cal/molK"],
                                                           "task_id": task_id})

                        if mol_entry.molecule.composition.alphabetical_formula == mol_graph.molecule.composition.alphabetical_formula:
                            mol_graph_in_db = mol_entry.mol_graph
                            total_charge = mol_entry.charge
                            if mol_graph_in_db.isomorphic_to(mol_graph):
                                free_energy = mol_entry.free_energy
                                if free_energy < info_dict[j][total_charge]["free_energy"]:
                                    info_dict[j][total_charge]["free_energy"] = free_energy
                                    info_dict[j][total_charge]["index"] = i
                    else:
                        if "entropy_cal/molK" in entry.keys() and "enthalpy_kcal/mol" in entry.keys():
                            if "task_id" in entry.keys():
                                task_id = entry["task_id"]
                            else:
                                task_id = 10000
                            mol_entry = MoleculeEntry(molecule=entry["molecule"],
                                                      energy=entry["energy_Ha"],
                                                      mol_doc={"enthalpy_kcal/mol": entry["enthalpy_kcal/mol"],
                                                               "entropy_cal/molK": entry["entropy_cal/molK"],
                                                               "task_id": task_id})
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
                self.opt_entries[key] = {}
                for charge in total_charges:
                    if info_dict[key][charge]["index"] != None:
                        index = info_dict[key][charge]["index"]
                        entry = self.target_entries[index]
                        if "mol_graph" in entry:
                            mol_entry = MoleculeEntry(molecule=entry["molecule"],
                                                      energy=entry["energy_Ha"],
                                                      mol_doc={"mol_graph": entry["mol_graph"],
                                                               "enthalpy_kcal/mol": entry["enthalpy_kcal/mol"],
                                                               "entropy_cal/molK": entry["entropy_cal/molK"],
                                                               "task_id": entry["task_id"]})
                        elif "entropy_cal/molK" in entry.keys() and "enthalpy_kcal/mol" in entry.keys() and "task_id" in entry.keys():
                            mol_entry = MoleculeEntry(molecule=entry["molecule"],
                                                      energy=entry["energy_Ha"],
                                                      mol_doc={"enthalpy_kcal/mol": entry["enthalpy_kcal/mol"],
                                                               "entropy_cal/molK": entry["entropy_cal/molK"],
                                                               "task_id": entry["task_id"]})

                        self.opt_entries[key][charge] = mol_entry
                        self.opt_species_w_charge.append(str(key)+'_'+str(charge))
        else:
            for key in info_dict.keys():
                self.opt_entries[key] = {}
                for charge in total_charges:
                    if info_dict[key][charge]["index"] != None:
                        index = info_dict[key][charge]["index"]
                        entry = self.target_entries[index]
                        if "mol_graph" in entry:
                            mol_entry = MoleculeEntry(molecule=entry["molecule"],
                                                      energy=entry["energy_Ha"],
                                                      mol_doc={"mol_graph": MoleculeGraph.from_dict(entry["mol_graph"]),
                                                               "enthalpy_kcal/mol": entry["enthalpy_kcal/mol"],
                                                               "entropy_cal/molK": entry["entropy_cal/molK"],
                                                               "task_id": entry["task_id"]})
                        elif "entropy_cal/molK" in entry.keys() and "enthalpy_kcal/mol" in entry.keys() and "task_id" in entry.keys():
                            mol_entry = MoleculeEntry(molecule=entry["molecule"],
                                                      energy=entry["energy_Ha"],
                                                      mol_doc={"enthalpy_kcal/mol": entry["enthalpy_kcal/mol"],
                                                               "entropy_cal/molK": entry["entropy_cal/molK"],
                                                               "task_id": entry["task_id"]})

                        self.opt_entries[key][charge] = mol_entry
                        self.opt_species_w_charge.append(str(key) + '_' + str(charge))

        dumpfn(self.opt_entries, 'opt_entries.json')
        dumpfn(self.opt_species_w_charge, 'opt_species_w_charge.json')
        return

    def get_optimized_structures_new(self,energy_dict_name='free_energy_dict',
                                 opt_to_orig_dict_name='opt_to_orig_index_dict',
                                 load_entries=False, entries_name=["smd_target_entries"]):
        '''
        Different from the previous function in that the loaded entries are already mol_entries.
        Different from the "get_optimized_structures" function from the FragmentRecombination class.
        Need to call this function after recombination. So call self.recombination() first.
        For getting optimized structures that are isomorphic to provided mol graphs in the mongodb database.
        Need to call this before recombination.
        If not load entries, must run self.query_database first. entries_name variable should be the same as that in self.query_database.
        :return: self.opt_entries_list (List[MoleculeEntry])
        '''
        # Make a dict that provides the map between optimized structures in self.opt_entries_list and
        # original mol graphs in self.total_mol_graphs_no_opt. {self.opt_entries_list index: mol_graph index}.

        self.opt_entries = {}
        self.opt_species_w_charge = []
        info_dict = {}
        print('Number of mol graphs:', len(self.total_mol_graphs_no_opt), flush=True)
        for i in range(len(self.total_mol_graphs_no_opt)):
            info_dict[i] = {}
            info_dict[i][1] = {"index":None, "free_energy":1e8}
            info_dict[i][-1] = {"index": None, "free_energy": 1e8}
            info_dict[i][0] = {"index": None, "free_energy": 1e8}
        if load_entries:
            self.target_entries = []
            for name in entries_name:
                self.target_entries += loadfn(name+".json")
            for i, entry in enumerate(self.target_entries):
                for j, mol_graph in enumerate(self.total_mol_graphs_no_opt):
                    mol_entry = entry
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
                self.opt_entries[key] = {}
                for charge in total_charges:
                    if info_dict[key][charge]["index"] != None:
                        index = info_dict[key][charge]["index"]
                        entry = self.target_entries[index]
                        mol_entry = entry
                        self.opt_entries[key][charge] = mol_entry
                        self.opt_species_w_charge.append(str(key)+'_'+str(charge))

        dumpfn(self.opt_entries, 'opt_entries.json')
        dumpfn(self.opt_species_w_charge, 'opt_species_w_charge.json')
        return

    def recombination(self):
        '''
        Recombine between mol_graphs in self.unique_fragments_new.
        :return:
        '''
        self.fragmentation()
        FR = Fragment_Recombination(self.unique_fragments_new)

        FR.recombine_between_mol_graphs_through_schrodinger_no_opt()
        self.total_mol_graphs_no_opt = FR.total_mol_graphs_no_opt
        self.recomb_dict_no_opt = FR.recomb_dict_no_opt
        dumpfn(self.total_mol_graphs_no_opt, 'total_mol_graphs_no_opt.json')
        dumpfn(self.recomb_dict_no_opt, 'recomb_dict_no_opt.json')
        FR.to_xyz(self.total_mol_graphs_no_opt,recomb_path='recomb_mols_no_opt')
        return

    def recombination_for_BDE_prediction(self,load_entries=True, entries_name="../../smd_target_entries"):
        self.fragmentation()
        FR = Fragment_Recombination(self.unique_fragments_new)
        FR.get_optimized_structures(load_entries=load_entries, entries_name=entries_name)
        FR.recombine_between_mol_graphs_through_schrodinger()
        FR.generate_files_for_BDE_prediction()
        FR.to_xyz(FR.total_mol_graphs, recomb_path='recomb_mols')
        return

    def recombination_for_BDE_prediction_from_mol_entries(self,entries_name="../../smd_target_entries"):
        self.fragmentation()
        FR = Fragment_Recombination(self.unique_fragments_new)
        FR.get_optimized_structures_from_mol_entries(entries_name=entries_name)
        FR.recombine_between_mol_graphs_through_schrodinger()
        FR.generate_files_for_BDE_prediction()
        FR.to_xyz(FR.total_mol_graphs, recomb_path='recomb_mols')
        return


    def to_xyz(self, mol_graphs, path='recomb_mols'):
        if not os.path.isdir(path):
            os.mkdir(path)
        for i, mol_graph in enumerate(mol_graphs):
            mol_graph.molecule.to('xyz',os.path.join(path, str(i)+'.xyz'))
        return

    def generate_stoichiometry_table(self):
        '''
        Generate a dictionary of stoichiometry table for self.total_mol_graphs_no_opt
        :return: self.all_stoi_dict (Dict) (key(int):index in self.total_mol_graphs_no_opt; content(dict) {'C':0,'O':0,'H':0,'Li':0, 'P':2, 'F':3})
        '''
        self.all_stoi_dict = {}
        for i, mol_graph in enumerate(self.total_mol_graphs_no_opt):
            species = [str(site.specie.symbol) for site in mol_graph.molecule]
            num_C, num_O, num_H, num_Li, num_P, num_F = 0, 0, 0, 0, 0, 0
            for j,specie in enumerate(species):
                if specie == 'C':
                    num_C += 1
                elif specie == 'O':
                    num_O += 1
                elif specie == 'H':
                    num_H += 1
                elif specie == 'Li':
                    num_Li += 1
                elif specie == 'P':
                    num_P += 1
                elif specie == 'F':
                    num_F += 1
                else:
                    raise ValueError("Element "+ specie+"not found when creating stoichiometry dict!")
            stoi_dict = {'C':num_C, 'O':num_O, 'H':num_H, 'Li':num_Li, 'P':num_P, 'F':num_F}
            self.all_stoi_dict[i] = stoi_dict
        # For electron
        self.all_stoi_dict['e'] = {'C': 0, 'O': 0, 'H': 0, 'Li': 0, 'P': 0, 'F': 0}
        return

    def split_species_dict(self,species_dict):
        '''
        Split species dict into a list of species. Species with multiple stoichiometry will be split into a multiple of same species name.
        i.e.  {'0_0': 1, '32_0': 2} -> ['0_0','32_0', '32_0']
        :param species_dict: dictionary of species with stoichiometry
        :return:
        '''
        species_list = []
        for key in species_dict.keys():
            count = species_dict[key]
            for i in range(count):
                species_list.append(key)
        return species_list

    def subtract_two_stoi_dict(self,stoi_dict1, stoi_dict2):
        new_stoi_dict = {'C': stoi_dict1['C'] - stoi_dict2['C'], 'O': stoi_dict1['O'] - stoi_dict2['O'],
                         'H': stoi_dict1['H'] - stoi_dict2['H'], 'Li': stoi_dict1['Li'] - stoi_dict2['Li'],
                         'P': stoi_dict1['P'] - stoi_dict2['P'], 'F': stoi_dict1['F'] - stoi_dict2['F']}
        return new_stoi_dict

    def reverse_species_dict(self,species_dict):
        new_species_dict = {}
        for key in species_dict.keys():
            new_species_dict[key] = - species_dict[key]
        return new_species_dict

    def merge_species_dict(self,species_dict1, species_dict2):
        merge_dict = {}
        key_merge = list(species_dict1.keys()) + list(species_dict2.keys())
        key_set = list(set(key_merge))
        for key in key_set:
            if key in species_dict1.keys() and key in species_dict2.keys():
                if species_dict1[key] + species_dict2[key] != 0:
                    merge_dict[key] = species_dict1[key] + species_dict2[key]
            elif key in species_dict1.keys():
                merge_dict[key] = species_dict1[key]
            elif key in species_dict2.keys():
                merge_dict[key] = species_dict2[key]
        return merge_dict

    def merge_multiple_species_dict(self,species_dict_list):
        num_species_dict = len(species_dict_list)
        first_species_dict = species_dict_list[0]
        for i in range(1, num_species_dict):
            second_species_dict = species_dict_list[i]
            first_species_dict = self.merge_species_dict(first_species_dict, second_species_dict)

        return first_species_dict

    def find_all_product_composition_from_target(self,target_composition, target_charge, energy_thresh):
        '''
        Find all possible combinations of molecules that match the target composition.
        The number of species is limited to <=5 molecules, <=2 electrons for now.
        Prevent the number of possible combinations to grow too large by setting a energy threshold. E.g. if we start from 2LiEC,
        we want the energy of the product to be less than the energy of 2LiEC.
        :param target_composition: {'C': 1, 'O': 2, 'H': 0, 'Li': 0, 'P': 0, 'F': 0}
        :param target_charge (int)
        :param energy_thresh (float): A crude upper bound of energy for the compositions.
        :return: all_possible_products: List[list[str(mol_index)+'_'+str(charge)]]
        all_possible_product_energies: List(float), corresponding free energies

        '''
        if target_composition == {'C': 0, 'O': 0, 'H': 0, 'Li': 0, 'P': 0, 'F': 0}:
            raise ValueError("target composition is not valid!")
        elif any(target_composition[ele] < 0 for ele in target_composition.keys()):
            raise ValueError("target composition is not valid!")
        all_possible_products = []
        all_possible_product_energies = []

        for i, mol in enumerate(self.opt_species_w_charge):
            mol_index, mol_charge = int(mol.split('_')[0]), int(mol.split('_')[1])
            print('finding composition round:', i, flush=True)
            if int(mol_charge) == target_charge and self.all_stoi_dict[mol_index] == target_composition:
                to_add = [mol]
                energy = self.opt_entries[int(mol.split('_')[0])][int(mol.split('_')[1])].free_energy
                if to_add not in all_possible_products and energy < energy_thresh:
                    all_possible_products.append(to_add)
                    all_possible_product_energies.append(energy)
            else:
                remain_stoi = self.subtract_two_stoi_dict(target_composition, self.all_stoi_dict[mol_index])
                remain_charge = target_charge - mol_charge
                if all(remain_stoi[ele] >= 0 for ele in remain_stoi):
                    for j, mol1 in enumerate(self.opt_species_w_charge[i:]):
                        mol1_index, mol1_charge = int(mol1.split('_')[0]), int(mol1.split('_')[1])
                        if (self.all_stoi_dict[mol1_index] == remain_stoi) and ((mol1_charge - remain_charge) <= 2):
                            if (mol1_charge - remain_charge) == 0:
                                to_add = [mol, mol1]
                                energy = np.sum([self.opt_entries[int(item.split('_')[0])][int(item.split('_')[1])].free_energy for item in to_add])
                                if energy < energy_thresh:
                                    to_add.sort()
                                    if to_add not in all_possible_products:
                                        all_possible_products.append(to_add)
                                        all_possible_product_energies.append(energy)
                            elif (mol1_charge - remain_charge) == 1:
                                to_add = [mol, mol1, 'e_-1']
                                energy = np.sum([self.opt_entries[int(item.split('_')[0])][int(item.split('_')[1])].free_energy for item in to_add[:2]]) + self.electron_free_energy
                                if energy < energy_thresh:
                                    to_add.sort()
                                    if to_add not in all_possible_products:
                                        all_possible_products.append(to_add)
                                        all_possible_product_energies.append(energy)
                            elif (mol1_charge - remain_charge) == 2:
                                to_add = [mol, mol1, 'e_-1', 'e_-1']
                                energy = np.sum([self.opt_entries[int(item.split('_')[0])][int(item.split('_')[1])].free_energy for item in to_add[:2]]) + self.electron_free_energy * 2
                                if energy < energy_thresh:
                                    to_add.sort()
                                    if to_add not in all_possible_products:
                                        all_possible_products.append(to_add)
                                        all_possible_product_energies.append(energy)
                        else:
                            remain_stoi1 = self.subtract_two_stoi_dict(remain_stoi, self.all_stoi_dict[mol1_index])
                            remain_charge1 = remain_charge - mol1_charge
                            if all(remain_stoi1[ele] >= 0 for ele in remain_stoi1):
                                for k, mol2 in enumerate(self.opt_species_w_charge[i + j:]):
                                    mol2_index, mol2_charge = int(mol2.split('_')[0]), int(mol2.split('_')[1])
                                    if (self.all_stoi_dict[mol2_index] == remain_stoi1) and (
                                            (mol2_charge - remain_charge1) <= 2):
                                        if (mol2_charge - remain_charge1) == 0:
                                            to_add = [mol, mol1, mol2]
                                            energy = np.sum([self.opt_entries[int(item.split('_')[0])][int(item.split('_')[1])].free_energy for item in to_add])
                                            if energy < energy_thresh:
                                                to_add.sort()
                                                if to_add not in all_possible_products:
                                                    all_possible_products.append(to_add)
                                                    all_possible_product_energies.append(energy)
                                        elif (mol2_charge - remain_charge1) == 1:
                                            to_add = [mol, mol1, mol2, 'e_-1']
                                            energy = np.sum([self.opt_entries[int(item.split('_')[0])][int(item.split('_')[1])].free_energy for item in to_add[:3]]) + self.electron_free_energy
                                            if energy < energy_thresh:
                                                to_add.sort()
                                                if to_add not in all_possible_products:
                                                    all_possible_products.append(to_add)
                                                    all_possible_product_energies.append(energy)
                                        elif (mol2_charge - remain_charge1) == 2:
                                            to_add = [mol, mol1, mol2, 'e_-1', 'e_-1']
                                            energy = np.sum([self.opt_entries[int(item.split('_')[0])][int(item.split('_')[1])].free_energy for item in to_add[:3]]) + self.electron_free_energy * 2
                                            if energy < energy_thresh:
                                                to_add.sort()
                                                if to_add not in all_possible_products:
                                                    all_possible_products.append(to_add)
                                                    all_possible_product_energies.append(energy)
                                    else:
                                        remain_stoi2 = self.subtract_two_stoi_dict(remain_stoi1, self.all_stoi_dict[mol2_index])
                                        remain_charge2 = remain_charge1 - mol2_charge
                                        if all(remain_stoi2[ele] >= 0 for ele in remain_stoi2):
                                            for m, mol3 in enumerate(self.opt_species_w_charge[i + j + k:]):
                                                mol3_index, mol3_charge = int(mol3.split('_')[0]), int(mol3.split('_')[1])
                                                if (self.all_stoi_dict[mol3_index] == remain_stoi2) and (
                                                        (mol3_charge - remain_charge2) <= 2):
                                                    if (mol3_charge - remain_charge2) == 0:
                                                        to_add = [mol, mol1, mol2, mol3]
                                                        energy = np.sum([self.opt_entries[int(item.split('_')[0])][int(item.split('_')[1])].free_energy for item in to_add])
                                                        if energy < energy_thresh:
                                                            to_add.sort()
                                                            if to_add not in all_possible_products:
                                                                all_possible_products.append(to_add)
                                                                all_possible_product_energies.append(energy)
                                                    elif (mol3_charge - remain_charge2) == 1:
                                                        to_add = [mol, mol1, mol2, mol3, 'e_-1']
                                                        energy = np.sum([self.opt_entries[int(item.split('_')[0])][int(item.split('_')[1])].free_energy for item in to_add[:4]]) + self.electron_free_energy
                                                        if energy < energy_thresh:
                                                            to_add.sort()
                                                            if to_add not in all_possible_products:
                                                                all_possible_products.append(to_add)
                                                                all_possible_product_energies.append(energy)
                                                    elif (mol3_charge - remain_charge2) == 2:
                                                        to_add = [mol, mol1, mol2, mol3, 'e_-1', 'e_-1']
                                                        energy = np.sum([self.opt_entries[int(item.split('_')[0])][int(item.split('_')[1])].free_energy for item in to_add[:4]]) + self.electron_free_energy * 2
                                                        if energy < energy_thresh:
                                                            to_add.sort()
                                                            if to_add not in all_possible_products:
                                                                all_possible_products.append(to_add)
                                                                all_possible_product_energies.append(energy)
                                                else:
                                                    remain_stoi3 = self.subtract_two_stoi_dict(remain_stoi2, self.all_stoi_dict[mol3_index])
                                                    remain_charge3 = remain_charge2 - mol3_charge
                                                    if all(remain_stoi3[ele] >= 0 for ele in remain_stoi3):
                                                        for n, mol4 in enumerate(self.opt_species_w_charge[i + j + k + m:]):
                                                            mol4_index, mol4_charge = int(mol4.split('_')[0]), int(mol4.split('_')[1])
                                                            if (self.all_stoi_dict[mol4_index] == remain_stoi3) and ((mol4_charge - remain_charge3) <= 2):
                                                                if (mol4_charge - remain_charge3) == 0:
                                                                    to_add = [mol, mol1, mol2, mol3, mol4]
                                                                    energy = np.sum([self.opt_entries[int(item.split('_')[0])][int(item.split('_')[1])].free_energy for item in to_add])
                                                                    if energy < energy_thresh:
                                                                        to_add.sort()
                                                                        if to_add not in all_possible_products:
                                                                            all_possible_products.append(to_add)
                                                                            all_possible_product_energies.append(energy)
                                                                elif (mol4_charge - remain_charge3) == 1:
                                                                    to_add = [mol, mol1, mol2, mol3, mol4, 'e_-1']
                                                                    energy = np.sum([self.opt_entries[int(item.split('_')[0])][int(item.split('_')[1])].free_energy for item in to_add[:5]]) + self.electron_free_energy
                                                                    if energy < energy_thresh:
                                                                        to_add.sort()
                                                                        if to_add not in all_possible_products:
                                                                            all_possible_products.append(to_add)
                                                                            all_possible_product_energies.append(energy)
                                                                elif (mol4_charge - remain_charge3) == 2:
                                                                    to_add = [mol, mol1, mol2, mol3, mol4, 'e_-1','e_-1']
                                                                    energy = np.sum([self.opt_entries[int(item.split('_')[0])][int(item.split('_')[1])].free_energy for item in to_add[:5]]) + self.electron_free_energy * 2
                                                                    if energy < energy_thresh:
                                                                        to_add.sort()
                                                                        if to_add not in all_possible_products:
                                                                            all_possible_products.append(to_add)
                                                                            all_possible_product_energies.append(energy)
        dumpfn(all_possible_products, 'all_possible_products.json')
        dumpfn(all_possible_product_energies, 'all_possible_product_energies.json')
        return all_possible_products, all_possible_product_energies

    def find_n_lowest_product_composition(self,all_possible_products, all_possible_product_energies, n=10):
        '''

        :param all_possible_products: List[list[str(mol_index)+'_'+str(charge)]]
        :param all_possible_product_energies: List(float), corresponding free energies
        :param n:
        :return:
        all_possible_product_lowest_n_new:
        A list of dictionary of possible products with stoichiometry. e.g. [{'0_0': 1, '32_0': 1}, {'0_1': 1, '32_-1': 1}]
        all_possible_product_energies_lowest_n (List(float)):
        Corresponding free energies.

        '''
        all_possible_product_lowest_n = []
        all_possible_product_energies_lowest_n = []
        indices_lowest_n = np.argsort(all_possible_product_energies)[:n]
        for index in indices_lowest_n:
            all_possible_product_lowest_n.append(all_possible_products[index])
            all_possible_product_energies_lowest_n.append(all_possible_product_energies[index])

        # Transform products from list format to stoichiometry dict format.
        all_possible_product_lowest_n_new = []
        for i in range(len(all_possible_product_lowest_n)):
            products = all_possible_product_lowest_n[i]
            print('products:',products, flush=True)
            products_dict = {}
            for item in products:
                if item in products_dict.keys():
                    products_dict[item] += 1
                else:
                    products_dict[item] = 1
            all_possible_product_lowest_n_new.append(products_dict)
        dumpfn(all_possible_product_lowest_n_new, 'all_possible_product_lowest_n_new.json')
        dumpfn(all_possible_product_energies_lowest_n, 'all_possible_product_energies_lowest_n.json')
        return all_possible_product_lowest_n_new, all_possible_product_energies_lowest_n

    def _trace_one_level_upwards(self, possible_products, starting_mols, allowed_num_mols=5, energy_thresh=0.0):
        '''
        Return all possible pathways that satisfy the energy requirement by tracing the parents,
        children of each product one level upwards, in the range of species existing in database.
        TODO: Not sure if I can lift the energy requirement too much. Possiblities might blow up.
        :param possible_products: A list of dictionary of possible products with stoichiometry.
                e.g. [{'0_0': 1, '32_0': 1}, {'0_1': 1, '32_-1': 1}]
        :param starting_mols: a list of starting mols. i.e. ['0_1','1_0']
        :param allowed_num_species: allowed number of molecules on each node, excluding electrons. Default to 5.
        :return: all possible pathways one level upwards (A list of dictionary of possible species with stoichiometry)
        '''
        charge_options = [-1, 0, 1]
        parents_list = []
        for n, item in enumerate(possible_products):
            species_list = self.split_species_dict(item)
            num_species = len(species_list)
            parents = [[] for i in range(int(num_species))]
            for i, mol in enumerate(species_list):
                #print('mol:',mol)
                if mol == 'e_-1':
                    parents[i].append({mol: 1})
                    continue
                else:
                    mol_index, mol_charge = int(mol.split('_')[0]),int(mol.split('_')[1])
                    # Include itself
                    parents[i].append({mol: 1})
                    if (mol_charge + 1) in self.opt_entries[mol_index]:
                        new_mol = str(mol_index) + '_' + str(mol_charge + 1)
                        parents[i].append({new_mol: 1, 'e_-1': 1})
                    if (mol_charge - 1) in self.opt_entries[mol_index]:
                        new_mol = str(mol_index) + '_' + str(mol_charge - 1)
                        parents[i].append({new_mol: 1, 'e_-1': -1})

                    for key in self.fragmentation_dict_new:
                        if key == mol_index:
                            for nodes in self.fragmentation_dict_new[key]:
                                if len(nodes) == 1:
                                    parent_mol_index = int(nodes[0])
                                    if mol_charge in self.opt_entries[parent_mol_index]:
                                        parent_mol = str(parent_mol_index) + '_' + str(mol_charge)
                                        parents[i].append({parent_mol: 1})
                                elif len(nodes) == 2:
                                    parent_mol1_index = int(nodes[0])
                                    parent_mol2_index = int(nodes[1])
                                    for parent_mol1_charge in charge_options:
                                        parent_mol2_charge = mol_charge - parent_mol1_charge
                                        if parent_mol1_charge in self.opt_entries[parent_mol1_index]:
                                            if parent_mol2_charge in self.opt_entries[parent_mol2_index]:
                                                parent_mol1 = str(parent_mol1_index) + '_' + str(parent_mol1_charge)
                                                parent_mol2 = str(parent_mol2_index) + '_' + str(parent_mol2_charge)
                                                if parent_mol1 == parent_mol2:
                                                    parents[i].append({parent_mol1: 2})
                                                else:
                                                    parents[i].append({parent_mol1: 1, parent_mol2: 1})
                                            if (parent_mol2_charge+1) in self.opt_entries[parent_mol2_index]:
                                                parent_mol1 = str(parent_mol1_index) + '_' + str(parent_mol1_charge)
                                                parent_mol2 = str(parent_mol2_index) + '_' + str(parent_mol2_charge+1)
                                                if parent_mol1 == parent_mol2:
                                                    parents[i].append({parent_mol1: 2, 'e_-1': 1})
                                                else:
                                                    parents[i].append({parent_mol1: 1, parent_mol2: 1, 'e_-1': 1})
                                            if (parent_mol2_charge-1) in self.opt_entries[parent_mol2_index]:
                                                parent_mol1 = str(parent_mol1_index) + '_' + str(parent_mol1_charge)
                                                parent_mol2 = str(parent_mol2_index) + '_' + str(parent_mol2_charge-1)
                                                if parent_mol1 == parent_mol2:
                                                    parents[i].append({parent_mol1: 2, 'e_-1': -1})
                                                else:
                                                    parents[i].append({parent_mol1: 1, parent_mol2: 1, 'e_-1': -1})

                        else:
                            for nodes in self.fragmentation_dict_new[key]:
                                if mol_index in nodes:
                                    parent_mol1_index = int(key)
                                    if len(nodes) == 1:
                                        if mol_charge in self.opt_entries[parent_mol1_index]:
                                            parent_mol1 = str(parent_mol1_index) + '_' + str(mol_charge)
                                            parents[i].append({parent_mol1: 1})
                                    elif len(nodes) == 2:
                                        if nodes[0] == mol_index:
                                            parent_mol2_index = int(nodes[1])
                                        elif nodes[1] == mol_index:
                                            parent_mol2_index = int(nodes[0])
                                        for parent_mol1_charge in charge_options:
                                            parent_mol2_charge = parent_mol1_charge - mol_charge
                                            if parent_mol1_charge in self.opt_entries[parent_mol1_index]:
                                                if parent_mol2_charge in self.opt_entries[parent_mol2_index]:
                                                    parent_mol1 = str(parent_mol1_index) + '_' + str(parent_mol1_charge)
                                                    parent_mol2 = str(parent_mol2_index) + '_' + str(parent_mol2_charge)
                                                    assert parent_mol1 != parent_mol2
                                                    parents[i].append({parent_mol1: 1, parent_mol2: -1})
                                                if (parent_mol2_charge+1) in self.opt_entries[parent_mol2_index]:
                                                    parent_mol1 = str(parent_mol1_index) + '_' + str(parent_mol1_charge)
                                                    parent_mol2 = str(parent_mol2_index) + '_' + str(parent_mol2_charge+1)
                                                    assert parent_mol1 != parent_mol2
                                                    parents[i].append({parent_mol1: 1, parent_mol2: -1, 'e_-1': -1})
                                                if (parent_mol2_charge-1) in self.opt_entries[parent_mol2_index]:
                                                    parent_mol1 = str(parent_mol1_index) + '_' + str(parent_mol1_charge)
                                                    parent_mol2 = str(parent_mol2_index) + '_' + str(parent_mol2_charge-1)
                                                    assert parent_mol1 != parent_mol2
                                                    parents[i].append({parent_mol1: 1, parent_mol2: -1, 'e_-1': 1})

                    for key in self.recomb_dict_no_opt:
                        inds = key.split('_')
                        mol_ind1, mol_ind2, atom_ind1, atom_ind2 = int(inds[0]), int(inds[1]), int(inds[2]), int(inds[3])
                        if mol_index == self.recomb_dict_no_opt[key]:
                            parent_mol1_index = mol_ind1
                            parent_mol2_index = mol_ind2
                            for parent_mol1_charge in charge_options:
                                parent_mol2_charge = mol_charge - parent_mol1_charge
                                if parent_mol1_charge in self.opt_entries[parent_mol1_index]:
                                    if parent_mol2_charge in self.opt_entries[parent_mol2_index]:
                                        parent_mol1 = str(parent_mol1_index) + '_' + str(parent_mol1_charge)
                                        parent_mol2 = str(parent_mol2_index) + '_' + str(parent_mol2_charge)
                                        if parent_mol1 == parent_mol2:
                                            parents[i].append({parent_mol1: 2})
                                        else:
                                            parents[i].append({parent_mol1: 1, parent_mol2: 1})
                                    if (parent_mol2_charge+1) in self.opt_entries[parent_mol2_index]:
                                        parent_mol1 = str(parent_mol1_index) + '_' + str(parent_mol1_charge)
                                        parent_mol2 = str(parent_mol2_index) + '_' + str(parent_mol2_charge+1)
                                        if parent_mol1 == parent_mol2:
                                            parents[i].append({parent_mol1: 2, 'e_-1': 1})
                                        else:
                                            parents[i].append({parent_mol1: 1, parent_mol2: 1, 'e_-1': 1})
                                    if (parent_mol2_charge-1) in self.opt_entries[parent_mol2_index]:
                                        parent_mol1 = str(parent_mol1_index) + '_' + str(parent_mol1_charge)
                                        parent_mol2 = str(parent_mol2_index) + '_' + str(parent_mol2_charge-1)
                                        if parent_mol1 == parent_mol2:
                                            parents[i].append({parent_mol1: 2, 'e_-1': -1})
                                        else:
                                            parents[i].append({parent_mol1: 1, parent_mol2: 1, 'e_-1': -1})
                        if mol_index == mol_ind1 or mol_index == mol_ind2:
                            parent_mol1_index = self.recomb_dict_no_opt[key]
                            if mol_index == mol_ind1:
                                parent_mol2_index = mol_ind2
                            elif mol_index == mol_ind2:
                                parent_mol2_index = mol_ind1
                            for parent_mol1_charge in charge_options:
                                parent_mol2_charge = parent_mol1_charge - mol_charge
                                if parent_mol1_charge in self.opt_entries[parent_mol1_index]:
                                    if parent_mol2_charge in self.opt_entries[parent_mol2_index]:
                                        parent_mol1 = str(parent_mol1_index) + '_' + str(parent_mol1_charge)
                                        parent_mol2 = str(parent_mol2_index) + '_' + str(parent_mol2_charge)
                                        assert parent_mol1 != parent_mol2
                                        parents[i].append({parent_mol1: 1, parent_mol2: -1})
                                    if (parent_mol2_charge+1) in self.opt_entries[parent_mol2_index]:
                                        parent_mol1 = str(parent_mol1_index) + '_' + str(parent_mol1_charge)
                                        parent_mol2 = str(parent_mol2_index) + '_' + str(parent_mol2_charge+1)
                                        assert parent_mol1 != parent_mol2
                                        parents[i].append({parent_mol1: 1, parent_mol2: -1, 'e_-1': -1})
                                    if (parent_mol2_charge-1) in self.opt_entries[parent_mol2_index]:
                                        parent_mol1 = str(parent_mol1_index) + '_' + str(parent_mol1_charge)
                                        parent_mol2 = str(parent_mol2_index) + '_' + str(parent_mol2_charge-1)
                                        assert parent_mol1 != parent_mol2
                                        parents[i].append({parent_mol1: 1, parent_mol2: -1, 'e_-1': 1})
            parents_list.append(parents)

        # merge species dict
        energy_standard = np.sum([self.opt_entries[int(key.split('_')[0])][int(key.split('_')[1])].free_energy for key in starting_mols if key!='e_-1'])
        if 'e_-1' in starting_mols:
            energy_standard += self.electron_free_energy * starting_mols.count('e_-1')
        possible_pathways_before_final = []
        for i in range(len(parents_list)):
            energy_reference = np.sum(
                [self.opt_entries[int(key.split('_')[0])][int(key.split('_')[1])].free_energy * possible_products[i][key] for key in possible_products[i] if key!='e_-1'])
            if 'e_-1' in possible_products[i]:
                energy_reference += self.electron_free_energy * possible_products[i]['e_-1']
            possible_pathways_dummy = []
            parents = parents_list[i]  # [[{1},{2}],  [{3},{4}],  [{5},{6},{7}],  [{8},{9},{10},{11}]]
            num_parents = len(parents)
            num_possible_parents_list = []
            for j in range(len(parents)):
                num_possible_parents_list.append(len(parents[j]))
            combinations = product(
                *[range(num_possible_parents_list[k]) for k in range(len(num_possible_parents_list))])
            for combo in combinations:
                species_dicts_to_merge = []
                for m in range(len(combo)):
                    dict_to_merge = parents[m][combo[m]]
                    species_dicts_to_merge.append(dict_to_merge)
                merge_species_dict = self.merge_multiple_species_dict(species_dicts_to_merge)
                # eliminate the case with negative stoichiometry
                if all(merge_species_dict[key] > 0 for key in merge_species_dict.keys()):
                    energy = np.sum(
                        [self.opt_entries[int(key.split('_')[0])][int(key.split('_')[1])].free_energy * merge_species_dict[key] for key in merge_species_dict.keys() if key!='e_-1'])
                    if 'e_-1' in merge_species_dict:
                        energy += self.electron_free_energy * merge_species_dict['e_-1']
                    if energy + energy_thresh >= energy_reference and energy <= energy_standard + energy_thresh:
                        possible_pathways_dummy.append(merge_species_dict)
            possible_pathways_before_final.append(possible_pathways_dummy)

        possible_pathways_dummy = []
        for i in range(len(possible_pathways_before_final)):
            pathways = possible_pathways_before_final[i]
            possible_pathways_dummy += pathways
        possible_pathways_final = []
        for i in range(len(possible_pathways_dummy)):
            if (possible_pathways_dummy[i] not in possible_pathways_final) and (
                    possible_pathways_dummy[i] not in possible_products):
                # the case with negative stoichiometry already eliminated
                possible_pathways_final.append(possible_pathways_dummy[i])
        new_possible_pathways_before_final = []
        new_possible_pathways_final = []

        for i in range(len(possible_pathways_before_final)):
            pathway_list = possible_pathways_before_final[i]
            new_pathway_list = []
            for item in pathway_list:
                num_species = 0
                for key in item.keys():
                    if key != 'e_-1':
                        num_species += item[key]
                if num_species <= allowed_num_mols:
                    new_pathway_list.append(item)
            new_possible_pathways_before_final.append(new_pathway_list)

        for item in possible_pathways_final:
            num_species = 0
            for key in item.keys():
                if key != 'e_-1':
                    num_species += item[key]
            if num_species <= allowed_num_mols:
                new_possible_pathways_final.append(item)

        return new_possible_pathways_before_final, new_possible_pathways_final

    def add_obvious_edges(self, nodes, edges):
        '''
        Add obvious possible transformations, i.e. electron transfer reactions to existing edges.
        :return: a list of new edges
        '''
        edges_new = []
        for i, node0 in enumerate(nodes):
            for j, node1 in enumerate(nodes):
                if i == j:
                    continue
                else:
                    species_in_node0 = list(set([key.split('_')[0] for key in node0.keys()]))
                    species_in_node1 = list(set([key.split('_')[0] for key in node1.keys()]))
                    if (list(np.sort(species_in_node0)) == list(np.sort(species_in_node1))) or \
                            (list(np.sort(species_in_node0)) == list(np.sort(species_in_node1 + ['e']))) or \
                            (list(np.sort(species_in_node1)) == list(np.sort(species_in_node0 + ['e']))):
                        edges_new.append((node0, node1))
                        edges_new.append((node1, node0))
        for edge in edges_new:
            if edge not in edges:
                edges.append(edge)

        return edges

    def eliminate_edges_by_energy(self,pathway_nodes, pathway_edges, energy_thresh=0.0):
        '''
        Eliminate edges by free energy change of the reaction.
        :param pathway_nodes: A list of mols (str(molecule index)+'_'+str(molecule charge)). e.g. ['0_1','1_0']
        :param pathway_edges:
        :param energy_thresh (float): free energy change of the reaction (product energy - reactant energy).
               Default to 0.0, so all endergonic reations are removed. This should always be usually positive.
        :return:
        '''
        node_energies = []
        for node in pathway_nodes:
            assert all(key in self.opt_species_w_charge for key in node if key != 'e_-1')
            energy = np.sum([self.opt_entries[int(key.split('_')[0])][int(key.split('_')[1])].free_energy * node[key] for key in node if key != 'e_-1'])
            if 'e_-1' in node:
                energy += self.electron_free_energy * node['e_-1']
            node_energies.append(energy)

        valid_edges = []
        for i, edge in enumerate(pathway_edges):
            if node_energies[pathway_nodes.index(edge[0])] + energy_thresh >= node_energies[pathway_nodes.index(edge[1])]:
                valid_edges.append(edge)
        return valid_edges, node_energies

    def look_for_parent_nodes(self,pathway_nodes, pathway_edges, index):
        '''
        Look for parent nodes(include itself) for a certain node with index by looping through the edges for one round. Need to execute
        multiple times to find all the parent nodes. This is for transformed nodes.

        :param pathway_nodes: List[int]
        :param pathway_edges: List[tuple(int)]
        :return: parent_nodes:  List[int]
        '''
        pathway_nodes = list(pathway_nodes)
        node = pathway_nodes[index]
        parent_nodes = [node]
        for edge in pathway_edges:
            if edge[1] == node:
                parent_nodes.append(edge[0])
        parent_nodes = list(set(parent_nodes))
        return parent_nodes

    def eliminate_nodes_from_nowhere(self,pathway_nodes, pathway_edges, starting_nodes, node_energies):
        '''
        Eliminates the nodes that comes from nowhere (not connected to starting nodes). This is for transformed nodes.
        :param pathway_nodes: List[int]
        :param pathway_edges: List[tuple(int)]
        :param starting_nodes: List[int]
        :param node_energies: free energies corresponding to pathway_nodes
        :return:
        '''
        pathway_nodes = list(pathway_nodes)
        pathway_nodes_new = []
        pathway_edges_new = []
        energies_new = []
        starting_nodes_index = [pathway_nodes.index(starting_nodes[i]) for i in range(len(starting_nodes))]
        for i, node in enumerate(pathway_nodes):
            if node not in starting_nodes:
                parent_nodes = self.look_for_parent_nodes(pathway_nodes, pathway_edges, i)
                if len(parent_nodes) == 0:
                    continue
                else:
                    new_parent_nodes = copy.deepcopy(parent_nodes)
                    for j, parent_node in enumerate(parent_nodes):
                        new_parent_nodes += self.look_for_parent_nodes(pathway_nodes, pathway_edges,
                                                                  pathway_nodes.index(parent_node))
                    new_parent_nodes = list(set(new_parent_nodes))
                    while len(new_parent_nodes) != len(parent_nodes):
                        parent_nodes = copy.deepcopy(new_parent_nodes)
                        for k, m in enumerate(parent_nodes):
                            new_parent_nodes += self.look_for_parent_nodes(pathway_nodes, pathway_edges,
                                                                      pathway_nodes.index(m))
                        new_parent_nodes = list(set(new_parent_nodes))
                if any(starting_nodes[i] in new_parent_nodes for i in range(len(starting_nodes))):
                    pathway_nodes_new.append(node)
                    energies_new.append(node_energies[i])

        pathway_nodes_new += starting_nodes
        energies_new += [node_energies[starting_nodes_index[i]] for i in range(len(starting_nodes_index))]

        for j, edge in enumerate(pathway_edges):
            if edge[0] in pathway_nodes_new and edge[1] in pathway_nodes_new:
                pathway_edges_new.append(edge)
        return pathway_nodes_new, pathway_edges_new, energies_new

    def find_starting_mols_and_crude_energy_thresh(self, mol_graphs, charges, num_electrons=0):
        '''
        Given a list of mol graphs and their total charges, find the mol indices in self.total_mol_graphs_no_opt.
        :param mol_graphs:
        :param charges:
        :return: a list of starting mols. List(str(molecule index)+'_'+str(molecule charge)). e.g. ['0_1','1_0']
        '''
        starting_mols = []
        crude_energy_thresh = 0.0
        for i, mol_graph in enumerate(self.total_mol_graphs_no_opt):
            for j, mol_graph_to_find in enumerate(mol_graphs):
                if mol_graph.molecule.composition.alphabetical_formula == mol_graph_to_find.molecule.composition.alphabetical_formula:
                    if mol_graph.isomorphic_to(mol_graph_to_find):
                        starting_mols.append(str(i)+'_'+str(charges[j]))
        for mol in starting_mols:
            crude_energy_thresh += self.opt_entries[int(mol.split('_')[0])][int(mol.split('_')[1])].free_energy
        if num_electrons != 0:
            for i in range(num_electrons):
                starting_mols.append('e_-1')
                crude_energy_thresh += self.electron_free_energy
        dumpfn(starting_mols, 'starting_mols.json')
        dumpfn(crude_energy_thresh, 'crude_energy_thresh.json')
        return starting_mols, crude_energy_thresh


    def map_all_possible_pathways(self,possible_products, starting_mols, allowed_num_mols=5, energy_thresh=0.0, max_iter=20):
        '''

        :param possible_products: A list of dictionary of possible products with stoichiometry.
                i.e. [{'0_0': 1, '32_0': 1}, {'0_1': 1, '32_-1': 1}]
        :param starting_mols: a list of starting mols. e.g.['0_1','1_0']
        :return:
        '''
        pathway_nodes = copy.deepcopy(possible_products)
        possible_products_new = copy.deepcopy(possible_products)
        pathway_edges = []
        possible_pathways_upwards_final = [{}]
        iter = 0
        while possible_pathways_upwards_final != [] and iter<max_iter:
            iter += 1
            print('iteration:', iter, flush=True)
            possible_pathways_upwards, possible_pathways_upwards_final = \
                self._trace_one_level_upwards(possible_products_new,starting_mols,allowed_num_mols)
            print('length of possible_pathways_upwards:',len(possible_pathways_upwards), flush=True)
            print('length of possible_pathways_upwards final:',len(possible_pathways_upwards_final), flush=True)
            pathway_nodes += possible_pathways_upwards_final
            for i in range(len(possible_pathways_upwards)):
                for j in range(len(possible_pathways_upwards[i])):
                    pathway_edges.append((possible_pathways_upwards[i][j],possible_products_new[i]))

            possible_products_new = possible_pathways_upwards_final
        print('mapping done!',flush=True)
        # take the set of all nodes
        pathway_nodes_final = []
        for i,node in enumerate(pathway_nodes):
            if node not in pathway_nodes_final:
                pathway_nodes_final.append(node)

        # take the set of all edges
        pathway_edges_final = []
        for i,edge in enumerate(pathway_edges):
            if edge not in pathway_edges_final and edge[0] != edge[1]:
                pathway_edges_final.append(edge)

        pathway_edges_final = self.add_obvious_edges(pathway_nodes_final, pathway_edges_final)
        pathway_edges_final, node_energies = self.eliminate_edges_by_energy(pathway_nodes_final, pathway_edges_final, energy_thresh)

        dumpfn(pathway_nodes_final, 'pathway_nodes_final.json')
        dumpfn(pathway_edges_final, 'pathway_edges_final.json')
        dumpfn(node_energies, 'node_energies.json')
        return pathway_nodes_final, pathway_edges_final, node_energies

    def transform_nodes_and_edges(self, pathway_nodes, pathway_edges, starting_mols_list, node_energies, starting_num_electrons_list):
        '''

        :param pathway_nodes: List[species_dict], species_dict = {'0_1': 2, '1_0': 2, 'e_-1': 2}
        :param pathway_edges:
        :param starting_mols_list: a list of starting mols. e.g.[['0_1','1_0']].
        :param starting_num_electrons_list: a list of number of electrons(int) cooresponding to starting_mols_list.
            Usually this should just be a list of one element(starting_mols).
            However, if there are specific nodes(composition) treat as the starting composition of the graph, we can change this.

        :return:
        '''
        # Transform the starting_mols to starting_nodes (stoi dict).
        starting_nodes = []
        for i in range(len(starting_mols_list)):
            mols = starting_mols_list[i]
            num_electrons = starting_num_electrons_list[i]
            mols_dict = {}
            for item in mols:
                if item in mols_dict.keys():
                    mols_dict[item] += 1
                else:
                    mols_dict[item] = 1
            if num_electrons != 0:
                for j in range(num_electrons):
                    if j == 0:
                        mols_dict['e_-1'] = 1
                    else:
                        mols_dict['e_-1'] += 1
            starting_nodes.append(mols_dict)

        self.number_to_nodes_dict = {}
        transformed_nodes = np.arange(len(pathway_nodes))
        transformed_edges = []
        for i, node in enumerate(pathway_nodes):
            self.number_to_nodes_dict[i] = node
        for j, edge in enumerate(pathway_edges):
            new_edge = (pathway_nodes.index(edge[0]), pathway_nodes.index(edge[1]))
            transformed_edges.append(new_edge)

        indices = []
        for i, key in enumerate(self.number_to_nodes_dict.keys()):
            if self.number_to_nodes_dict[key] in starting_nodes:
                indices.append(key)
        print("Starting node indices:", indices, flush=True)

        new_pathway_nodes, new_pathway_edges, new_energies = \
            self.eliminate_nodes_from_nowhere(transformed_nodes, transformed_edges, indices, node_energies)

        dumpfn(self.number_to_nodes_dict, 'number_to_nodes_dict.json')
        dumpfn(new_pathway_nodes, 'new_pathway_nodes.json')
        dumpfn(new_pathway_edges, 'new_pathway_edges.json')
        dumpfn(new_energies, 'new_energies.json')
        return new_pathway_nodes, new_pathway_edges, new_energies

    def get_color_from_list(self, num, list):
        new_colors = []
        interval = int(len(list) / num)
        begin_index = interval // 2 + 2
        for i in range(num):
            color = list[begin_index + interval * i]
            new_colors.append(color)
        new_colors.reverse()

        return new_colors

    def visualize_reaction_network(self, pathway_nodes, pathway_edges, node_energies, file_name='reaction_network'):
        sorted_energy = np.sort(node_energies)
        # for each node, a number of energy ranking
        nodes_energy_ranking = [list(sorted_energy).index(node_energies[i]) for i in range(len(pathway_nodes))]
        colors = np.concatenate([[i for j in range(40)] for i in range(1, 9)])
        new_colors = self.get_color_from_list(len(pathway_nodes), colors)  # deep to light color

        u = Digraph(file_name, filename=file_name)
        u.attr(size='6,6')
        for i, node in enumerate(pathway_nodes):
            u.attr('node', style='filled', colorscheme='purples9', fillcolor=str(new_colors[nodes_energy_ranking[i]]),
                   shape='circle', width='5')
            u.node(str(node))

        _edge = '\t%s -> %s%s'
        # line = _edge % ('2', '1', '')
        for (i, j) in pathway_edges:
            u.edge(str(i), str(j))
        u.render('test_output/' + file_name, view=False)

        return

    def generate_entries(self, pathway_nodes, entries_file_name='valid'):
        '''
        This is for transformed nodes.
        :param pathway_nodes: List[int]
        :return:
        '''
        unique_species = []
        self.good_entries = []
        for i in pathway_nodes:
            composition = self.number_to_nodes_dict[i]
            for key in composition.keys():
                if key not in unique_species:
                    unique_species.append(key)
                    if key != 'e_-1':
                        mol_entry = self.opt_entries[int(key.split('_')[0])][int(key.split('_')[1])]
                        self.good_entries.append(mol_entry)
                else:
                    continue
        dumpfn(self.good_entries, "smd_production_entries_"+entries_file_name+".json")
        return

    def whole_workflow(self,target_composition, target_charge, starting_mol_graphs, starting_charges, starting_num_electrons,
                       allowed_num_mols=5, energy_thresh=0.0, load_entries_name=['smd_target_entries'],graph_file_name='reaction_network',
                       entries_file_name='valid'):
        '''
        Have to run self.query_database beforehand and save the entries.
        :param target_composition:
        :param target_charge:
        :param crude_energy_thresh:
        :return:
        '''
        print('working on fragmentation and recombination!',flush=True)
        self.recombination()
        print('recombination done!',flush=True)
        print('working on creating stoichiometry table!',flush=True)
        self.generate_stoichiometry_table()
        print('creating stoichiometry table done!',flush=True)
        print('working on getting optimized structures!',flush=True)
        self.get_optimized_structures(load_entries=True,entries_name=load_entries_name)
        print('getting optimized structures done!',flush=True)
        starting_mols, crude_energy_thresh = self.find_starting_mols_and_crude_energy_thresh(starting_mol_graphs, starting_charges, starting_num_electrons)
        starting_mols_list = [starting_mols]
        all_possible_products, all_possible_product_energies = \
            self.find_all_product_composition_from_target(target_composition, target_charge, crude_energy_thresh)
        all_possible_product_lowest_n, all_possible_product_energies_lowest_n = \
            self.find_n_lowest_product_composition(all_possible_products, all_possible_product_energies)
        pathway_nodes_final, pathway_edges_final, node_energies = \
            self.map_all_possible_pathways(all_possible_product_lowest_n, starting_mols, allowed_num_mols, energy_thresh)
        new_pathway_nodes, new_pathway_edges, new_energies = self.transform_nodes_and_edges(pathway_nodes_final, pathway_edges_final, starting_mols_list, node_energies)
        self.visualize_reaction_network(new_pathway_nodes, new_pathway_edges, new_energies, graph_file_name)
        self.generate_entries(new_pathway_nodes,entries_file_name)

        return

    def whole_workflow_from_mol_entries(self,target_composition, target_charge, starting_mol_graphs, starting_charges, starting_num_electrons,
                       allowed_num_mols=5, energy_thresh=0.0, load_entries_name=['smd_target_entries'],graph_file_name='reaction_network',
                       entries_file_name='valid'):
        '''
        Have to run self.query_database beforehand and save the entries.
        :param target_composition:
        :param target_charge:
        :param crude_energy_thresh:
        :return:
        '''
        print('working on fragmentation and recombination!',flush=True)
        self.recombination()
        print('recombination done!',flush=True)
        print('working on creating stoichiometry table!',flush=True)
        self.generate_stoichiometry_table()
        print('creating stoichiometry table done!',flush=True)
        print('working on getting optimized structures!',flush=True)
        self.get_optimized_structures_new(load_entries=True,entries_name=load_entries_name)
        print('getting optimized structures done!',flush=True)
        starting_mols, crude_energy_thresh = self.find_starting_mols_and_crude_energy_thresh(starting_mol_graphs, starting_charges, starting_num_electrons)
        starting_mols_list = [starting_mols]
        all_possible_products, all_possible_product_energies = \
            self.find_all_product_composition_from_target(target_composition, target_charge, crude_energy_thresh)
        all_possible_product_lowest_n, all_possible_product_energies_lowest_n = \
            self.find_n_lowest_product_composition(all_possible_products, all_possible_product_energies)
        pathway_nodes_final, pathway_edges_final, node_energies = \
            self.map_all_possible_pathways(all_possible_product_lowest_n, starting_mols, allowed_num_mols, energy_thresh)
        new_pathway_nodes, new_pathway_edges, new_energies = self.transform_nodes_and_edges(pathway_nodes_final, pathway_edges_final, starting_mols_list, node_energies)
        self.visualize_reaction_network(new_pathway_nodes, new_pathway_edges, new_energies, graph_file_name)
        self.generate_entries(new_pathway_nodes,entries_file_name)

        return

    def whole_workflow_load_file(self,target_composition, target_charge, starting_mol_graphs, starting_charges, starting_num_electrons,
                       allowed_num_mols=5, energy_thresh=0.0, load_entries_name=['smd_target_entries'], graph_file_name='reaction_network',
                       entries_file_name='valid',path=''):
        '''
        Have to run self.query_database beforehand and save the entries.
        :param target_composition:
        :param target_charge:
        :param crude_energy_thresh:
        :return:
        '''

        self.fragmentation_dict_new = loadfn(path+'fragmentation_dict_new.json')
        self.recomb_dict_no_opt = loadfn(path+'recomb_dict_no_opt.json')
        self.opt_entries = loadfn(path+'opt_entries.json')
        self.opt_species_w_charge = loadfn(path+'opt_species_w_charge.json')
        self.total_mol_graphs_no_opt = loadfn(path+'total_mol_graphs_no_opt.json')

        # clean up some dicts b/c loadfn will make the keys into strings
        self.fragmentation_dict_new_2 = {}
        for key in self.fragmentation_dict_new:
            self.fragmentation_dict_new_2[int(key)] = self.fragmentation_dict_new[key]
        self.fragmentation_dict_new = copy.deepcopy(self.fragmentation_dict_new_2)
        del self.fragmentation_dict_new_2

        self.opt_entries_new = {}
        for key in self.opt_entries:
            self.opt_entries_new[int(key)] = {}
            for key1 in self.opt_entries[key]:
                self.opt_entries_new[int(key)][int(key1)] = self.opt_entries[key][key1]
        self.opt_entries = copy.deepcopy(self.opt_entries_new)
        del self.opt_entries_new

        print('working on creating stoichiometry table!', flush=True)
        self.generate_stoichiometry_table()
        print('creating stoichiometry table done!', flush=True)

        starting_mols, crude_energy_thresh = self.find_starting_mols_and_crude_energy_thresh(starting_mol_graphs, starting_charges, starting_num_electrons)
        starting_mols_list = [starting_mols]
        starting_num_electrons_list = [starting_num_electrons]
        all_possible_products, all_possible_product_energies = \
            self.find_all_product_composition_from_target(target_composition, target_charge, crude_energy_thresh)
        #all_possible_products = loadfn('all_possible_products.json')
        #all_possible_product_energies = loadfn('all_possible_product_energies.json')
        # all_possible_products, all_possible_product_energies = \
        #     self.find_all_product_composition_from_target(target_composition, target_charge, crude_energy_thresh)
        all_possible_product_lowest_n, all_possible_product_energies_lowest_n = \
            self.find_n_lowest_product_composition(all_possible_products, all_possible_product_energies)
        pathway_nodes_final, pathway_edges_final, node_energies = \
            self.map_all_possible_pathways(all_possible_product_lowest_n, starting_mols, allowed_num_mols, energy_thresh)
        self.new_pathway_nodes, self.new_pathway_edges, self.new_energies = self.transform_nodes_and_edges(pathway_nodes_final, pathway_edges_final, starting_mols_list, node_energies,starting_num_electrons_list)
        self.generate_entries(self.new_pathway_nodes, entries_file_name)
        self.visualize_reaction_network(self.new_pathway_nodes, self.new_pathway_edges, self.new_energies, graph_file_name)


        return

    def whole_workflow_load_file_2(self,target_composition, target_charge, starting_mol_graphs, starting_charges, starting_num_electrons,
                       allowed_num_mols=5, energy_thresh=0.0, load_entries_name=['smd_target_entries'], graph_file_name='reaction_network',
                       entries_file_name='valid',path=''):
        '''
        Have to run self.query_database beforehand and save the entries.
        :param target_composition:
        :param target_charge:
        :param crude_energy_thresh:
        :return:
        '''

        self.fragmentation_dict_new = loadfn(path+'fragmentation_dict_new.json')
        self.recomb_dict_no_opt = loadfn(path+'recomb_dict_no_opt.json')
        self.total_mol_graphs_no_opt = loadfn(path+'total_mol_graphs_no_opt.json')

        # clean up some dicts b/c loadfn will make the keys into strings
        self.fragmentation_dict_new_2 = {}
        for key in self.fragmentation_dict_new:
            self.fragmentation_dict_new_2[int(key)] = self.fragmentation_dict_new[key]
        self.fragmentation_dict_new = copy.deepcopy(self.fragmentation_dict_new_2)
        del self.fragmentation_dict_new_2


        print('working on creating stoichiometry table!', flush=True)
        self.generate_stoichiometry_table()
        print('creating stoichiometry table done!', flush=True)

        print('working on getting optimized structures!', flush=True)
        self.get_optimized_structures_new(load_entries=True, entries_name=load_entries_name)
        print('getting optimized structures done!', flush=True)

        starting_mols, crude_energy_thresh = self.find_starting_mols_and_crude_energy_thresh(starting_mol_graphs, starting_charges, starting_num_electrons)
        starting_mols_list = [starting_mols]
        starting_num_electrons_list = [starting_num_electrons]
        all_possible_products, all_possible_product_energies = \
            self.find_all_product_composition_from_target(target_composition, target_charge, crude_energy_thresh)
        #all_possible_products = loadfn('all_possible_products.json')
        #all_possible_product_energies = loadfn('all_possible_product_energies.json')
        # all_possible_products, all_possible_product_energies = \
        #     self.find_all_product_composition_from_target(target_composition, target_charge, crude_energy_thresh)
        all_possible_product_lowest_n, all_possible_product_energies_lowest_n = \
            self.find_n_lowest_product_composition(all_possible_products, all_possible_product_energies)
        pathway_nodes_final, pathway_edges_final, node_energies = \
            self.map_all_possible_pathways(all_possible_product_lowest_n, starting_mols, allowed_num_mols, energy_thresh)
        self.new_pathway_nodes, self.new_pathway_edges, self.new_energies = self.transform_nodes_and_edges(pathway_nodes_final, pathway_edges_final, starting_mols_list, node_energies,starting_num_electrons_list)
        self.generate_entries(self.new_pathway_nodes, entries_file_name)
        self.visualize_reaction_network(self.new_pathway_nodes, self.new_pathway_edges, self.new_energies, graph_file_name)

        return

    def whole_workflow_load_file_3(self,target_composition, target_charge, starting_mol_graphs, starting_charges, starting_num_electrons,
                       allowed_num_mols=5, energy_thresh=0.0, load_entries_name=['smd_target_entries'], graph_file_name='reaction_network',
                       entries_file_name='valid',path=''):
        '''
        Have to run self.query_database beforehand and save the entries.
        :param target_composition:
        :param target_charge:
        :param crude_energy_thresh:
        :return:
        '''

        self.fragmentation_dict_new = loadfn(path+'fragmentation_dict_new.json')
        self.recomb_dict_no_opt = loadfn(path+'recomb_dict_no_opt.json')
        self.total_mol_graphs_no_opt = loadfn(path+'total_mol_graphs_no_opt.json')

        # clean up some dicts b/c loadfn will make the keys into strings
        self.fragmentation_dict_new_2 = {}
        for key in self.fragmentation_dict_new:
            self.fragmentation_dict_new_2[int(key)] = self.fragmentation_dict_new[key]
        self.fragmentation_dict_new = copy.deepcopy(self.fragmentation_dict_new_2)
        del self.fragmentation_dict_new_2


        print('working on creating stoichiometry table!', flush=True)
        self.generate_stoichiometry_table()
        print('creating stoichiometry table done!', flush=True)

        print('working on getting optimized structures!', flush=True)
        self.get_optimized_structures_new(load_entries=True, entries_name=load_entries_name)
        print('getting optimized structures done!', flush=True)

        starting_mols, crude_energy_thresh = self.find_starting_mols_and_crude_energy_thresh(starting_mol_graphs, starting_charges, starting_num_electrons)
        starting_mols_list = [starting_mols]
        starting_num_electrons_list = [starting_num_electrons]
        all_possible_products, all_possible_product_energies = \
            self.find_all_product_composition_from_target(target_composition, target_charge, crude_energy_thresh)
        #all_possible_products = loadfn('all_possible_products.json')
        #all_possible_product_energies = loadfn('all_possible_product_energies.json')
        # all_possible_products, all_possible_product_energies = \
        #     self.find_all_product_composition_from_target(target_composition, target_charge, crude_energy_thresh)
        all_possible_product_lowest_n, all_possible_product_energies_lowest_n = \
            self.find_n_lowest_product_composition(all_possible_products, all_possible_product_energies)
        pathway_nodes_final, pathway_edges_final, node_energies = \
            self.map_all_possible_pathways(all_possible_product_lowest_n, starting_mols, allowed_num_mols, energy_thresh)
        self.new_pathway_nodes, self.new_pathway_edges, self.new_energies = self.transform_nodes_and_edges(pathway_nodes_final, pathway_edges_final, starting_mols_list, node_energies,starting_num_electrons_list)
        self.generate_entries(self.new_pathway_nodes, entries_file_name)
        self.visualize_reaction_network(self.new_pathway_nodes, self.new_pathway_edges, self.new_energies, graph_file_name)

        return

    def whole_workflow_load_file_4(self,target_composition, target_charge, starting_mol_graphs, starting_charges, starting_num_electrons,
                       allowed_num_mols=5, energy_thresh=0.0, load_entries_name='smd_target_entries', graph_file_name='reaction_network',
                       entries_file_name='valid',path=''):
        '''
        Have to run self.query_database beforehand and save the entries.
        :param target_composition:
        :param target_charge:
        :param crude_energy_thresh:
        :return:
        '''

        self.fragmentation_dict_new = loadfn(path+'fragmentation_dict_new.json')
        self.recomb_dict_no_opt = loadfn(path+'recomb_dict_no_opt.json')
        self.opt_entries = loadfn(path+'opt_entries.json')
        self.opt_species_w_charge = loadfn(path+'opt_species_w_charge.json')
        self.total_mol_graphs_no_opt = loadfn(path+'total_mol_graphs_no_opt.json')

        # clean up some dicts b/c loadfn will make the keys into strings
        self.fragmentation_dict_new_2 = {}
        for key in self.fragmentation_dict_new:
            self.fragmentation_dict_new_2[int(key)] = self.fragmentation_dict_new[key]
        self.fragmentation_dict_new = copy.deepcopy(self.fragmentation_dict_new_2)
        del self.fragmentation_dict_new_2

        self.opt_entries_new = {}
        for key in self.opt_entries:
            self.opt_entries_new[int(key)] = {}
            for key1 in self.opt_entries[key]:
                self.opt_entries_new[int(key)][int(key1)] = self.opt_entries[key][key1]
        self.opt_entries = copy.deepcopy(self.opt_entries_new)
        del self.opt_entries_new

        print('working on creating stoichiometry table!', flush=True)
        self.generate_stoichiometry_table()
        print('creating stoichiometry table done!', flush=True)

        #starting_mols, crude_energy_thresh = self.find_starting_mols_and_crude_energy_thresh(starting_mol_graphs, starting_charges, starting_num_electrons)
        starting_mols = loadfn('starting_mols.json')
        crude_energy_thresh = loadfn('crude_energy_thresh.json')
        starting_mols_list = [starting_mols]
        starting_num_electrons_list = [starting_num_electrons]
        # all_possible_products, all_possible_product_energies = \
        #     self.find_all_product_composition_from_target(target_composition, target_charge, crude_energy_thresh)
        all_possible_products = loadfn('all_possible_products.json')
        all_possible_product_energies = loadfn('all_possible_product_energies.json')
        # all_possible_products, all_possible_product_energies = \
        #     self.find_all_product_composition_from_target(target_composition, target_charge, crude_energy_thresh)
        # all_possible_product_lowest_n, all_possible_product_energies_lowest_n = \
        #     self.find_n_lowest_product_composition(all_possible_products, all_possible_product_energies)
        all_possible_product_lowest_n = loadfn('all_possible_product_lowest_n_new.json')
        all_possible_product_energies_lowest_n = loadfn('all_possible_product_energies_lowest_n.json')
        pathway_nodes_final, pathway_edges_final, node_energies = \
            self.map_all_possible_pathways(all_possible_product_lowest_n, starting_mols, allowed_num_mols, energy_thresh)
        self.new_pathway_nodes, self.new_pathway_edges, self.new_energies = self.transform_nodes_and_edges(pathway_nodes_final, pathway_edges_final, starting_mols_list, node_energies,starting_num_electrons_list)
        self.generate_entries(self.new_pathway_nodes, entries_file_name)
        self.visualize_reaction_network(self.new_pathway_nodes, self.new_pathway_edges, self.new_energies, graph_file_name)

        return



if __name__ == "__main__":
    LiEC_neutral = Molecule.from_file('/Users/xiaoweixie/PycharmProjects/electrolyte/fragmentation/LiEC_neutral.xyz')
    LiEC_neutral_graph = MoleculeGraph.with_local_env_strategy(LiEC_neutral, OpenBabelNN(),
                                                               reorder=False,
                                                               extend_structure=False)
    LiEC_neutral_graph_extender = metal_edge_extender(LiEC_neutral_graph)

    water = Molecule.from_file('/Users/xiaoweixie/PycharmProjects/electrolyte/PF6/water.xyz')
    water_graph = MoleculeGraph.with_local_env_strategy(water, OpenBabelNN(),
                                                        reorder=False,
                                                        extend_structure=False)

    mol_graphs = [LiEC_neutral_graph_extender,water_graph]
    FCN = FixedCompositionNetwork(mol_graphs,[2,1],False)
    FCN.query_database(save=True)
    #unique_fragmentation_list, unique_fragments = FCN._fragment_one_level(LiEC_neutral_graph_extender)
    # unique_fragments_new, fragmentation_dict_new  = FCN.fragmentation()
    #
    # path = '/Users/xiaoweixie/pymatgen/pymatgen/analysis/reaction_network/fixed_composition/test_xyz/'
    # for i, frag in enumerate(unique_fragments_new):
    #     frag.molecule.to('xyz',path+str(i)+'.xyz')


    FCN.fragmentation_dict_new = loadfn('fragmentation_dict_new.json')
    FCN.recomb_dict_no_opt = loadfn('recomb_dict_no_opt.json')
    FCN.opt_entries = loadfn('opt_entries.json')
    FCN.opt_species_w_charge = loadfn('opt_species_w_charge.json')
    FCN.total_mol_graphs_no_opt = loadfn('total_mol_graphs_no_opt.json')

    # clean up some dicts b/c loadfn will make the keys into strings
    FCN.fragmentation_dict_new_2 = {}
    for key in FCN.fragmentation_dict_new:
        FCN.fragmentation_dict_new_2[int(key)] = FCN.fragmentation_dict_new[key]
    FCN.fragmentation_dict_new = copy.deepcopy(FCN.fragmentation_dict_new_2)
    del FCN.fragmentation_dict_new_2

    FCN.opt_entries_new = {}
    for key in FCN.opt_entries:
        FCN.opt_entries_new[int(key)] = {}
        for key1 in FCN.opt_entries[key]:
            FCN.opt_entries_new[int(key)][int(key1)] = FCN.opt_entries[key][key1]
    FCN.opt_entries = copy.deepcopy(FCN.opt_entries_new)
    del FCN.opt_entries_new

    print('working on creating stoichiometry table!', flush=True)
    FCN.generate_stoichiometry_table()
    print('creating stoichiometry table done!', flush=True)

    starting_mols, crude_energy_thresh = FCN.find_starting_mols_and_crude_energy_thresh(starting_mol_graphs,
                                                                                         starting_charges,
                                                                                         starting_num_electrons)
    allowed_num_mols=5
    energy_thresh=0.0
    starting_mols_list = [starting_mols]
    all_possible_products = loadfn('all_possible_products.json')
    all_possible_product_energies = loadfn('all_possible_product_energies.json')
    # all_possible_products, all_possible_product_energies = \
    #     self.find_all_product_composition_from_target(target_composition, target_charge, crude_energy_thresh)
    all_possible_product_lowest_n, all_possible_product_energies_lowest_n = \
        FCN.find_n_lowest_product_composition(all_possible_products, all_possible_product_energies)
    pathway_nodes_final, pathway_edges_final, node_energies = \
                FCN.map_all_possible_pathways(all_possible_product_lowest_n, starting_mols, allowed_num_mols, energy_thresh)

    FCN.new_pathway_nodes, FCN.new_pathway_edges, FCN.new_energies = FCN.transform_nodes_and_edges(pathway_nodes_final,
                                                                                                       pathway_edges_final,
                                                                                                       starting_mols_list,
                                                                                                       node_energies)

