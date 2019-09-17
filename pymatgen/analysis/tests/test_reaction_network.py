# coding: utf-8


import os
import unittest
import time
from pymatgen.core.structure import Molecule
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import OpenBabelNN
from pymatgen.util.testing import PymatgenTest
from pymatgen.analysis.reaction_network import ReactionNetwork, convert_to_igraph
from pymatgen.entries.mol_entry import MoleculeEntry
from monty.serialization import dumpfn, loadfn
from pymatgen.analysis.fragmenter import metal_edge_extender

try:
    import openbabel as ob
except ImportError:
    ob = None

__author__ = "Samuel Blau"
__email__ = "samblau1@gmail.com"

test_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                        'test_files', 'reaction_network_files')


class TestReactionNetwork(PymatgenTest):
    @classmethod
    def setUpClass(cls):
        LiEC_mg =  MoleculeGraph.with_local_env_strategy(
            Molecule.from_file(os.path.join(test_dir,"LiEC.xyz")),
            OpenBabelNN(),
            reorder=False,
            extend_structure=False)
        cls.LiEC_mg = metal_edge_extender(LiEC_mg)

        LEDC_mg =  MoleculeGraph.with_local_env_strategy(
            Molecule.from_file(os.path.join(test_dir,"LEDC.xyz")),
            OpenBabelNN(),
            reorder=False,
            extend_structure=False)
        cls.LEDC_mg = metal_edge_extender(LEDC_mg)

        LEMC_mg =  MoleculeGraph.with_local_env_strategy(
            Molecule.from_file(os.path.join(test_dir,"LEMC.xyz")),
            OpenBabelNN(),
            reorder=False,
            extend_structure=False)
        cls.LEMC_mg = metal_edge_extender(LEMC_mg)

        cls.LiEC_extended_entries = []
        entries = loadfn(os.path.join(test_dir,"LiEC_extended_entries.json"))
        for entry in entries:
            mol = entry["output"]["optimized_molecule"]
            E = float(entry["output"]["final_energy"])
            H = float(entry["output"]["enthalpy"])
            S = float(entry["output"]["entropy"])
            mol_entry = MoleculeEntry(molecule=mol,energy=E,enthalpy=H,entropy=S,entry_id=entry["task_id"])
            cls.LiEC_extended_entries.append(mol_entry)

    def _test_build_graph(self):
        RN = ReactionNetwork(
            self.LiEC_extended_entries,
            electron_free_energy=-2.15,
            free_energy_cutoff=4.0,
            prereq_cutoff=4)
        self.assertEqual(len(RN.entries_list),251)
        self.assertEqual(len(RN.graph.nodes),1511)
        self.assertEqual(len(RN.graph.edges),3484)
        # dumpfn(RN,"RN.json")
        loaded_RN = loadfn("RN.json")
        self.assertEqual(RN.as_dict(),loaded_RN.as_dict())

    def test_find_paths(self):
        RN = loadfn("RN.json")
        LiEC_ind = None
        LEDC_ind = None
        for entry in RN.entries["C3 H4 Li1 O3"][12][1]:
            if self.LiEC_mg.isomorphic_to(entry.mol_graph):
                LiEC_ind = entry.parameters["ind"]
                break
        for entry in RN.entries["C4 H4 Li2 O6"][17][0]:
            if self.LEDC_mg.isomorphic_to(entry.mol_graph):
                LEDC_ind = entry.parameters["ind"]
                break
        paths = RN.find_paths([LiEC_ind],LEDC_ind,weight="softplus",use_igraph=False,num_paths=10)
        # self.assertEqual(paths[0]["overall_free_energy_change"],-5.131657887139408)
        # self.assertEqual(paths[0]["cost"],1.7660275897855464)
        # self.assertEqual(paths[0]["hardest_step_deltaG"],0.36044270861384575)
        for path in paths:
            for val in path:
                print(val, path[val])
            print()

    def _test_solve_prerequisites(self):
        RN = loadfn("RN.json")
        LiEC_ind = None
        LEDC_ind = None
        for entry in RN.entries["C3 H4 Li1 O3"][12][1]:
            if self.LiEC_mg.isomorphic_to(entry.mol_graph):
                LiEC_ind = entry.parameters["ind"]
                break
        for entry in RN.entries["C4 H4 Li2 O6"][17][0]:
            if self.LEDC_mg.isomorphic_to(entry.mol_graph):
                LEDC_ind = entry.parameters["ind"]
                break
        RN.solve_prerequisites([LiEC_ind],LEDC_ind,weight="softplus")

    def _test_convert_to_igraph(self):
        RN = loadfn("RN.json")
        ig = convert_to_igraph(RN.graph)
        print(ig.vs.find("220"))
        print(ig.vs.find("220").index)
        print(ig.vs.find("220,219"))
        # print(ig.vs[220])
        print(ig.es.find(_source=437,_target=219))
        print(ig.es.find(_source=437,_target=219).source,ig.es.find(_source=437,_target=219).target)

        # print(ig)

    def _test_find_paths_igraph(self):
        RN = loadfn("RN.json")
        LiEC_ind = None
        LEDC_ind = None
        for entry in RN.entries["C3 H4 Li1 O3"][12][1]:
            if self.LiEC_mg.isomorphic_to(entry.mol_graph):
                LiEC_ind = entry.parameters["ind"]
                break
        for entry in RN.entries["C4 H4 Li2 O6"][17][0]:
            if self.LEDC_mg.isomorphic_to(entry.mol_graph):
                LEDC_ind = entry.parameters["ind"]
                break
        RN.find_paths([LiEC_ind],LEDC_ind,weight="softplus",use_igraph=True,num_paths=10)



if __name__ == "__main__":
    unittest.main()
