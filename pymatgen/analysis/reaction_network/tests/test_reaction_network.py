# coding: utf-8


import os
import unittest
import time
from pymatgen.core.structure import Molecule
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import OpenBabelNN
from pymatgen.util.testing import PymatgenTest
from pymatgen.analysis.reaction_network.reaction_network import ReactionNetwork
from pymatgen.entries.mol_entry import MoleculeEntry
from monty.serialization import dumpfn, loadfn
from pymatgen.analysis.fragmenter import metal_edge_extender

try:
    import openbabel as ob
except ImportError:
    ob = None

__author__ = "Samuel Blau"
__email__ = "samblau1@gmail.com"

test_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..","..",
                        'test_files', 'reaction_network_files')


class TestReactionNetwork(PymatgenTest):
    @classmethod
    def setUpClass(cls):
        EC_mg =  MoleculeGraph.with_local_env_strategy(
            Molecule.from_file(os.path.join(test_dir,"EC.xyz")),
            OpenBabelNN(),
            reorder=False,
            extend_structure=False)
        cls.EC_mg = metal_edge_extender(EC_mg)

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

        H2O_mg =  MoleculeGraph.with_local_env_strategy(
            Molecule.from_file(os.path.join(test_dir,"H2O.xyz")),
            OpenBabelNN(),
            reorder=False,
            extend_structure=False)
        cls.H2O_mg = metal_edge_extender(H2O_mg)

        cls.LiEC_extended_entries = []
        entries = loadfn(os.path.join(test_dir,"LiEC_extended_entries.json"))
        for entry in entries:
            mol = entry["output"]["optimized_molecule"]
            E = float(entry["output"]["final_energy"])
            H = float(entry["output"]["enthalpy"])
            S = float(entry["output"]["entropy"])
            mol_entry = MoleculeEntry(molecule=mol,energy=E,enthalpy=H,entropy=S,entry_id=entry["task_id"])
            cls.LiEC_extended_entries.append(mol_entry)

        cls.LiEC_reextended_entries = []
        entries = loadfn(os.path.join(test_dir,"LiEC_reextended_entries.json"))
        for entry in entries:
            if "optimized_molecule" in entry["output"]:
                mol = entry["output"]["optimized_molecule"]
            else:
                mol = entry["output"]["initial_molecule"]
            E = float(entry["output"]["final_energy"])
            H = float(entry["output"]["enthalpy"])
            S = float(entry["output"]["entropy"])
            mol_entry = MoleculeEntry(molecule=mol,energy=E,enthalpy=H,entropy=S,entry_id=entry["task_id"])
            if mol_entry.formula == "Li1":
                if mol_entry.charge == 1:
                    cls.LiEC_reextended_entries.append(mol_entry)
            else:
                cls.LiEC_reextended_entries.append(mol_entry)

    def _test_LEMC(self):
        RN = ReactionNetwork(
            self.LiEC_reextended_entries,
            electron_free_energy=-2.15)
        self.assertEqual(len(RN.entries_list),569)
        self.assertEqual(len(RN.graph.nodes),10482)
        self.assertEqual(len(RN.graph.edges),22893)

        EC_ind = None
        LEMC_ind = None
        H2O_ind = None
        for entry in RN.entries["C3 H4 O3"][10][0]:
            if self.EC_mg.isomorphic_to(entry.mol_graph):
                EC_ind = entry.parameters["ind"]
                break
        for entry in RN.entries["C3 H5 Li1 O4"][13][0]:
            if self.LEMC_mg.isomorphic_to(entry.mol_graph):
                LEMC_ind = entry.parameters["ind"]
                break
        for entry in RN.entries["H2 O1"][2][0]:
            if self.H2O_mg.isomorphic_to(entry.mol_graph):
                H2O_ind = entry.parameters["ind"]
                break
        Li1_ind = RN.entries["Li1"][0][1][0].parameters["ind"]
        OHminus_ind = RN.entries["H1 O1"][1][-1][0].parameters["ind"]
        self.assertEqual(EC_ind,456)
        self.assertEqual(LEMC_ind,469)
        self.assertEqual(H2O_ind,554)
        self.assertEqual(Li1_ind,556)
        self.assertEqual(OHminus_ind,548)


        # PR_paths, paths = RN.find_paths([H2O_ind],OHminus_ind,weight="softplus",num_paths=10)
        # PR_paths, paths = RN.find_paths([EC_ind,Li1_ind,H2O_ind],OHminus_ind,weight="softplus",num_paths=10)
        PR_paths, paths = RN.find_paths([EC_ind,Li1_ind,H2O_ind],LEMC_ind,weight="softplus",num_paths=10)
        # for path in paths:
        #     for val in path:
        #         print(val, path[val])
        #     print()
        # RN.identify_sinks()

    def test_reextended(self):
        RN = ReactionNetwork(
            self.LiEC_reextended_entries,
            electron_free_energy=-2.15)
        self.assertEqual(len(RN.entries_list),569)
        self.assertEqual(len(RN.graph.nodes),10482)
        self.assertEqual(len(RN.graph.edges),22893)
        # print(len(RN.entries_list))
        # print(len(RN.graph.nodes))
        # print(len(RN.graph.edges))

        EC_ind = None
        LEDC_ind = None
        LiEC_ind = None
        for entry in RN.entries["C3 H4 Li1 O3"][12][1]:
            if self.LiEC_mg.isomorphic_to(entry.mol_graph):
                LiEC_ind = entry.parameters["ind"]
                break
        for entry in RN.entries["C3 H4 O3"][10][0]:
            if self.EC_mg.isomorphic_to(entry.mol_graph):
                EC_ind = entry.parameters["ind"]
                break
        for entry in RN.entries["C4 H4 Li2 O6"][17][0]:
            if self.LEDC_mg.isomorphic_to(entry.mol_graph):
                LEDC_ind = entry.parameters["ind"]
                break
        Li1_ind = RN.entries["Li1"][0][1][0].parameters["ind"]
        self.assertEqual(EC_ind,456)
        self.assertEqual(LEDC_ind,511)
        self.assertEqual(Li1_ind,556)
        self.assertEqual(LiEC_ind,424)
        # print(EC_ind)
        # print(LEDC_ind)
        # print(Li1_ind)
        # print(LiEC_ind)

        PR_paths, paths = RN.find_paths([EC_ind,Li1_ind],LEDC_ind,weight="softplus",num_paths=10)
        # PR_paths, paths = RN.find_paths([LiEC_ind],LEDC_ind,weight="softplus",num_paths=10)
        self.assertEqual(paths[0]["byproducts"],[164])
        self.assertEqual(paths[1]["all_prereqs"],[556,420,556])
        self.assertEqual(paths[0]["cost"],2.313631862390461)
        self.assertEqual(paths[0]["overall_free_energy_change"],-6.240179642711564)
        self.assertEqual(paths[0]["hardest_step_deltaG"],0.3710129384598986)
        self.assertEqual(paths[0]["all_prereqs"],[556,41,556])

        # PR_paths, paths = RN.find_paths([LiEC_ind],LEDC_ind,weight="softplus",num_paths=10)
        # PR_paths, paths = RN.find_paths([LiEC_ind],42,weight="softplus",num_paths=10)
        # PR_paths, paths = RN.find_paths([EC_ind,Li1_ind],LiEC_ind,weight="softplus",num_paths=10)
        # PR_paths, paths = RN.find_paths([EC_ind,Li1_ind],42,weight="exponent",num_paths=10)
        # for path in paths:
        #     for val in path:
        #         print(val, path[val])
        #     print()

    def test_build_graph(self):
        RN = ReactionNetwork(
            self.LiEC_extended_entries,
            electron_free_energy=-2.15)
        self.assertEqual(len(RN.entries_list),251)
        self.assertEqual(len(RN.graph.nodes),2021)
        self.assertEqual(len(RN.graph.edges),4022)
        # dumpfn(RN,"RN.json")
        loaded_RN = loadfn("RN.json")
        self.assertEqual(RN.as_dict(),loaded_RN.as_dict())

    def test_solve_prerequisites(self):
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
        RN.num_starts = 1
        PRs = RN.solve_prerequisites([LiEC_ind],LEDC_ind,weight="softplus")
        # dumpfn(PRs,"PRs.json")
        loaded_PRs = loadfn("PRs.json")
        for key in PRs:
            if PRs[key] == {}:
                self.assertEqual(PRs[key],loaded_PRs[str(key)])
            # else:
            #     self.assertEqual(PRs[key][LiEC_ind],loaded_PRs[str(key)][str(LiEC_ind)])
            elif PRs[key][LiEC_ind] != loaded_PRs[str(key)][str(LiEC_ind)]:
                new_path = RN.characterize_path(path=PRs[key][LiEC_ind]["path"],weight="softplus",PR_paths=PRs,final=True)
                old_path = RN.characterize_path(path=loaded_PRs[str(key)][str(LiEC_ind)]["path"],weight="softplus",PR_paths=PRs,final=True)
                self.assertTrue(abs(new_path["hardest_step_deltaG"]-old_path["hardest_step_deltaG"])<0.000000000001)
                self.assertTrue(abs(new_path["overall_free_energy_change"]-old_path["overall_free_energy_change"])<0.000000000001)
                self.assertTrue(abs(new_path["cost"]-old_path["cost"])<0.000000000001)

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
        PR_paths, paths = RN.find_paths([LiEC_ind],LEDC_ind,weight="softplus",num_paths=10)
        self.assertEqual(paths[0]["cost"],1.7660275897855464)
        self.assertEqual(paths[0]["overall_free_energy_change"],-5.131657887139409)
        self.assertEqual(paths[0]["hardest_step_deltaG"],0.36044270861384575)
        self.assertEqual(paths[9]["cost"],3.7546340395839226)
        self.assertEqual(paths[9]["overall_free_energy_change"],-5.13165788713941)
        self.assertEqual(paths[9]["hardest_step_deltaG"],2.7270388301945787)

if __name__ == "__main__":
    unittest.main()
