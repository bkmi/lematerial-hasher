from aflow_xtal_finder import XtalFinder
from monty.tempfile import ScratchDir
from pymatgen.core import Structure

from monty.json import MSONable

from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.structure_analyzer import SpacegroupAnalyzer
from pymatgen.analysis.local_env import CrystalNN, VoronoiNN, EconNN, NearNeighbors

from structuregraph_helpers.hash import generate_hash
import copy

from scipy.stats import wasserstein_distance
import time

from itertools import combinations
import networkx as nx

import numpy as np

from shutil import which

import json

import jsonpickle
import hashlib

## Hash function / structure matcher
# For EconNN, cutoff of np.average([x.atomic_radius for x in structure.composition])*3 and tol of 0.9
# could be interesting

class EntalpicMaterialsHasher():
    def __init__(self, 
                 graphing_algorithm: str = 'WL',
                 bonding_algorithm: NearNeighbors = EconNN,
                 bonding_kwargs: dict = {},
                 include_composition: bool = False,
                 symmetry_labeling: str = 'PMG',):
        self.graphing_algorithm = graphing_algorithm
        self.bonding_algorithm = bonding_algorithm
        self.bonding_kwargs = bonding_kwargs
        self.include_composition = include_composition
        self.symmetry_labeling = symmetry_labeling
        

    def get_weisfeiler_lehman_hash(self,structure,):
        structure_graph = StructureGraph.with_local_env_strategy(
            structure=structure,
            strategy=self.bonding_algorithm(**self.bonding_kwargs),)
        for n, site in zip(range(len(structure)), structure):
            structure_graph.graph.nodes[n]['specie'] = site.specie.name
        for edge in structure_graph.graph.edges:
            structure_graph.graph.edges[edge]['voltage'] = structure_graph.graph.edges[edge]['to_jimage']
        return generate_hash(structure_graph.graph, True, False, 100)

    def get_entalpic_materials_data(self, structure):
        data = dict()
        if self.graphing_algorithm == 'WL':
            data['bonding_graph_hash'] = self.get_weisfeiler_lehman_hash(structure)
        else:
            raise ValueError("Graphing algorithm {} not implemented".format(self.graphing_algorithm))
        if self.symmetry_labeling == 'AFLOW':
            data['symmetry_label'] = AFLOWStructureMatcher().aflow_label(
                structure)['aflow_prototype_label']
        elif self.symmetry_labeling == 'PMG':
            sga = SpacegroupAnalyzer(structure)
            data['symmetry_label'] = sga.get_symmetry_dataset().number
        else:
            raise ValueError("Symmetry algorithm {} not implemented".format(self.symmetry_labeling))
        if self.include_composition:
            data['composition'] = structure.composition.formula.replace(' ','')
        return data

    def get_entalpic_material_id(self, structure):
        data = self.get_entalpic_materials_data(structure)
        return "_".join([str(v) for k,v in data.items()])

class AFLOWStructureMatcher(MSONable):
    '''
    Structure matcher adapter class for AFLOW modeled similarly to StructureMatcher class in PMG
    '''

    def __init__(self, aflow_executable = None):
        self._structure1 = None
        self._structure2 = None
        self._data = None

        self.aflow_executable = aflow_executable or which("aflow")

        print('aflow found in {}'.format(self.aflow_executable))

        if self.aflow_executable is None:
            raise RuntimeError(
                "Requires aflow to be in the PATH or the absolute path to "
                f"the binary to be specified via {self.aflow_executable=}.\n"
            )

    def aflow_label(self, structure: Structure, tolerance=0.2)-> dict:
        '''
        Returns AFLOW label for a given structure

        :param structure: pymatgen structure object
        :return: dict (aflow label, params)
        '''
        xtf = XtalFinder(self.aflow_executable)
        with ScratchDir("."):
            structure.to_file('POSCAR')
            data = xtf.get_prototype_label(['POSCAR',], options="--tolerance={}".format(tolerance))
        return data

    def fit(self, structure1: Structure, structure2: Structure)->bool:
        '''
        Returns whether two structures are the same

        :param structure1: Structure
        :param structure2: Structure
        :return: bool
        '''

        if not self._data and (self._structure1 != structure1 or self._structure2 != structure2):
            self._run_aflow(structure1, structure2)
        return len(self._data[0].get('structures_duplicate',[]))>0

    def _run_aflow(self, structure1: Structure, structure2: Structure):
        xtf = XtalFinder(self.aflow_executable)
        self._structure1 = structure1
        self._structure2 = structure2
        with ScratchDir("."):
            structure1.to_file('POSCAR')
            structure2.to_file('POSCAR2')
            self._data = xtf.compare_structures(['POSCAR', 'POSCAR2'])

    def get_mapping(self, structure1: Structure, structure2: Structure)->dict:
        '''
        Returns how structure1 -> structure2 by atom ID's
        :param structure1: Structure
        :param structure2: Structure
        :return: dict, mapping
        '''
        if not self._data and (self._structure1 != structure1 or self._structure2 != structure2):
            self._run_aflow(structure1, structure2)
        return {n:v for n,v in enumerate(self._data['structures_duplicate'][0]['atom_map'])}

    def get_transformation(self, structure1: Structure, structure2: Structure):
        if not self._data and (self._structure1 != structure1 or self._structure2 != structure2):
            self._run_aflow(structure1, structure2)
            mapping = {n: v for n, v in enumerate(self._data['structures_duplicate'][0]['atom_map'])}
            supercell = self._data['structures_duplicate'][0]['basis_transformation']
            vector = [site1.coords-site2.coords for site1,site2 in enumerate(structure1,structure2[mapping])]
        return supercell, vector, mapping

    def get_similarity_metrics(self, structure1: Structure, structure2:Structure)-> dict:
        '''
        Returns aflow similarity metrics
        :param structure1:
        :param structure2:
        :return:  misfit, lattice_deviation, coordinate_displacement, basis_transformation, rotation, origin_shift,
        atom_map, basis_map
        '''
        if not self._data and (self._structure1 != structure1 or self._structure2 != structure2):
            self._run_aflow(structure1, structure2)
            d = self._data['structures_duplicates'][0]
            d.pop('name')
            d.pop('number_compounds_matching_structure')
            d.pop('failure')
            return d