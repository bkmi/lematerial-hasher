# Copyright 2025 Entalpic
import warnings
import math

from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.local_env import EconNN, NearNeighbors
from pymatgen.core import Structure
from networkx import Graph
from moyopy import MoyoDataset
from moyopy.interface import MoyoAdapter


def get_structure_graph(
    structure: Structure,
    bonding_kwargs: dict = {},
    bonding_algorithm: NearNeighbors = EconNN,
    primitive_reduction: bool = False,
    symprec: float = 0.01,
    rad_angle_tolerance: float = 5 * math.pi / 180,
) -> Graph:
    """Method to build networkx graph object based on
    bonding algorithm from Pymatgen Structure

    Args:
        structure (Structure): Pymatgen Structure object
        bonding_kwargs (dict, optional): kwargs to pass to
            NearNeighbor class. Defaults to {}.
        bonding_algorithm (NearNeighbors, optional): NearNeighbor
            class to build bonded structure. Defaults to EconNN.
        primitive_reduction (bool, optional): Whether to reduce the
            structure to its primitive cell before computing the hash.
            Defaults to False.
        symprec (float, optional): Distance tolerance in Cartesian 
            coordinates to find crystal symmetry. May not be supported 
            for all backends. Defaults  to 0.01, default `symprec` for 
            pytmatgen's `SpacegroupAnalyzer`.
        rad_angle_tolerance (float, optional): Tolerance of angle between 
            basis vectors in degrees to be tolerated in the symmetry 
            finding. Value in radians. Defaults to 5 degrees ~ 0.087 rad,
            default `angle_tolerance` for pytmatgen's `SpacegroupAnalyzer`.

    Returns:
        Graph: networkx Graph object
    """
    assess_structure = (
        MoyoAdapter.get_structure(
            MoyoDataset(
                MoyoAdapter.from_structure(structure),
                symprec=symprec,
                angle_tolerance=rad_angle_tolerance,
            ).prim_std_cell
        )
        if primitive_reduction
        else structure.copy()
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        structure_graph = StructureGraph.from_local_env_strategy(
            structure=assess_structure,
            strategy=bonding_algorithm(**bonding_kwargs),
        )
    for n, site in zip(range(len(assess_structure)), assess_structure):
        structure_graph.graph.nodes[n]["specie"] = site.specie.name
    for edge in structure_graph.graph.edges:
        structure_graph.graph.edges[edge]["voltage"] = structure_graph.graph.edges[
            edge
        ]["to_jimage"]

    return structure_graph.graph
