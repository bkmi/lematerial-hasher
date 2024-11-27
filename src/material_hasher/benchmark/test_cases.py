import random
from typing import Union

import numpy as np
from pymatgen.core import Structure, SymmOp

def get_new_structure_with_gaussian_noise(
    structure: Structure, sigma: float = 0.001
) -> Structure:
    """Returns new structure with gaussian noise on atomic positions

    Args:
        structure (Structure): Input structure to modify
        sigma (float, optional): Noise applied to atomic position. Defaults to 0.001.

    Returns:
        Structure: new structure with modification
    """
    return Structure(
        structure.lattice,
        structure.species,
        structure.cart_coords
        + np.random.normal(np.zeros(structure.cart_coords.shape), sigma),
        coords_are_cartesian=True,
    )

def get_new_structure_with_isometric_strain(
    structure: Structure, pct: float
) -> Structure:
    """_summary_

    Args:
        structure (Structure): Input structure to modify
        pct (float): Strain applied to lattice in all direction

    Returns:
        Structure: new structure with modification
    """
    s = structure.copy()
    return s.scale_lattice(structure.volume * pct)

def get_new_structure_with_strain(structure: Structure, sigma: float) -> Structure:
    """_summary_

    Args:
        structure (Structure): Input structure to modify
        sigma (float): Percent noise applied to lattice vectors

    Returns:
        Structure: new structure with modification
    """
    s = structure.copy()
    return s.apply_strain(np.random.normal(np.zeros(3), sigma))

def get_new_structure_with_translation(
    structure: Structure, sigma: float
) -> Structure:
    """_summary_

    Args:
        structure (Structure): Input structure to modify
        sigma (float): Noise to apply to lattice vectors

    Returns:
        Structure: new structure with modification
    """
    s = structure.copy()
    return s.translate_sites(
        [n for n in range(len(structure))], np.random.normal(np.zeros(3), sigma)
    )

def get_new_structure_with_symm_ops(
    structure: Structure, symm_ops: Union[SymmOp]
) -> Structure:
    """_summary_

    Args:
        structure (Structure): Input structure to modify
        symm_ops (Union[SymmOp]): List of symmetry operation to test on structure

    Returns:
        Structure: new structure with modification
    """
    s = structure.copy()
    symm_op = random.choices(symm_ops)
    return s.apply_operation(symm_op)
