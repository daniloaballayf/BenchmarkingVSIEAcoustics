"""
Module has functions to calculate the VSIE operators.
"""

import numpy as _np
from numba import jit

@jit(nopython=True)
def _incident_plane(wavenumber, dom, direction):
    """
    Incident Plane Wave field for the volume integral formulation.
    Accelerated with numba.

    Parameters
    -----------
    wavenumber: float, complex
        Exterior wave number.
    dom: np.ndarray
        Centers of the voxels of the domain in order.
        The shape is N x 3, with N the number of voxels.
    direction: np.ndarray
        Unitary vector with the direction of the plane wave.
        The shape is 3 x 1.

    Returns
    -------
    exp_kr: np.ndarray
        Values of the incident wave planeon every voxel.
        The shape is N x 1, with N the number of voxels.
    """
    krx = wavenumber * direction[0] * dom[:, 0]
    kry = wavenumber * direction[1] * dom[:, 1]
    krz = wavenumber * direction[2] * dom[:, 2]

    total_kr = krx + kry + krz

    exp_kr = _np.exp(1j*total_kr)

    return exp_kr

def incident_plane(subdomain,direction, evaluation_points=None):
    """
    Incident Plane Wave field for the volume integral formulation.

    Parameters
    -----------
    subdomain: Subdomain
        Subdomain of the problem.
    direction: np.ndarray   
        Unitary vector with the direction of the plane wave.
        The shape is 3 x 1.
    evaluation_points: np.ndarray, optional
        Points where the incident plane wave is evaluated.
        The shape is N x 3, with N the number of points.
        If None, the incident plane wave is evaluated in the center of the voxels.

    Returns
    -------
    exp_kr: np.ndarray
        Values of the incident wave planeon every voxel.
        The shape is N x 1, with N the number of voxels.
    """
    wavenumber = subdomain.problem.ext_wavenumber
    dom = subdomain.mesh
    if evaluation_points is not None:
        dom = evaluation_points

    return _incident_plane(wavenumber, dom, direction)
