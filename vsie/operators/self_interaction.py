"""
Module has the function to calculate the VSIE operators for self interactions.
"""

import numpy as _np
from numba import jit, prange

_np.seterr(all='ignore')

@jit(nopython=True)
def _norm(vector):
    aux = 0
    for i in range(len(vector)):
        aux += vector[i] * vector[i]
    return aux**0.5

@jit(nopython=True)
def _dot(vector1, vector2):
    aux = 0
    for i in range(len(vector1)):
        aux += vector1[i] * vector2[i]
    return aux

@jit(nopython=True, parallel=True)
def _njit_single_layer(wavenumber, dom, alpha, beta, weights):
    """
    Calculates Single Layer Volume Integral Operator for the same domain.
    Accelerated with Numba.

    Parameters
    -----------
    wavenumber: float, complex
        Exterior wave number.
    dom: np.ndarray
        Centers of the voxels of the space in order.
        The shape is N x 3, with N the number of voxels.
    alpha: np.ndarray
        Values of alpha in the voxels.
    beta: np.ndarray
        Values of beta in the voxels.
    weights: np.ndarray 
        Volume of each voxel. 

    Returns
    --------
    b_matrix: np.ndarray
        Values of the operator B in each voxel.
        The shape is N x N.
    """
    n_vox = dom.shape[0]
    b_matrix = _np.zeros((n_vox,n_vox), dtype=_np.complex128)
    for i in prange(n_vox):
        for j in range(n_vox):
            if i == j:
                r_self = (3/(4*_np.pi) * weights[j])**(1/3)
                self = (1/wavenumber**2 - 1j*r_self/wavenumber)*_np.exp(1j*wavenumber*r_self) - 1/wavenumber**2
                b_matrix[i, j] = (beta[j] - wavenumber**2*alpha[j]) * self
            else:
                ri_a_rj = dom[i] - dom[j]
                norm = _norm(ri_a_rj)
                b_matrix[i, j] = weights[j] * _np.exp(1j * wavenumber * norm) / (4*_np.pi * norm)
                b_matrix[i, j] = (beta[j] - wavenumber**2*alpha[j]) * b_matrix[i, j]
    return b_matrix
    

@jit(nopython=True, parallel=True)
def _njit_double_layer(wavenumber, dom, normals, diff_alpha, weights):
    """
    Double Layer Volume Integral Operator for self interactions.
    Accelerated with numba.

    Parameters
    -----------
    wavenumber: float, complex
        Exterior wave number.
    dom: np.ndarray
        Centers of the voxels of the domain in order.
        The shape is N x 3, with N the number of voxels.
    normals: np.ndarray
        Directions of the normal vectors in each cell of the surface.
    diff_alpha: np.ndarray
        Values of the jump of alpha at the surface.
    weights: np.ndarray 
        Volume of each voxel. 

    Returns
    --------
    d_matrix: np.ndarray
        Values of the operator D in each voxel.
        The shape is N x N.
    """
    n_vox = len(dom)
    d_matrix = _np.zeros((n_vox,n_vox), dtype=_np.complex128)

    for i in prange(n_vox):
        for j in range(n_vox):
            if i == j:
                d_matrix[i,j] = 0
            else:
                ri_a_rj = dom[i] - dom[j]
                norm = _norm(ri_a_rj)
                normal_dot = _dot(ri_a_rj, normals[j])
                d_matrix[i,j] = weights[j] * _np.exp(1j*wavenumber*norm)/(4*_np.pi*(norm**2))\
                            * (1/norm -1j*wavenumber) * normal_dot
                d_matrix[i,j] = diff_alpha[j] * d_matrix[i, j]

    return d_matrix

@jit(nopython=True, parallel=True)
def _njit_ad_double_layer(wavenumber, dom, grad_alpha, weights):
    """
    Adjoint Double Layer Volume Integral Operator for self interaction.
    Accelerated with numba.

    Parameters
    -----------
    wavenumber: float, complex
        Exterior wave number.
    dom: np.ndarray
        Centers of the voxels of the space in order.
        The shape is N1 x 3, with N1 the number of voxels.
    codom: np.ndarray
        Centers of the voxels of the range in order.
        The shape is N2 x 3, with N2 the number of voxels.
    grad_alpha: np.ndarray
        Values of the gradient of alpha in the voxels.
    weights: np.ndarray 
        Volume of each voxel. 

    Returns
    --------
    c_matrix: np.ndarray
        Values of the operator C in each voxel.
        The shape is N1 x N2.
    """
    n_vox = dom.shape[0]
    c_matrix = _np.zeros((n_vox,n_vox), dtype=_np.complex128)

    for i in prange(n_vox):
        for j in range(n_vox):
            if i == j:
                c_matrix[i, j] = 0
            else:
                ri_a_rj = dom[i] - dom[j]
                norm = _norm(ri_a_rj)
                aux = weights[j] * (_np.exp(1j*wavenumber*norm) / (4*_np.pi * norm**2))*(1j*wavenumber - 1/norm)
                c_matrix[i, j] = aux * (ri_a_rj[0]*grad_alpha[j][0] + ri_a_rj[1]*grad_alpha[j][1] \
                                 + ri_a_rj[2]*grad_alpha[j][2])

    return c_matrix


def mass_matrix(subdomain):
    """
    Mass Matrix in the volume integral formulation.

    Parameters
    -----------
    subdomain: Subdomain
        Subdomain object with the mesh and the alpha values.
        The mesh is a numpy array with the centers of the voxels in order.

    Returns
    --------
    a_matrix: np.ndarray
        Values of the mass matrix in each voxel.
        The shape is N x N.
    """
    dom = subdomain.mesh
    alpha = subdomain.alpha

    n_vox = dom.shape[0]
    a_matrix = _np.zeros((n_vox,n_vox), dtype=_np.complex128)
    _np.fill_diagonal(a_matrix, alpha + 1)

    return a_matrix

def single_layer(subdomain):
    """
    Single Layer Volume Integral Operator for the same domain.
    Wrapper for the function single_layer.
    Accelerated with Numba.

    Parameters
    -----------
    subdomain: Subdomain
        Subdomain object with the mesh and the alpha values.
        The mesh is a numpy array with the centers of the voxels in order.

    Returns
    --------
    b_matrix: np.ndarray
        Values of the operator B in each voxel.
        The shape is N x N.
    """

    dom = subdomain.mesh
    alpha = subdomain.alpha
    beta = subdomain.beta
    weights = subdomain.volumes
    wavenumber = subdomain.problem.ext_wavenumber

    b_matrix = _njit_single_layer(wavenumber, dom, alpha, beta, weights)

    return b_matrix

def double_layer(subdomain):
    """
    Double Layer Volume Integral Operator for self interactions.
    Wrapper for the function double_layer.
    Accelerated with Numba.

    Parameters
    -----------
    subdomain: Surfdomain
        Surfdomain object with the mesh and the alpha values.
        The mesh is a numpy array with the centers of the voxels in order.

    Returns
    --------
    d_matrix: np.ndarray
        Values of the operator D in each voxel.
        The shape is N x N.
    """

    dom = subdomain.mesh
    normals = subdomain.normals
    diff_alpha = subdomain.diff_alpha
    weights = subdomain.volumes
    wavenumber = subdomain.problem.ext_wavenumber

    d_matrix = _njit_double_layer(wavenumber, dom, normals, diff_alpha, weights)

    return d_matrix

def ad_double_layer(subdomain):
    """
    Adjoint Double Layer Volume Integral Operator for self interactions.
    Wrapper for the function ad_double_layer.
    Accelerated with Numba.

    Parameters
    -----------
    subdomain: Subdomain
        Subdomain object with the mesh and the alpha values.
        The mesh is a numpy array with the centers of the voxels in order.

    Returns
    --------
    c_matrix: np.ndarray
        Values of the operator C in each voxel.
        The shape is N x N.
    """

    dom = subdomain.mesh
    grad_alpha = subdomain.alpha_gradient
    weights = subdomain.volumes
    wavenumber = subdomain.problem.ext_wavenumber

    c_matrix = _njit_ad_double_layer(wavenumber, dom, grad_alpha, weights)

    return c_matrix
    