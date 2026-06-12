"""
Module has the function to calculate the VSIE operators for cross interactions.
"""

import numpy as _np
from numba import jit, prange

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
def _njit_cross_single_layer(wavenumber, dom, codom, alphacod, betacod, weights):
    """
    The Single Layer Volume Integral Operator for disjoint domain and range geometry.
    Accelerated with numba.

    Parameters
    -----------
    wavenumber: float, complex
        Exterior wave number.
    dom: np.ndarray
        Centers of the voxels of the domain in order.
        The shape is N1 x 3, with N1 the number of voxels.
    codom: np.ndarray
        Centers of the voxels of the range in order.
        The shape is N2 x 3, with N2 the number of voxels.
    alphacod: np.ndarray
        Values of alpha in the voxels of codomain.
    betacod: np.ndarray
        Values of beta in the voxels of codomain.
    weights: np.ndarray 
        Volume of each voxel of the codomain. 

    Returns
    --------
    b_matrix: np.ndarray
        Values of the operator B in each voxel.
        The shape is N1 x N2.
    """
    n_vox_codom = len(codom)
    n_vox_dom = len(dom)
    b_matrix = _np.zeros((n_vox_dom,n_vox_codom), dtype=_np.complex128)
    for i in prange(n_vox_dom):
        for j in range(n_vox_codom):
            ri_a_rj = dom[i] - codom[j]
            norm = _norm(ri_a_rj)
            b_matrix[i,j] = weights[j] * _np.exp(1j*wavenumber*norm)/(4*_np.pi*norm)
            b_matrix[i, j] = (betacod[j] - wavenumber**2*alphacod[j]) * b_matrix[i, j]
    return b_matrix


@jit(nopython=True, parallel=True)
def _njit_cross_double_layer(wavenumber, dom, codom, normals, diff_alpha, weights):
    """
    Double Layer Volume Integral Operator for cross interactions between
    disjoint domain and range space.
    Accelerated with numba.

    Parameters
    -----------
    wavenumber: float, complex
        Exterior wave number.
    dom: np.ndarray
        Centers of the voxels of the domain in order.
        The shape is N1 x 3, with N1 the number of voxels.
    codom: np.ndarray
        Centers of the voxels of the range in order.
        The shape is N2 x 3, with N2 the number of voxels.
    normals: np.ndarray
        Directions of the normal vectors in each cell of the surface.
    alpha_ext: np.ndarray
        Values of alpha in the voxels of exterior space.
    alpha_int: np.ndarray
        Values of alpha in the voxels of interior space.
    weights: np.ndarray 
        Volume of each voxel of the codomain. 

    Returns
    --------
    d_matrix: np.ndarray
        Values of the operator D in each voxel.
        The shape is N1 x N2.
    """
    n_vox_dom = len(dom)
    n_vox_codom = len(codom)
    d_matrix = _np.zeros((n_vox_dom,n_vox_codom), dtype=_np.complex128)
    for i in prange(n_vox_dom):
        for j in range(n_vox_codom):
            ri_a_rj = dom[i] - codom[j]
            norm = _norm(ri_a_rj)
            normal_dot = _dot(ri_a_rj, normals[j])
            d_matrix[i,j] = weights[j] * _np.exp(1j*wavenumber*norm)/(4*_np.pi*(norm**2))*\
                            (1/norm -1j*wavenumber) * normal_dot
            d_matrix[i,j] = diff_alpha[j] * d_matrix[i, j]
    return d_matrix

@jit(nopython=True, parallel=True)
def _njit_cross_ad_double_layer(wavenumber, dom, codom, grad_alpha, weights):
    """
    Adjoint Double Layer Volume Integral Operator for cross interaction.
    Accelerated with numba.

    Parameters
    -----------
    wavenumber: float, complex
        Exterior wave number.
    dom: np.ndarray
        Centers of the voxels of the space in order.
        The shape is N x 3, with N the number of voxels.
    grad_alpha: np.ndarray
        Values of the gradient of alpha in the voxels of codomain.
    weights: np.ndarray 
        Volume of each voxel of the codomain. 

    Returns
    --------
    c_matrix: np.ndarray
        Values of the operator C in each voxel.
        The shape is N x N.
    """
    n_vox_dom = dom.shape[0]
    n_vox_codom = codom.shape[0]
    c_matrix = _np.zeros((n_vox_dom,n_vox_codom), dtype=_np.complex128)

    for i in prange(n_vox_dom):
        for j in range(n_vox_codom):
            ri_a_rj = dom[i] - codom[j]
            norm = _norm(ri_a_rj)
            aux = weights[j] * (_np.exp(1j * wavenumber * norm) /\
                (4*_np.pi * norm**2)) * (1j*wavenumber- 1/norm)
            c_matrix[i, j] = aux * (ri_a_rj[0]*grad_alpha[j][0] + ri_a_rj[1]*grad_alpha[j][1] \
                              + ri_a_rj[2]*grad_alpha[j][2])

    return c_matrix

def cross_single_layer(domain, codomain):
    """
    The Single Layer Volume Integral Operator for disjoint domain and range geometry.

    Parameters
    -----------
    domain: Subdomain
        Subdomain object with the domain geometry.
    codomain: Subdomain
        Subdomain object with the codomain geometry.
    
    Returns
    --------
    b_matrix: np.ndarray
        Values of the operator B in each voxel.
        The shape is N1 x N2.
    """
    wavenumber = domain.problem.ext_wavenumber
    dom = domain.mesh
    codom = codomain.mesh
    alphacod = codomain.alpha
    betacod = codomain.beta
    weights = codomain.volumes

    b_matrix = _njit_cross_single_layer(wavenumber, dom, codom, alphacod, betacod, weights)

    return b_matrix

def cross_double_layer(domain, codomain):
    """
    Double Layer Volume Integral Operator for cross interactions between
    disjoint domain and range space.

    Parameters
    -----------
    domain: Subdomain
        Subdomain object with the domain geometry.
    codomain: Subdomain
        Subdomain object with the codomain geometry.
    
    Returns
    --------
    d_matrix: np.ndarray
        Values of the operator D in each voxel.
        The shape is N1 x N2.
    """
    wavenumber = domain.problem.ext_wavenumber
    dom = domain.mesh
    codom = codomain.mesh
    normals = codomain.normals
    diff_alpha = codomain.diff_alpha
    weights = codomain.volumes

    d_matrix = _njit_cross_double_layer(wavenumber, dom, codom, normals, diff_alpha, weights)

    return d_matrix    

def cross_ad_double_layer(domain, codomain):
    """
    Adjoint Double Layer Volume Integral Operator for cross interaction.

    Parameters
    -----------
    domain: Subdomain
        Subdomain object with the domain geometry.
    codomain: Subdomain
        Subdomain object with the codomain geometry.
    
    Returns
    --------
    c_matrix: np.ndarray
        Values of the operator C in each voxel.
        The shape is N1 x N2.
    """
    wavenumber = domain.problem.ext_wavenumber
    dom = domain.mesh
    codom = codomain.mesh
    grad_alpha = codomain.alpha_gradient
    weights = codomain.volumes

    c_matrix = _njit_cross_ad_double_layer(wavenumber, dom, codom, grad_alpha, weights)

    return c_matrix
