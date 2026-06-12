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
def _njit_single_layer_potential(space, dom, weigth, u, alpha, beta, wavenumber):
    '''
    Evaluates the Single Layer Potential.
    Accelerated with numba.

    Parameters
    -----------
    space: np.ndarray
        Points of the space to evaluate in order.
        The shape is N x 3, with N the number of points.
    dom: np.ndarray
        Centers of the voxels of the solved domain in order.
        The shape is N1 x 3, with N1 the number of voxels.
    weigth: np.ndarray
        Volume of each voxel.
        The shape is N1 x 1.
    u: np.ndarray
        Values of the function u in the center of the voxels.
        The shape is N1 x 1.
    alpha: np.ndarray
        Values of the function alpha in the center of the voxels.
        The shape is N1 x 1.
    beta: np.ndarray
        Values of the function beta in the center of the voxels.
        The shape is N1 x 1.
    wavenumber: float, complex
        Exterior wave number.

    Returns
    --------
    U: np.ndarray
        Values of the operator in each point of the space.
        The shape is N x 1.
    '''
    n_vox_dom = len(dom)
    n_vox = len(space)
    U = _np.zeros(n_vox, dtype=_np.complex128)
    for i in prange(n_vox):
        for j in range(n_vox_dom):
            ri_a_rj = space[i] - dom[j] 
            norm = _norm(ri_a_rj)
            U[i] += weigth[j] * _np.exp(1j*wavenumber*norm)/(4*_np.pi*norm) * (beta[j] - wavenumber**2*alpha[j]) * u[j]
    return U

@jit(nopython=True, parallel=True)
def _njit_double_layer_potential(space, dom, normals, weigth, u, diff_alpha, wavenumber):
    '''
    Evaluates the Double Layer Potential.
    Accelerated with numba.

    Parameters 
    -----------
    space: np.ndarray
        Points of the space to evaluate in order.
        The shape is N x 3, with N the number of points.
    dom: np.ndarray
        Centers of the voxels of the solved domain in order.
        The shape is N1 x 3, with N1 the number of voxels.
    normals: np.ndarray
        Unitary direction of the normal vector in each cell of the surface.
        The direction must be pointing outwards.
        The shape is N1 x 3, with N1 the number of surface cells.
    weigth: np.ndarray
        Volume of each voxel.
        The shape is N1 x 1.
    u: np.ndarray
        Values of the function u in the center of the voxels.
        The shape is N1 x 1.
    diff_alpha: np.ndarray
        Values of the jump of alpha at the surface.
        The shape is N1 x 1.
    wavenumber: float, complex
        Exterior wave number.
    '''

    n_vox_dom = len(dom)
    n_vox = len(space)
    U = _np.zeros(n_vox, dtype=_np.complex128)
    for i in prange(n_vox):
        for j in range(n_vox_dom):
            ri_a_rj = space[i] - dom[j] 
            norm = _norm(ri_a_rj)
            normal_dot = _dot(ri_a_rj, normals[j])
            U[i] += weigth[j] * _np.exp(1j*wavenumber*norm)/(4*_np.pi*norm**2) * (-1j*wavenumber + 1/norm)\
                 * normal_dot * u[j] * diff_alpha[j]
    return U

@jit(nopython=True, parallel=True)
def _njit_ad_double_layer_potential(space, dom, weigth, u, grad_alpha, wavenumber):
    '''
    Evaluates the Adjoint Double Layer Potential.
    Accelerated with numba.

    Parameters
    -----------
    space: np.ndarray
        Points of the space to evaluate in order.
        The shape is N x 3, with N the number of points.
    dom: np.ndarray
        Centers of the voxels of the solved domain in order.
        The shape is N1 x 3, with N1 the number of voxels.
    weigth: np.ndarray
        Volume of each voxel.
        The shape is N1 x 1.
    u: np.ndarray
        Values of the function u in the center of the voxels.
        The shape is N1 x 1.
    grad_alpha: np.ndarray
        Values of the gradient of alpha in the voxels.
        The shape is N1 x 3.
    wavenumber: float, complex
        Exterior wave number.

    Returns
    --------
    U: np.ndarray
        Values of the operator in each point of the space.
        The shape is N x 1.
    '''

    n_vox_dom = len(dom)  
    n_vox = len(space)
    U = _np.zeros(n_vox, dtype=_np.complex128)
    for i in prange(n_vox):
        for j in range(n_vox_dom):
            ri_a_rj = space[i] - dom[j] 
            norm = _norm(ri_a_rj)
            U[i] += weigth[j] * _np.exp(1j*wavenumber*norm)/(4*_np.pi*norm**2) * (1j*wavenumber - 1/norm)\
                 * (ri_a_rj[0]*grad_alpha[j][0] + ri_a_rj[1]*grad_alpha[j][1] \
                                 + ri_a_rj[2]*grad_alpha[j][2]) * u[j]
    return U

def single_layer_potential(coordinates, solution, subdomain):
    """
    Evaluates the Single Layer Potential.

    Parameters
    -----------
    coordinates: np.ndarray
    Points of the space to evaluate in order.
        The shape is N x 3, with N the number of points.
    solution: np.ndarray
        Values of the solution at the center of the voxels.
        The shape is N1 x 1.
    subdom: Subdomain
        Subdomain object with the properties of the subdomain of the solved solution

    Returns
    --------
    U: np.ndarray
        Values of the operator in each point of the space.
        The shape is N x 1.
    """
    return _njit_single_layer_potential(coordinates, subdomain.mesh, subdomain.volumes, 
                                        solution, subdomain.alpha, subdomain.beta, 
                                        subdomain.problem.ext_wavenumber)

def double_layer_potential(coordinates, solution, subdomain):
    """
    Evaluates the Double Layer Potential.

    Parameters
    -----------
    coordinates: np.ndarray
        Points of the space to evaluate in order.
        The shape is N x 3, with N the number of points.
    solution: np.ndarray
        Values of the solution at the center of the voxels.
        The shape is N1 x 1.
    subdom: Surfdomain
        Surfdomain object with the properties of the subdomain of the solved solution

    Returns
    --------
    U: np.ndarray
        Values of the operator in each point of the space.
        The shape is N x 1.
    """
    return _njit_double_layer_potential(coordinates, subdomain.mesh, subdomain.normals, 
                                        subdomain.volumes, solution, subdomain.diff_alpha, 
                                        subdomain.problem.ext_wavenumber)

def ad_double_layer_potential(coordinates, solution, subdomain):
    """
    Evaluates the Adjoint Double Layer Potential.

    Parameters
    -----------
    coordinates: np.ndarray
        Points of the space to evaluate in order.
        The shape is N x 3, with N the number of points.
    solution: np.ndarray
        Values of the solution at the center of the voxels.
        The shape is N1 x 1.
    subdom: Subdomain
        Subdomain object with the properties of the subdomain of the solved solution

    Returns
    --------
    U: np.ndarray
        Values of the operator in each point of the space.
        The shape is N x 1.
    """
    return _njit_ad_double_layer_potential(coordinates, subdomain.mesh, subdomain.volumes, 
                                        solution, subdomain.alpha_gradient, 
                                        subdomain.problem.ext_wavenumber)