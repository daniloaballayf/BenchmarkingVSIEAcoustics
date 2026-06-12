"""
Module has the functions to use voxel data.
"""
import numpy as _np


def data_physical_functions(wavenumber, ext_density, dom_wave, dom_density, dom):
    """
    Creates array with values of alpha, beta in
    the domain.

    Parameters
    -----------
    wavenumber: float, complex
        Exterior wave number.
    ext_density: float
        Exterior density value.
    dom_wave: np.ndarray
        Value of the wavespeed in the nodes.
        The shape is N x 3, with N the number of voxels.
    dom_density: np.ndarray
        Value of the density in the nodes
        The shape is N x 3, with N the number of voxels.

    Returns
    --------
    alpha: np.ndarray
        Values of alpha in each voxel of exterior cube.
        The shape is N x 3, with N1 number of voxels in the domain.
    beta: np.ndarray
        Values of beta in each voxel of exterior cube.
        The shape is N x 3, with N1 number of voxels in the domain.
    """

    num_voxels = dom.shape[0]
    alpha = _np.zeros(num_voxels, dtype=_np.complex128)
    beta = _np.zeros(num_voxels, dtype=_np.complex128)

    alpha[:] = ext_density/dom_density - 1
    beta[:] = ext_density*dom_wave**2/dom_density - wavenumber**2
    
    return alpha, beta

def data_discont_physical_functions(
        wavenumber,
        ext_density,  
        intdom_wave, 
        extdom_wave,
        intdom_density,
        extdom_density, 
        space, 
        idx_int, 
        idx_ext):
    """
    Creates array with values of alpha and beta in the domain.

    Parameters
    -----------
    wavenumber: float, complex
        Exterior wave number.
    ext_density: float
        Exterior density value.
    intdom_wave: np.ndarray
        Value of the wavespeed in the nodes of the interior domain.
        The shape is N x 3, with N the number of voxels.
    extdom_wave: np.ndarray
        Value of the wavespeed in the nodes of the exterior domain.
        The shape is N x 3, with N the number of voxels.
    intdom_density: np.ndarray
        Value of the density in the nodes of the interior domain.
        The shape is N x 3, with N the number of voxels.
    extdom_density: np.ndarray
        Value of the density in the nodes of the exterior domain.
        The shape is N x 3, with N the number of voxels.
    space: np.ndarray
        Centers of the voxels of the space in order.
        The shape is N x 3, with N the number of voxels.
    idx_int: np.ndarray
        A boolean array with the information of which voxels are in the interior cube.
        The shape is N x 1, with N the number of voxels.
    idx_ext: np.ndarray
        A boolean array with the information of which voxels are in the interior cube.
        The shape is N x 1, with N the number of voxels.

    Returns
    --------
    alpha: np.ndarray
        Values of alpha in each voxel of exterior cube.
        The shape is N x 3, with N1 number of voxels in the domain.
    beta: np.ndarray
        Values of beta in each voxel of exterior cube.
        The shape is N x 3, with N1 number of voxels in the domain.
    """
    num_voxels = space.shape[0]
    alpha = _np.zeros(num_voxels, dtype=_np.complex128)
    beta = _np.zeros(num_voxels, dtype=_np.complex128)
    
    alpha[idx_int] = ext_density/intdom_density
    alpha[idx_int] +=  - 1
    alpha[idx_ext] = ext_density/extdom_density
    alpha[idx_ext] +=  - 1

    beta[idx_int] = ext_density*intdom_wave**2\
                /intdom_density - wavenumber**2
    beta[idx_ext] = ext_density*extdom_wave**2\
                /extdom_density - wavenumber**2
    
    return alpha, beta

def data_grad(dom, dom_density, density_x, density_y, density_z, ext_density):
    """
    Creates array with the values of the gradient of alpha in the domain. 
    
    Parameters
    -----------
    dom: np.ndarray
        Centers of the voxels of the space in order.
        The shape is N x 3, with N the number of voxels.
    dom_density: np.ndarray
        Array with the value of the density in the domain. 
        The shape is N x 3, with N the number of voxels.
    density_x: np.ndarray
        Array with the value of the partial derivative of 
        the density  with respecto to x in the domain.
        The shape is N x 3, with N the number of voxels.
    density_y: np.ndarray
        Array with the value of the partial derivative of 
        the density  with respecto to y in the domain.
        The shape is N x 3, with N the number of voxels.
    density_z: np.ndarray
        Array with the value of the partial derivative of 
        the density  with respecto to z in the domain.
        The shape is N x 3, with N the number of voxels.
    ext_density: float
        Exterior density value.

    Returns
    --------
    grad_alpha: np.ndarray
        Values of the gradient of alpha in each voxel.
        The shape is N x 3
    """
    
    num_voxels = dom.shape[0]
    grad_alpha = _np.zeros((num_voxels, 3), dtype=_np.complex128)
    grad_alpha[:, 0] = -density_x / (dom_density)**2
    grad_alpha[:, 1] = -density_y / (dom_density)**2
    grad_alpha[:, 2] = -density_z / (dom_density)**2

    return ext_density * grad_alpha



def surface_dif(rho_0, ext_density, int_density, surf):
    """
    Creates array with the difference of interior alpha and 
    exterior alpha in the limits of the domain.

    Parameters
    -----------
    rho_0: float
        Exterior density value.
    ext_density: np.ndarray
        Value of the density in the nodes of the exterior domain.
        The shape is N x 3, with N the number of voxels.
    int_density: np.ndarray
        Value of the density in the nodes of the interior domain.
        The shape is N x 3, with N the number of voxels.
    surf: np.ndarray
        Centers of the voxels of the surface in order.
        The shape is N x 3, with N the number of voxels.

    Returns
    --------
    diff_alpha: np.ndarray
        Diference of the value of alpha in each cell of the surface.
    """

    num_voxels = surf.shape[0]
    diff_alpha = _np.zeros(num_voxels, dtype=_np.complex128)
    int_aux = _np.zeros(num_voxels, dtype=_np.complex128)
    ext_aux = _np.zeros(num_voxels, dtype=_np.complex128)

    int_aux[:] = rho_0/int_density(surf[:, 0], surf[:, 1], surf[:, 2]) - 1
    ext_aux[:] = rho_0/ext_density(surf[:, 0], surf[:, 1], surf[:, 2]) - 1
    diff_alpha[:] = int_aux - ext_aux

    return diff_alpha
