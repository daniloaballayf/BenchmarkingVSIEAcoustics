"""
Module has the function to create arrays with values of density and speed wave functions.
"""

import numpy as _np

def physical_functions(wavenumber, ext_density, dom_wave, dom_density, dom):
    """
    Creates array with values of alpha, beta in
    the domain.

    Parameters
    -----------
    wavenumber: float, complex
        Exterior wave number.
    ext_density: float
        Exterior density value.
    dom_wave: function
        Python function that determines the value of the wave number in 
        the domain. Recives as parameters the values of x, y and z.
    dom_density: function
        Python function that determines the value of the density in the
        domain. Recives as parameters the values of x, y and z.
    dom: np.ndarray
        Centers of the voxels of the space in order.
        The shape is N x 3, with N the number of voxels.

    Returns
    --------
    alpha: np.ndarray
        Values of alpha in each voxel of exterior cube.
        The shape is N x 1, with N1 number of voxels in the domain.
    beta: np.ndarray
        Values of beta in each voxel of exterior cube.
        The shape is N x 1, with N1 number of voxels in the domain.
    """
    num_voxels = dom.shape[0]
    alpha = _np.zeros(num_voxels, dtype=_np.complex128)
    beta = _np.zeros(num_voxels, dtype=_np.complex128)

    alpha[:] = ext_density/dom_density(*dom.T) - 1
    beta[:] = ext_density*dom_wave(*dom.T)**2\
                /dom_density(*dom.T) - wavenumber**2
    
    return alpha, beta

def discont_physical_functions(
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
    intdom_wave: function
        Python function that determines the value of the wave number in 
        the interior domain. Recives as parameters the values of x, y and z.
    extdom_wave: function
        Python function that determines the value of the wave number in 
        the exterior domain. Recives as parameters the values of x, y and z.
    intdom_density: function
        Python function that determines the value of the density in the
        interior domain. Recives as parameters the values of x, y and z.
    extdom_density: function
        Python function that determines the value of the density in the
        interior domain. Recives as parameters the values of x, y and z.
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
        The shape is N x 1, with N1 number of voxels in the domain.
    beta: np.ndarray
        Values of beta in each voxel of exterior cube.
        The shape is N x 1, with N1 number of voxels in the domain.
    """
    num_voxels = space.shape[0]
    alpha = _np.zeros(num_voxels, dtype=_np.complex128)
    beta = _np.zeros(num_voxels, dtype=_np.complex128)
    int_space = space[idx_int]
    ext_space = space[idx_ext]
    
    alpha[idx_int] = ext_density/intdom_density(*int_space.T)
    alpha[idx_int] +=  - 1
    alpha[idx_ext] = ext_density/extdom_density(*ext_space.T)
    alpha[idx_ext] +=  - 1

    beta[idx_int] = ext_density*intdom_wave(*int_space.T)**2\
                /intdom_density(*int_space.T)\
                - wavenumber**2
    print(extdom_density(ext_space[0, 0], ext_space[0, 1], ext_space[0, 2]))
    print(wavenumber)
    print(ext_density)
    print(extdom_density(ext_space[0, 0], ext_space[0, 1], ext_space[0, 2]))
    beta[idx_ext] = ext_density*extdom_wave(*ext_space.T)**2\
                /extdom_density(*ext_space.T)\
                - wavenumber**2
    
    return alpha, beta


def anal_grad(dom, dom_density, density_x, density_y, density_z, ext_density):
    """
    Creates array with the values of the gradient of alpha in the domain. 
    
    Parameters
    -----------
    dom: np.ndarray
        Centers of the voxels of the space in order.
        The shape is N x 3, with N the number of voxels.
    dom_density: function
        Python function that determines the value of the density in the
        domain. Recives as parameters the values of x, y and z.
    density_x: function
        Python function that determines the value of the partial derivative 
        of the density  with respecto to x in the domain.
        Recives as parameters the values of x, y and z.
    density_y: function
        Python function that determines the value of the partial derivative 
        of the density  with respecto to y in the domain.
        Recives as parameters the values of x, y and z.
    density_z: function
        Python function that determines the value of the partial derivative 
        of the density  with respecto to z in the domain.
        Recives as parameters the values of x, y and z.

    Returns
    --------
    grad_alpha: np.ndarray
        Values of the gradient of alpha in each voxel.
        The shape is N x 3
    """
    
    num_voxels = dom.shape[0]
    grad_alpha = _np.zeros((num_voxels, 3), dtype=_np.complex128)
    vox_density = dom_density(*dom.T)
    grad_alpha[:, 0] = -density_x(*dom.T) / (vox_density)**2
    grad_alpha[:, 1] = -density_y(*dom.T) / (vox_density)**2
    grad_alpha[:, 2] = -density_z(*dom.T) / (vox_density)**2

    return ext_density*grad_alpha


def surface_dif(rho_0, ext_density, int_density, surf):
    """
    Creates array with the difference of interior alpha and 
    exterior alpha in the limits of the domain.

    Parameters
    -----------
    rho_0: float
        Exterior density value.
    ext_density: function
        Python function that determines the value of the density in the
        exterior domain. Recives as parameters the values of x, y and z.
    int_density: function
        Python function that determines the value of the density in the
        interior domain. Recives as parameters the values of x, y and z.
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

    int_aux[:] = rho_0/int_density(*surf.T) - 1
    ext_aux[:] = rho_0/ext_density(*surf.T) - 1
    diff_alpha[:] = int_aux - ext_aux

    return diff_alpha

def max_wavenumber(dom_lims, wavefunction):
    '''
    Seeks an aproximation for the maximum wavenumber in the domain.

    Parameters
    -----------
    dom_lims: np.ndarray
        The limits of the domain for x, y and z. 
        The shape is 3 x 2.
    wavefunction: function
        Python function that determines the value of the wave number in 
        the domain. Recives as parameters the values of x, y and z.

    Returns
    --------
    k_max: float
        The aproximated value of the maximum of the wavenumber in the domain.
    '''
    x_lim, y_lim, z_lim = dom_lims
    x = _np.linspace(x_lim[0], x_lim[1], 100)
    y = _np.linspace(y_lim[0], y_lim[1], 100)
    z = _np.linspace(z_lim[0], z_lim[1], 100)
    xx, yy, zz = _np.meshgrid(x, y, z)
    wave_eval = wavefunction(xx, yy, zz)
    k_max = _np.max(wave_eval)
    return k_max
