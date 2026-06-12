"""
Module has the function to create the domian.
"""
import sys
import os
import numpy as _np

os.path.join(os.path.abspath(''), '../')

from vsie.model.problem import Subdomain, SurfDomain

def concentric_cubes(
    len_cube_int,
    len_cube_ext,
    vox_length,
    problem,
    ):
    """
    Function creates the grid of two concentric cubes
    centered in (0,0) and the grid of the surface for the interior cube.

    Parameters
    -----------
    len_cube_int: float
        The length of a side of the interior cube.
    len_cub_ext: float
        The length of a side of the exterior cube.
    vox_length: float
        The length of a side of the voxels.
    problem: Problem
        The problem object that contains the information of the problem.

    Returns
    --------
    ext_cube: Subdomain
        The exterior cube subdomain object.
    int_cube: Subdomain
        The interior cube subdomain object.  
    """

    # Calculate vox lenght
    # vox_length = wavelength/vox_per_wave

    # Inner discretization

    n_vox_cube_int = int(_np.ceil(len_cube_int / vox_length))
    if n_vox_cube_int % 2 == 0:
        n_vox_cube_int += 1
    start_int = -len_cube_int/2 + len_cube_int/(2*n_vox_cube_int)
    end_int = len_cube_int/2 - len_cube_int/(2*n_vox_cube_int)
    x_1_int = _np.linspace(start_int, end_int, num=n_vox_cube_int)
    y_1_int = _np.linspace(start_int, end_int, num=n_vox_cube_int)
    z_1_int = _np.linspace(start_int, end_int, num=n_vox_cube_int)

    dx_int = x_1_int[1] - x_1_int[0]

    # Outer discretization

    n_vox_cube_ext = int(_np.ceil((len_cube_ext - len_cube_int) / (2*vox_length)))
    start_ext = -len_cube_ext/2 + (len_cube_ext - len_cube_int)/(4*n_vox_cube_ext)
    end_ext = -len_cube_int/2 - (len_cube_ext - len_cube_int)/(4*n_vox_cube_ext)

    x_1_ext = _np.linspace(start_ext, end_ext, num=n_vox_cube_ext)
    y_1_ext = _np.linspace(start_ext, end_ext, num=n_vox_cube_ext)
    z_1_ext = _np.linspace(start_ext, end_ext, num=n_vox_cube_ext)

    dx_ext = x_1_ext[1] - x_1_ext[0]

    # Generate coordinates 

    x_1 = _np.concatenate((x_1_ext, x_1_int, - _np.flip(x_1_ext)))
    y_1 = _np.concatenate((y_1_ext, y_1_int, - _np.flip(y_1_ext)))
    z_1 = _np.concatenate((z_1_ext, z_1_int, - _np.flip(z_1_ext)))

    n_vox_cube = n_vox_cube_int + 2*n_vox_cube_ext

    r_1, r_2, r_3 = _np.meshgrid(x_1, y_1, z_1, indexing="ij")
    space = _np.stack((r_1.ravel(), r_2.ravel(), r_3.ravel()), axis=1)

    rad = len_cube_int / 2
    x, y, z = space[:, 0], space[:, 1], space[:, 2]
    idx_ext = _np.logical_or(
                            _np.abs(x) > rad, 
                            _np.logical_or(_np.abs(y) > rad, _np.abs(z) > rad)
                            )
    idx_int = _np.logical_not(idx_ext)

    # Obtain weights

    weights = _np.zeros(space.shape[0], dtype=_np.float64)
    idx_x1 = _np.abs(x) > len_cube_int/2
    idx_y1 = _np.abs(y) > len_cube_int/2
    idx_z1 = _np.abs(z) > len_cube_int/2
    weights[idx_x1] = dx_ext
    weights[_np.logical_not(idx_x1)] = dx_int
    weights[idx_y1] = weights[idx_y1] * dx_ext
    weights[_np.logical_not(idx_y1)] = weights[_np.logical_not(idx_y1)] * dx_int
    weights[idx_z1] = weights[idx_z1] * dx_ext
    weights[_np.logical_not(idx_z1)] = weights[_np.logical_not(idx_z1)] * dx_int

    # Getting a value for each voxel face. 
    # Has value 0 if in that direction stays in the same space. 
    # Has value -1 if that face contains the change from external space to internal space
    # Has value 1 if that face contains the change from internal space to external space

    int_array = _np.zeros(space.shape[0])
    int_array[idx_ext] = 1
    int_array = _np.reshape(int_array ,(n_vox_cube,n_vox_cube, n_vox_cube), order="C")
    int_array_x = _np.zeros((n_vox_cube, n_vox_cube, n_vox_cube))
    int_array_y = _np.zeros((n_vox_cube, n_vox_cube, n_vox_cube))
    int_array_z = _np.zeros((n_vox_cube, n_vox_cube, n_vox_cube))
    for i in range(n_vox_cube - 1):
        int_array_x[i, :, :] = int_array[i+1, :, :] - int_array[i, :, :]
        int_array_y[:, i, :] = int_array[:, i+1, :] - int_array[:, i, :]
        int_array_z[:, :, i] = int_array[:, :, i+1] - int_array[:, :, i]

    int_array_x = _np.reshape(int_array_x, space.shape[0])
    int_array_y = _np.reshape(int_array_y, space.shape[0])
    int_array_z = _np.reshape(int_array_z, space.shape[0])

    # Generates surface coordinates and normal vectors

    surf_grid = []
    normals = []
    for i in range(space.shape[0]):
        if int_array_x[i] != 0:
            surf_grid.append([len_cube_int/2 * int_array_x[i], space[i, 1], space[i, 2]])
            nor_vec = _np.array([1, 0, 0]) * int_array_x[i]
            normals.append(nor_vec)
        if int_array_y[i] != 0:
            surf_grid.append([space[i, 0], len_cube_int/2 * int_array_y[i], space[i, 2]])
            nor_vec = _np.array([0, 1, 0]) * int_array_y[i]
            normals.append(nor_vec)
        if int_array_z[i] != 0:
            surf_grid.append([space[i, 0], space[i, 1], len_cube_int/2 * int_array_z[i]])
            nor_vec = _np.array([0, 0, 1]) * int_array_z[i]
            normals.append(nor_vec)

    surf_grid = _np.array(surf_grid)
    normals = _np.array(normals)
    surf_weights = _np.ones(surf_grid.shape[0]) * dx_int**2

    ext_cube = Subdomain(
        problem,
        name="Exterior cube"
    )
    ext_cube.set_geometry(space[idx_ext], weights[idx_ext])

    int_cube = Subdomain(
        problem,
        name="Interior cube"
    )
    int_cube.set_geometry(space[idx_int], weights[idx_int])

    surface = SurfDomain(
        problem,
        name="Surface cube"
    )
    surface.set_geometry(surf_grid, surf_weights, normals, int_cube, ext_cube)

    problem.mesh = space
    problem.masks.append(idx_ext)
    problem.masks.append(idx_int)

    return ext_cube, int_cube, surface

def sphere(radius, len_cube, vox_length, problem):
    """
    Function creates the grid of a concentric sphere and cube
    centered in (0,0) and the grid of the surface for the sphere.

    Parameters
    -----------
    radius: float
        The length of the radius of the interior sphere.
    len_cube: float
        The length of a side of the exterior cube.
    vox_length: float
        The length of a side of the voxels.
    problem: Problem
        The problem object that contains the information of the problem.

    Returns
    --------
    space: np.ndarray
        A grid containing the center of the voxels of all space.
        The shape is (N1 + N2) x 3, with N1 the number of voxels in the exterior cube exclusively,
        and N2 the number of voxels in the interior sphere.
    idx_int: np.ndarray
        A boolean array with the information of which voxels are in the interior sphere.
        The shape is (N1 + N2) x 1, with N1 the number of voxels in the exterior cube exclusively,
        and N2 the number of voxels in the interior sphere.
    ext_space: np.ndarray
        A boolean array with the information of which voxels are in the exterior cube.
        The shape is (N1 + N2) x 1, with N1 the number of voxels in the exterior cube exclusively,
        and N2 the number of voxels in the interior sphere.
    surf_grid: np.ndarray
        A grid containing the center of the voxels of the surface of the interior sphere.
        The shape is Ns X 3, with Ns the number of voxels in the surface of the interior sphere.
    weights: np.ndarray
        An array containing the volume of each voxel. 
        The shape is (N1 + N2) x 3, with N1 the number of voxels in the exterior cube exclusively,
        and N2 the number of voxels in the interior sphere.
    surf_weights: np.ndarray
        An array containing the area of each voxel in the surface. 
        The shape is Ns X 3, with Ns the number of voxels in the surface of the interior sphere.
    normals: np.ndarray
        An array containing the direction of the normal vector in the center of the voxels of the
        surface of the interior sphere.
        The shape is Ns X 3, with Ns the number of voxels in the surface of the interior sphere.
    """

    # Generate coordinates 

    n_vox_cube = int(_np.ceil(len_cube / vox_length))
    if n_vox_cube % 2 == 0:
        n_vox_cube += 1
    start_ext = -len_cube/2 + len_cube/(2*n_vox_cube)
    end_ext = len_cube/2 - len_cube/(2*n_vox_cube)
    x_1 = _np.linspace(start_ext, end_ext, num=n_vox_cube)
    y_1 = _np.linspace(start_ext, end_ext, num=n_vox_cube)
    z_1 = _np.linspace(start_ext, end_ext, num=n_vox_cube)

    r_1, r_2, r_3 = _np.meshgrid(x_1, y_1, z_1, indexing="ij")
    space = _np.stack((r_1.ravel(), r_2.ravel(), r_3.ravel()), axis=1)
    vox_length = abs(space[0, 2] - space[1, 2])

    weights = _np.ones(space.shape[0]) * vox_length**3

    idx_int = space[:, 0]**2 + space[:, 1]**2 + space[:, 2]**2 <= radius**2
    idx_ext = space[:, 0]**2 + space[:, 1]**2 + space[:, 2]**2 > radius**2

    # Getting a value for each voxel face. 
    # Has value 0 if in that direction stays in the same space. 
    # Has value -1 if that face contains the change from external space to internal space
    # Has value 1 if that face contains the change from internal space to external space

    int_array = _np.zeros(space.shape[0])
    int_array[idx_ext] = 1
    int_array = _np.reshape(int_array ,(n_vox_cube,n_vox_cube, n_vox_cube), order="C")
    int_array_x = _np.zeros((n_vox_cube, n_vox_cube, n_vox_cube))
    int_array_y = _np.zeros((n_vox_cube, n_vox_cube, n_vox_cube))
    int_array_z = _np.zeros((n_vox_cube, n_vox_cube, n_vox_cube))
    for i in range(n_vox_cube - 1):
        int_array_x[i, :, :] = int_array[i+1, :, :] - int_array[i, :, :]
        int_array_y[:, i, :] = int_array[:, i+1, :] - int_array[:, i, :]
        int_array_z[:, :, i] = int_array[:, :, i+1] - int_array[:, :, i]

    int_array_x = _np.reshape(int_array_x, space.shape[0])
    int_array_y = _np.reshape(int_array_y, space.shape[0])
    int_array_z = _np.reshape(int_array_z, space.shape[0])

    surf_grid = []
    normals = []
    for i in range(space.shape[0]):
        if int_array_x[i] != 0:
            surf_grid.append([space[i, 0] + vox_length/2, space[i, 1], space[i, 2]])
            nor_vec = _np.array([1, 0, 0]) * int_array_x[i]
            normals.append(nor_vec)
        if int_array_y[i] != 0:
            surf_grid.append([space[i, 0], space[i, 1] + vox_length/2, space[i, 2]])
            nor_vec = _np.array([0, 1, 0]) * int_array_y[i]
            normals.append(nor_vec)
        if int_array_z[i] != 0:
            surf_grid.append([space[i, 0], space[i, 1], space[i, 2] + vox_length/2])
            nor_vec = _np.array([0, 0, 1]) * int_array_z[i]
            normals.append(nor_vec)

    surf_grid = _np.array(surf_grid)
    normals = _np.array(normals)
    surf_weights = _np.ones(surf_grid.shape[0]) * vox_length**2

    ext_cube = Subdomain(
        problem,
        name="Exterior cube"
    )
    ext_cube.set_geometry(space[idx_ext], weights[idx_ext])

    int_sphere = Subdomain(
        problem,
        name="Inner sphere"
    )
    int_sphere.set_geometry(space[idx_int], weights[idx_int])
    
    surface = SurfDomain(
        problem,
        name="Surface sphere"
    )
    surface.set_geometry(surf_grid, surf_weights, normals, int_sphere, ext_cube)

    problem.mesh = space
    problem.masks.append(idx_ext)
    problem.masks.append(idx_int)

    return ext_cube, int_sphere, surface