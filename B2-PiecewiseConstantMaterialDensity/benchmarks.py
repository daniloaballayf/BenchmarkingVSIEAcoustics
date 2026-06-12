import os
import sys

sys.path.append(os.path.join(os.path.abspath(''), '../'))

import numpy as np
from vsie.operators.cross_interaction import cross_single_layer, cross_double_layer
from vsie.operators.self_interaction import single_layer,  double_layer, mass_matrix
from vsie.operators.wave_op import incident_plane
from vsie.geometry.grid import concentric_cubes
from vsie.model.problem import Problem
from scipy.sparse.linalg import gmres

import bempp.api



def VSIE_solve(points_per_wavelength, parameters):
    l_cube_ext = parameters['SIZE_EXT']                  
    l_cube_int = parameters['SIZE_INT']                 
    wavespeed = parameters['WAVESPEED']                 
    frec = parameters['FREC']                           
    rho_0 = parameters['RHO_0']                                    

    problem = Problem(rho_0, wavespeed, frec)

    int_density = parameters['INTDENSITYFUN']
    ext_density = parameters['EXTDENSITYFUN']

    int_wavespeed = parameters['INTWAVESPEEDFUN']
    ext_wavespeed = parameters['EXTWAVESPEEDFUN']

    lambda_1 = int_wavespeed(0,0,0) / frec
    lambda_2 = ext_wavespeed(0,0,0) / frec

    lambda_imp = min(lambda_1, lambda_2)

    dx = lambda_imp / points_per_wavelength

    ext_space, int_space, surface = concentric_cubes(l_cube_int, l_cube_ext, dx, problem)

    ext_space.density = ext_density
    int_space.density = int_density
    ext_space.wavespeed = ext_wavespeed
    int_space.wavespeed = int_wavespeed

    problem.set_problem()

    mass_op_ext = mass_matrix(ext_space)
    mass_op_int = mass_matrix(int_space)
    mass_op_bdy = mass_matrix(surface)
    sl_op_ext = single_layer(ext_space)
    sl_op_int = single_layer(int_space)
    dl_op_bdy = double_layer(surface)
    sl_op_int_ext = cross_single_layer(int_space, ext_space)
    sl_op_ext_int = cross_single_layer(ext_space, int_space) 
    sl_op_bdy_ext = cross_single_layer(surface, ext_space)
    sl_op_bdy_int = cross_single_layer(surface, int_space)
    dl_op_ext_bdy = cross_double_layer(ext_space, surface)
    dl_op_int_bdy = cross_double_layer(int_space, surface)

    M = np.block([
        [mass_op_ext-sl_op_ext, dl_op_ext_bdy          , -sl_op_ext_int       ],
        [-sl_op_bdy_ext       , mass_op_bdy + dl_op_bdy, -sl_op_bdy_int       ],
        [-sl_op_int_ext       , dl_op_int_bdy          , mass_op_int-sl_op_int]
    ])

    direction = np.array((1., 0., 0.))
    direction /= np.linalg.norm(direction)
    uinc_ext = incident_plane(ext_space, direction)
    uinc_bdy = incident_plane(surface, direction)
    uinc_int = incident_plane(int_space, direction)

    uinc = np.block([uinc_ext, uinc_bdy, uinc_int])


    sol, info = gmres(M, uinc, rtol=1e-5)

    sol_ext = sol[:len(ext_space.mesh)]
    sol_bdy = sol[len(ext_space.mesh):len(ext_space.mesh)+len(surface.mesh)]
    sol_int = sol[-len(int_space.mesh):]

    Sol = np.zeros(problem.mesh.shape[0], dtype=np.complex128)
    Sol[problem.masks[0]] = sol_ext
    Sol[problem.masks[1]] = sol_int

    return Sol, problem
    

def bempp_solve(points_per_wavelength, parameters, points_eval):
    l_0 = parameters['LAMBDA_EXT']
    f = parameters['FREC']
    c = parameters['WAVESPEED']
    c1 = parameters['WAVESPEED_1']
    c2 = parameters['WAVESPEED_2']
    k0 = 2 * np.pi / l_0
    k1 = 2 * np.pi * f / c1
    k2 = 2 * np.pi * f / c2

    rho0 = parameters['RHO_0']
    rho1 = parameters['RHO_1']
    rho2 = parameters['RHO_2']
    l_cube_1 = parameters['SIZE_EXT']/2
    l_cube_2 = parameters['SIZE_INT']/2
    k_max = max(k0, k1, k2)
    h = 2 * np.pi/(points_per_wavelength * k_max)
    grid1 = bempp.api.shapes.cube(2*l_cube_1, (-l_cube_1, -l_cube_1, -l_cube_1) , h)
    grid2 = bempp.api.shapes.cube(2*l_cube_2, (-l_cube_2, -l_cube_2, -l_cube_2) , h)

    l1 = 2 * np.pi / k1
    l2 = 2 * np.pi / k2

    @bempp.api.complex_callable
    def dirichlet_fun(x, n, domain_index, result):
        result[0] = np.exp(1j * k0 * x[0])

    @bempp.api.complex_callable
    def neumann_fun(x, n, domain_index, result):
        result[0] = 1j * k0 * n[0] * np.exp(1j *k0 * x[0])

    space1 = bempp.api.function_space(grid1, "P", 1)
    space2 = bempp.api.function_space(grid2, 'P', 1)

    sl_11_0 = bempp.api.operators.boundary.helmholtz.single_layer(space1, space1, space1, k0)
    sl_11_1 = bempp.api.operators.boundary.helmholtz.single_layer(space1, space1, space1, k1)
    sl_12_1 = bempp.api.operators.boundary.helmholtz.single_layer(space1, space2, space2, k1)
    sl_21_1 = bempp.api.operators.boundary.helmholtz.single_layer(space2, space1, space1, k1)
    sl_22_1 = bempp.api.operators.boundary.helmholtz.single_layer(space2, space2, space2, k1)
    sl_22_2 = bempp.api.operators.boundary.helmholtz.single_layer(space2, space2, space2, k2)

    dl_11_0 = bempp.api.operators.boundary.helmholtz.double_layer(space1, space1, space1, k0)
    dl_11_1 = bempp.api.operators.boundary.helmholtz.double_layer(space1, space1, space1, k1)
    dl_12_1 = bempp.api.operators.boundary.helmholtz.double_layer(space1, space2, space2, k1)
    dl_21_1 = bempp.api.operators.boundary.helmholtz.double_layer(space2, space1, space1, k1)
    dl_22_1 = bempp.api.operators.boundary.helmholtz.double_layer(space2, space2, space2, k1)
    dl_22_2 = bempp.api.operators.boundary.helmholtz.double_layer(space2, space2, space2, k2)

    ad_11_0 = bempp.api.operators.boundary.helmholtz.adjoint_double_layer(space1, space1, space1, k0)
    ad_11_1 = bempp.api.operators.boundary.helmholtz.adjoint_double_layer(space1, space1, space1, k1)
    ad_12_1 = bempp.api.operators.boundary.helmholtz.adjoint_double_layer(space1, space2, space2, k1)
    ad_21_1 = bempp.api.operators.boundary.helmholtz.adjoint_double_layer(space2, space1, space1, k1)
    ad_22_1 = bempp.api.operators.boundary.helmholtz.adjoint_double_layer(space2, space2, space2, k1)
    ad_22_2 = bempp.api.operators.boundary.helmholtz.adjoint_double_layer(space2, space2, space2, k2)

    hs_11_0 = bempp.api.operators.boundary.helmholtz.hypersingular(space1, space1, space1, k0)
    hs_11_1 = bempp.api.operators.boundary.helmholtz.hypersingular(space1, space1, space1, k1)
    hs_12_1 = bempp.api.operators.boundary.helmholtz.hypersingular(space1, space2, space2, k1)
    hs_21_1 = bempp.api.operators.boundary.helmholtz.hypersingular(space2, space1, space1, k1)
    hs_22_1 = bempp.api.operators.boundary.helmholtz.hypersingular(space2, space2, space2, k1)
    hs_22_2 = bempp.api.operators.boundary.helmholtz.hypersingular(space2, space2, space2, k2)

    block_op = bempp.api.BlockedOperator(4, 4)
    block_op[0, 0] = -dl_11_0 - dl_11_1
    block_op[0, 1] = sl_11_0 + (rho1/rho0) * sl_11_1
    block_op[1, 0] = hs_11_0 + (rho0/rho1) * hs_11_1
    block_op[1, 1] = ad_11_0 + ad_11_1
    block_op[2, 2] = -dl_22_1 - dl_22_2
    block_op[2, 3] = sl_22_1 + (rho2/rho1) * sl_22_2
    block_op[3, 2] = hs_22_1 + (rho1/rho2) * hs_22_2
    block_op[3, 3] = ad_22_1 + ad_22_2
    block_op[0, 2] = dl_21_1
    block_op[0, 3] = -sl_21_1
    block_op[1, 2] = -(rho0/rho1) * hs_21_1
    block_op[1, 3] = -(rho0/rho1) * ad_21_1
    block_op[2, 0] = dl_12_1
    block_op[2, 1] = -(rho1/rho0) * sl_12_1
    block_op[3, 0] = - hs_12_1
    block_op[3, 1] = -(rho1/rho0) * ad_12_1

    dirichlet_grid_fun = bempp.api.GridFunction(space1, fun=dirichlet_fun)
    neumann_grid_fun = bempp.api.GridFunction(space1, fun=neumann_fun)

    rhs = np.concatenate([dirichlet_grid_fun.coefficients,
                        neumann_grid_fun.coefficients,np.zeros(space2.global_dof_count * 2)])

    block_sf = block_op.strong_form()


    x, info = gmres(block_sf, rhs, rtol=1e-5)

    split = np.cumsum([space1.global_dof_count, space1.global_dof_count, space2.global_dof_count])
    sol = np.split(x, split)
    surface_potential_dirichlet1 = bempp.api.GridFunction(
        space1, coefficients=sol[0])
    surface_potential_neumann1 = bempp.api.GridFunction(
        space1, coefficients=sol[1])
    surface_potential_dirichlet2 = bempp.api.GridFunction(
        space2, coefficients=sol[2])
    surface_potential_neumann2 = bempp.api.GridFunction(
        space2, coefficients=sol[3])
    

    points = points_eval
    u_evaluated = np.zeros(points.shape[1], dtype=np.complex128)

    x, y, z = points

    idx_2 = np.logical_and(np.logical_and(np.abs(x) <= l_cube_2, np.abs(y) <= l_cube_2), np.abs(z) <= l_cube_2)
    idx_1 = np.logical_or(np.logical_and(np.logical_and(l_cube_2 < np.abs(x), np.abs(x) <= l_cube_1), np.abs(y)<=l_cube_1),np.logical_and(np.logical_and(l_cube_2 < np.abs(y) ,np.abs(y) <= l_cube_1), np.abs(x)<=l_cube_1))
    idx_0 = np.logical_or(np.abs(x) > l_cube_1, np.abs(y) > l_cube_1)

    points_1 = points[:, idx_1]
    points_2 = points[:, idx_2]

    slp_pot_1_1 = bempp.api.operators.potential.helmholtz.single_layer(
        space1, points_1, k1)
    slp_pot_2_1 = bempp.api.operators.potential.helmholtz.single_layer(
        space2, points_1, k1)
    slp_pot_2_2 = bempp.api.operators.potential.helmholtz.single_layer(
        space2, points_2, k2)

    dlp_pot_1_1 = bempp.api.operators.potential.helmholtz.double_layer(
        space1, points_1, k1)
    dlp_pot_2_1 = bempp.api.operators.potential.helmholtz.double_layer(
        space2, points_1, k1)
    dlp_pot_2_2 = bempp.api.operators.potential.helmholtz.double_layer(
        space2, points_2, k2)

    total_field_2 = (slp_pot_2_2 * (rho2/rho1) * surface_potential_neumann2                
                    - dlp_pot_2_2 * surface_potential_dirichlet2).ravel()
    total_field_1 = (slp_pot_1_1 * (rho1/rho0) * surface_potential_neumann1               
                    - dlp_pot_1_1 * surface_potential_dirichlet1).ravel()
    total_field_1 += (-slp_pot_2_1 * surface_potential_neumann2
                    + dlp_pot_2_1 * surface_potential_dirichlet2).ravel()             



    total_field = np.zeros(points.shape[1], dtype='complex128')
    total_field[idx_1] = total_field_1
    total_field[idx_2] = total_field_2

    return total_field
