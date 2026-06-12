import os
import sys

sys.path.append(os.path.join(os.path.abspath(''), '../'))

import numpy as np
from vsie.operators.cross_interaction import cross_single_layer, cross_double_layer, cross_ad_double_layer
from vsie.operators.self_interaction import single_layer, double_layer, ad_double_layer, mass_matrix
from vsie.operators.wave_op import incident_plane
from vsie.geometry.grid import concentric_cubes
from vsie.model.problem import Problem, Subdomain
from scipy.sparse.linalg import gmres

import dolfinx
from dolfinx.fem import FunctionSpace, Function, Expression
from dolfinx.mesh import create_unit_cube
import dolfinx.geometry
import ufl
from mpi4py import MPI
import bempp.api
from bempp.api.external import fenicsx
from bempp.api.assembly.blocked_operator import BlockedDiscreteOperator
from bempp.api.assembly.discrete_boundary_operator import InverseSparseDiscreteBoundaryOperator
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg.interface import LinearOperator as LO

import logging
from dolfinx import log


logging.getLogger("bempp").setLevel(logging.WARNING)
logging.getLogger("numba").setLevel(logging.WARNING)

log.set_log_level(log.LogLevel.WARNING)


def VSIE_solve(points_per_wavelength, parameters, err=False):
    l_cube_ext = parameters["SIZE_EXT"]
    l_cube_int = parameters["SIZE_INT"]
    wavespeed = parameters["WAVESPEED"]
    frec = parameters["FREC"]
    rho_0 = parameters["RHO_0"]

    if err:
        frec = parameters["FREC_ERR"]

    problem = Problem(rho_0, wavespeed, frec)

    int_density = parameters["INTDENSITYFUN"]
    ext_density = parameters["EXTDENSITYFUN"]

    int_wavespeed = parameters["INTWAVESPEEDFUN"]
    ext_wavespeed = parameters["EXTWAVESPEEDFUN"]

    int_density_x = parameters["DXINTDENSITYFUN"]
    int_density_y = parameters["DYINTDENSITYFUN"]
    int_density_z = parameters["DZINTDENSITYFUN"]

    ext_density_x = parameters["DXEXTDENSITYFUN"]
    ext_density_y = parameters["DYEXTDENSITYFUN"]
    ext_density_z = parameters["DZEXTDENSITYFUN"]

    lambda_1 = int_wavespeed(0, 0, 0) / frec

    dx = lambda_1 / points_per_wavelength

    ext_space, int_space, surface = concentric_cubes(l_cube_int, l_cube_ext, dx, problem)

    ext_space.density = ext_density
    int_space.density = int_density
    ext_space.wavespeed = ext_wavespeed
    int_space.wavespeed = int_wavespeed
    ext_space.density_gradient = [ext_density_x, ext_density_y, ext_density_z]
    int_space.density_gradient = [int_density_x, int_density_y, int_density_z]

    problem.set_problem() 

    mass_op_ext = mass_matrix(ext_space)
    mass_op_int = mass_matrix(int_space)
    mass_op_bdy = mass_matrix(surface)
    sl_op_ext = single_layer(ext_space)
    sl_op_int = single_layer(int_space)
    sl_op_bdy_ext = cross_single_layer(surface, ext_space)
    sl_op_bdy_int = cross_single_layer(surface, int_space)
    sl_op_ext_int = cross_single_layer(ext_space, int_space)
    sl_op_int_ext = cross_single_layer(int_space, ext_space)
    dl_op_ext_bdy = cross_double_layer(ext_space, surface)
    dl_op_int_bdy = cross_double_layer(int_space, surface)
    dl_op_bdy = double_layer(surface)
    ad_dl_op_ext = ad_double_layer(ext_space)
    ad_dl_op_int = ad_double_layer(int_space)
    ad_dl_op_bdy_ext = cross_ad_double_layer(surface, ext_space)
    ad_dl_op_bdy_int = cross_ad_double_layer(surface, int_space)
    ad_dl_op_ext_int = cross_ad_double_layer(ext_space, int_space) 
    ad_dl_op_int_ext = cross_ad_double_layer(int_space, ext_space)

    M = np.block([
    [mass_op_ext - sl_op_ext + ad_dl_op_ext ,  dl_op_ext_bdy        ,  -sl_op_ext_int + ad_dl_op_ext_int     ],
    [-sl_op_bdy_ext + ad_dl_op_bdy_ext      ,  mass_op_bdy+dl_op_bdy,  -sl_op_bdy_int + ad_dl_op_bdy_int     ],
    [-sl_op_int_ext + ad_dl_op_int_ext      ,  dl_op_int_bdy        ,  mass_op_int - sl_op_int + ad_dl_op_int]
    ])

    direction = np.array((1., 0., 0.))
    direction = direction/np.linalg.norm(direction)
    uinc_ext = incident_plane(ext_space, direction)
    uinc_int = incident_plane(int_space, direction)
    uinc_bdy = incident_plane(surface, direction)

    uinc = np.block([uinc_ext, uinc_bdy, uinc_int])

    sol, info = gmres(M, uinc, rtol=1e-5)

    sol_ext = sol[:len(ext_space.mesh)]
    sol_bdy = sol[len(ext_space.mesh):len(ext_space.mesh)+len(surface.mesh)]
    sol_int = sol[-len(int_space.mesh):]

    Sol = np.zeros(problem.mesh.shape[0], dtype=np.complex128)
    Sol[problem.masks[0]] = sol_ext
    Sol[problem.masks[1]] = sol_int

    return Sol, problem

def fembem_solve(points_per_wavelength, parameters, points_eval, err=False):
    l_cube_ext = parameters["SIZE_EXT"]
    l_cube_int = parameters["SIZE_INT"]
    R = l_cube_ext/2
    r = l_cube_int/2
    la_0 = parameters["LAMBDA_EXT"]
    frec = parameters["FREC"]
    wavespeed = parameters["WAVESPEED"]
    if err:
        la_0 = parameters["LAMBDA_ERR"]
        frec = parameters["FREC_ERR"]


    k_0 = 2*np.pi/la_0
    k_1 = 2*np.pi/(parameters["EXTWAVESPEEDFUN"](0,0,0) / frec)
    k_2 = 2*np.pi/(parameters["INTWAVESPEEDFUN"](0,0,0) / frec)

    la_1 = 2*np.pi/k_1
    la_2 = 2*np.pi/k_2

    rho_0 = parameters["RHO_0"]
    rho_1 = parameters["RHO_1"]
    rho_2 = parameters["RHO_2"]

    d = np.array([1., 0., 0.], dtype=np.float64)

    h = la_1 / points_per_wavelength
    n_t = int(2*r / (la_2 / (2*points_per_wavelength))) 
    Om2_grid = dolfinx.mesh.create_box(MPI.COMM_WORLD, [np.array([-r, -r, -r]), 
                                                    np.array([r, r, r])],
                    [n_t,n_t,n_t], cell_type=dolfinx.mesh.CellType.tetrahedron)

    Om1_grid = bempp.api.shapes.cube(2*R, (-R, -R, -R) , h)

    from bempp.api.external import fenicsx

    fenics_space = FunctionSpace(Om2_grid, ("CG", 1))
    trace_space, trace_matrix = \
        fenicsx.fenics_to_bempp_trace_data(fenics_space)

    bempp_space = bempp.api.function_space(Om1_grid, 'P', 1)

    fem_size = fenics_space.dofmap.index_map.size_global
    bem1_size = trace_space.global_dof_count
    bem2_size = bempp_space.global_dof_count

    print("FEM dofs: {0}".format(fem_size))
    print("Interior BEM dofs: {0}".format(bem1_size))
    print("Exterior BEM dofs: {0}".format(bem2_size))

    id_op = bempp.api.operators.boundary.sparse.identity(
        trace_space, trace_space, trace_space)
    mass = bempp.api.operators.boundary.sparse.identity(
        trace_space, trace_space, trace_space)

    zero_op = bempp.api.ZeroBoundaryOperator(bempp_space, trace_space, trace_space)

    sl_22_1 = bempp.api.operators.boundary.helmholtz.single_layer(trace_space, trace_space, trace_space, k_1)
    sl_12_1 = bempp.api.operators.boundary.helmholtz.single_layer(trace_space, bempp_space, bempp_space, k_1)
    sl_21_1 = bempp.api.operators.boundary.helmholtz.single_layer(bempp_space, trace_space, trace_space, k_1)
    sl_11_1 = bempp.api.operators.boundary.helmholtz.single_layer(bempp_space, bempp_space, bempp_space, k_1)
    sl_11_0 = bempp.api.operators.boundary.helmholtz.single_layer(bempp_space, bempp_space, bempp_space, k_0)

    dl_22_1 = bempp.api.operators.boundary.helmholtz.double_layer(trace_space, trace_space, trace_space, k_1)
    dl_12_1 = bempp.api.operators.boundary.helmholtz.double_layer(trace_space, bempp_space, bempp_space, k_1)
    dl_21_1 = bempp.api.operators.boundary.helmholtz.double_layer(bempp_space, trace_space, trace_space, k_1)
    dl_11_1 = bempp.api.operators.boundary.helmholtz.double_layer(bempp_space, bempp_space, bempp_space, k_1)
    dl_11_0 = bempp.api.operators.boundary.helmholtz.double_layer(bempp_space, bempp_space, bempp_space, k_0)

    ad_12_1 = bempp.api.operators.boundary.helmholtz.adjoint_double_layer(trace_space, bempp_space, bempp_space, k_1)
    ad_11_1 = bempp.api.operators.boundary.helmholtz.adjoint_double_layer(bempp_space, bempp_space, bempp_space, k_1)
    ad_11_0 = bempp.api.operators.boundary.helmholtz.adjoint_double_layer(bempp_space, bempp_space, bempp_space, k_0)

    hs_12_1 = bempp.api.operators.boundary.helmholtz.hypersingular(trace_space, bempp_space, bempp_space, k_1)
    hs_11_1 = bempp.api.operators.boundary.helmholtz.hypersingular(bempp_space, bempp_space, bempp_space, k_1)
    hs_11_0 = bempp.api.operators.boundary.helmholtz.hypersingular(bempp_space, bempp_space, bempp_space, k_0)

    u = ufl.TrialFunction(fenics_space)
    v = ufl.TestFunction(fenics_space)

    @bempp.api.complex_callable
    def u_inc(x, n, domain_index, result):
        result[0] = np.exp(1j * k_0 * np.dot(x, d))
        
    @bempp.api.complex_callable
    def u_inc_n(x, n, domain_index, result):
        result[0] = 1j * k_0 * np.dot(d, n) * np.exp(1j * k_0 * np.dot(x, d))
        
    u_inc_grid2 = bempp.api.GridFunction(bempp_space, fun=u_inc)
    u_inc_n_grid2 = bempp.api.GridFunction(bempp_space, fun=u_inc_n)

    rhs_fem = np.zeros(fem_size)
    rhs_bem = np.zeros(bem1_size)
    rhs_bem_d = u_inc_grid2.projections(bempp_space)
    rhs_bem_n = u_inc_n_grid2.projections(bempp_space)
    rhs = np.concatenate([rhs_bem_n, rhs_bem_d, rhs_fem, rhs_bem])

    from bempp.api.assembly.blocked_operator import BlockedDiscreteOperator
    from scipy.sparse.linalg import LinearOperator
    blocks = [[None,None,None,None],[None,None,None,None],[None,None,None,None],[None,None,None,None]]

    trace_op = LinearOperator(trace_matrix.shape, matvec=lambda x:trace_matrix @ x, rmatvec=lambda x: trace_matrix.T @ x)


    normal_vec = ufl.FacetNormal(Om2_grid)
    x = ufl.SpatialCoordinate(Om2_grid)
    gl = r
    rho = 800 * ufl.sin((x[0]-gl)*ufl.pi/gl)*  ufl.sin((x[1]-gl)*ufl.pi/gl) * ufl.sin((x[2]-gl)*0.5*ufl.pi/gl) + rho_2

    k_int = k_2

    meta={"quadrature_degree":5}
    dx=ufl.Measure('dx',domain=Om2_grid,metadata=meta)

    At = (ufl.inner((1/rho)*ufl.grad(u), ufl.grad(rho*v)) - k_int**2 * ufl.inner(u, v))* dx

    A = fenicsx.FenicsOperator(At)

    blocks[1][0] = (-dl_11_0 - dl_11_1).weak_form()
    blocks[1][1] = (sl_11_0 + (rho_1/rho_0)*sl_11_1).weak_form()
    blocks[1][2] =  dl_12_1.weak_form() * trace_op
    blocks[1][3] = -sl_12_1.weak_form()
    blocks[0][0] = (hs_11_0 + (rho_0/rho_1)*hs_11_1).weak_form()
    blocks[0][1] = (ad_11_0 + ad_11_1).weak_form()
    blocks[0][2] = -(rho_0/rho_1)*hs_12_1.weak_form() * trace_op
    blocks[0][3] = -(rho_0/rho_1) * ad_12_1.weak_form() 
    blocks[3][0] = -dl_21_1.weak_form()
    blocks[3][1] = ((rho_1/rho_0)*sl_21_1).weak_form()
    blocks[3][2] = (dl_22_1 - 0.5*id_op).weak_form() * trace_op
    blocks[3][3] = -sl_22_1.weak_form()
    blocks[2][0] = trace_op.T * zero_op.weak_form()
    blocks[2][1] = trace_op.T * zero_op.weak_form()
    blocks[2][2] = A.weak_form()
    blocks[2][3] = -(rho_2/rho_1) * trace_op.T * id_op.weak_form() 

    blocked = BlockedDiscreteOperator(np.array(blocks))
    from bempp.api.assembly.discrete_boundary_operator import InverseSparseDiscreteBoundaryOperator
    from scipy.sparse.linalg import LinearOperator

    P1 = InverseSparseDiscreteBoundaryOperator(
        blocked[2,2].to_sparse().tocsc())

    P2 = InverseSparseDiscreteBoundaryOperator(
        bempp.api.operators.boundary.sparse.identity(
            trace_space, trace_space, trace_space).weak_form())

    P3 = InverseSparseDiscreteBoundaryOperator(
        bempp.api.operators.boundary.sparse.identity(
            bempp_space, bempp_space, bempp_space).weak_form())


    def apply_prec(x):
        """Apply the block diagonal preconditioner"""
        m1 = P1.shape[0]
        m2 = P2.shape[0]
        m3 = P3.shape[0]
        n1 = P1.shape[1]
        n2 = P2.shape[1]
        n3 = P3.shape[1]
        
        res1 = P3.dot(x[:n3])
        res2 = P3.dot(x[n3:n3 + n3])
        res3 = P1.dot(x[n3 + n3: n3 + n3 + n1])
        res4 = P2.dot(x[n3 + n3 + n1: ])
        return np.concatenate([res1, res2, res3, res4])

    p_shape = (P1.shape[0] + P2.shape[0] + 2*P3.shape[0], P1.shape[1] + P2.shape[1]+ 2*P3.shape[1])
    P = LinearOperator(p_shape, apply_prec, dtype=np.dtype('complex128'))


    from scipy.sparse.linalg import gmres
    sol, info = gmres(blocked, rhs, M = P, restart=5000, maxiter=20000)

    split = np.cumsum([bem2_size, bem2_size, fem_size])
    soln = np.split(sol, split)

    sol_fem = soln[2]
    soln_bem = soln[3]
    sold_bem2 = soln[0]
    soln_bem2 = soln[1]

    u_sol = Function(fenics_space)
    u_sol.vector[:] = np.ascontiguousarray(np.real(sol_fem))

    dirichlet_data = trace_matrix * sol_fem
    dirichlet_fun_2 = bempp.api.GridFunction(trace_space, coefficients=dirichlet_data)

    dirichlet_fun_1 = bempp.api.GridFunction(bempp_space, coefficients=sold_bem2) 

    neumann_fun_2 = bempp.api.GridFunction(trace_space, coefficients=soln_bem)

    neumann_fun_1 = bempp.api.GridFunction(bempp_space, coefficients=soln_bem2)

    points = points_eval
    plot_me = np.zeros(points.shape[1], dtype=np.complex128)

    x,y,z = points
    bem_x = np.logical_not((np.abs(x)<r) * (np.abs(y)<r) * (np.abs(z)<r))

    slp_pot_2= bempp.api.operators.potential.helmholtz.single_layer(
        trace_space, points[:, bem_x], k_1)
    dlp_pot_2= bempp.api.operators.potential.helmholtz.double_layer(
        trace_space, points[:, bem_x], k_1)
    slp_pot_1= bempp.api.operators.potential.helmholtz.single_layer(
        bempp_space, points[:, bem_x], k_1)
    dlp_pot_1= bempp.api.operators.potential.helmholtz.double_layer(
        bempp_space, points[:, bem_x], k_1)

    plot_me[bem_x] += (-dlp_pot_1.evaluate(dirichlet_fun_1) + dlp_pot_2.evaluate(dirichlet_fun_2)).flat
    plot_me[bem_x] += ((rho_1/rho_0)*slp_pot_1.evaluate(neumann_fun_1) - slp_pot_2.evaluate(neumann_fun_2)).flat

    fem_points = points[:, np.logical_not(bem_x)].transpose()
    try:
        tree = dolfinx.geometry.BoundingBoxTree(Om2_grid, 3)
    except TypeError:
        tree = dolfinx.geometry.bb_tree(Om2_grid, 3)
    midpoint_tree = dolfinx.geometry.create_midpoint_tree(
        Om2_grid, 3, list(range(Om2_grid.topology.connectivity(3, 0).num_nodes))
    )
    entities = []
    for point in fem_points:
        entities.append(dolfinx.geometry.compute_closest_entity(tree, midpoint_tree, Om2_grid, point)[0])
    fem_val = u_sol.eval(fem_points, entities)

    plot_me[np.logical_not(bem_x)] += fem_val.T[0]

    return plot_me

    

