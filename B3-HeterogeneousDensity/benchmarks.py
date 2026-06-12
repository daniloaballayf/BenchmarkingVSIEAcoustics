import os
import sys

sys.path.append(os.path.join(os.path.abspath(''), '../'))

import numpy as np
from vsie.operators.self_interaction import single_layer,  ad_double_layer, mass_matrix
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

def VSIE_solve(points_per_wavelength, parameters):
    l_cube_ext = parameters["SIZE"]
    wavespeed = parameters["WAVESPEED"]
    frec = parameters["FREC"]
    rho_0 = parameters["RHO_0"]

    problem = Problem(rho_0, wavespeed, frec)

    density = parameters["INTDENSITYFUN"]
    int_wavespeed = parameters["WAVESPEEDFUN"]

    density_x = parameters["DXINTDENSITYFUN"]
    density_y = parameters["DYINTDENSITYFUN"]
    density_z = parameters["DZINTDENSITYFUN"]

    lambda_1 = int_wavespeed(0,0,0) / frec
    dx = lambda_1 / points_per_wavelength

    ext_space, int_space, surface = concentric_cubes(l_cube_ext/2, l_cube_ext, dx, problem)
    ext_space.wavespeed = int_wavespeed
    int_space.wavespeed = int_wavespeed
    ext_space.density = density
    int_space.density = density
    ext_space.density_gradient = [density_x, density_y, density_z]
    int_space.density_gradient = [density_x, density_y, density_z]

    problem.set_problem()

    domain = Subdomain(problem)
    weights = np.zeros(problem.mesh.shape[0], dtype=np.complex128)
    alpha = np.zeros(problem.mesh.shape[0], dtype=np.complex128)
    beta = np.zeros(problem.mesh.shape[0], dtype=np.complex128)
    alpha_gradient = np.zeros((problem.mesh.shape[0], 3), dtype=np.complex128)
    weights[problem.masks[0]] = ext_space.volumes
    weights[problem.masks[1]] = int_space.volumes
    alpha[problem.masks[0]] = ext_space.alpha
    alpha[problem.masks[1]] = int_space.alpha
    beta[problem.masks[0]] = ext_space.beta
    beta[problem.masks[1]] = int_space.beta
    alpha_gradient[problem.masks[0], :] = ext_space.alpha_gradient
    alpha_gradient[problem.masks[1], :] = int_space.alpha_gradient

    domain.set_geometry(problem.mesh, weights)
    domain.alpha = alpha
    domain.beta = beta
    domain.alpha_gradient = alpha_gradient

    mass_op = mass_matrix(domain)
    sl_op = single_layer(domain)
    ad_dl_op = ad_double_layer(domain)

    M = mass_op - sl_op + ad_dl_op


    direction = np.array((1., 0., 0.))
    direction /= np.linalg.norm(direction)

    uinc = incident_plane(domain, direction)

    sol, info = gmres(M, uinc, rtol=1e-5)

    return sol, problem 

def bempp_solve(points_per_wavelength, parameters, points_eval):
    l_cube_ext = parameters["SIZE"]
    rad = l_cube_ext/2
    wavespeed = parameters["WAVESPEED"]
    frec = parameters["FREC"]
    d = np.array([1., 0., 0.], dtype=np.float64)

    la_ext = parameters["LAMBDA_EXT"]
    k_ext = 2*np.pi/la_ext
    la_int = parameters["WAVESPEEDFUN"](0,0,0) / frec
    k_int = 2*np.pi/la_int


    n_t = int(2*rad / (min(la_int, la_ext) / (2*points_per_wavelength)))
    mesh = dolfinx.mesh.create_box(MPI.COMM_WORLD, [np.array([-rad, -rad, -rad]), 
                                                    np.array([rad, rad, rad])],
                                   [n_t,n_t,n_t], cell_type=dolfinx.mesh.CellType.tetrahedron)
    fenics_space = FunctionSpace(mesh, ("CG", 1))
    trace_space, trace_matrix = \
        fenicsx.fenics_to_bempp_trace_data(fenics_space)
    bempp_space = bempp.api.function_space(trace_space.grid, "P", 1)

    fem_size = fenics_space.dofmap.index_map.size_global
    bem_size = bempp_space.global_dof_count

    id_op = bempp.api.operators.boundary.sparse.identity(
        trace_space, bempp_space, bempp_space)
    mass = bempp.api.operators.boundary.sparse.identity(
        bempp_space, bempp_space, trace_space)
    dlp = bempp.api.operators.boundary.helmholtz.double_layer(
        trace_space, bempp_space, bempp_space, k_ext)
    slp = bempp.api.operators.boundary.helmholtz.single_layer(
        bempp_space, bempp_space, bempp_space, k_ext)
    
    u = ufl.TrialFunction(fenics_space)
    v = ufl.TestFunction(fenics_space)

    @bempp.api.complex_callable
    def u_inc(x, n, domain_index, result):
        result[0] = np.exp(1j * k_ext * np.dot(x, d))
    u_inc = bempp.api.GridFunction(bempp_space, fun=u_inc)

    rhs_fem = np.zeros(fem_size)
    rhs_bem = u_inc.projections(bempp_space)
    rhs = np.concatenate([rhs_fem, rhs_bem])
    blocks = [[None,None],[None,None]]

    trace_op = LO(trace_matrix.shape, lambda x:trace_matrix @ x)


    normal_vec = ufl.FacetNormal(mesh)
    x = ufl.SpatialCoordinate(mesh)
    gl = rad
    rho = 800 * ufl.sin((x[0]-gl)*ufl.pi/gl)*  ufl.sin((x[1]-gl)*ufl.pi/gl) * ufl.sin((x[2]-gl)*0.5*ufl.pi/gl) + 1000 
    k_int = 1.2 * k_ext


    At = (ufl.inner((1/rho)*ufl.grad(u), ufl.grad(rho*v)) - k_int**2 * ufl.inner(u, v))* ufl.dx


    A = fenicsx.FenicsOperator(At)

    blocks[0][0] = A.weak_form()
    blocks[0][1] = -trace_matrix.T * mass.weak_form().to_sparse()    
    blocks[1][0] = (.5 * id_op - dlp).weak_form() * trace_op
    blocks[1][1] = slp.weak_form()

    blocked = BlockedDiscreteOperator(np.array(blocks))

    P1 = InverseSparseDiscreteBoundaryOperator(
        blocked[0,0].to_sparse().tocsc())

    P2 = InverseSparseDiscreteBoundaryOperator(
        bempp.api.operators.boundary.sparse.identity(
            bempp_space, bempp_space, bempp_space).weak_form())

    def apply_prec(x):
        """Apply the block diagonal preconditioner"""
        m1 = P1.shape[0]
        m2 = P2.shape[0]
        n1 = P1.shape[1]
        n2 = P2.shape[1]
        
        res1 = P1.dot(x[:n1])
        res2 = P2.dot(x[n1:])
        return np.concatenate([res1, res2])

    p_shape = (P1.shape[0] + P2.shape[0], P1.shape[1] + P2.shape[1])
    P = LinearOperator(p_shape, apply_prec, dtype=np.dtype('complex128'))


    soln, info = gmres(blocked, rhs, M=P, restart=1000)

    soln_fem = soln[:fem_size]
    soln_bem = soln[fem_size:]

    u_sol = Function(fenics_space)
    u_sol.vector[:] = np.ascontiguousarray(np.real(soln_fem))

    points = points_eval
    plot_me = np.zeros(points.shape[1], dtype=np.complex128)

    x,y,z = points
    bem_x = np.logical_not((np.abs(x)<rad) * (np.abs(y)<rad) * (np.abs(z)<rad))

    fem_points = points[:, np.logical_not(bem_x)].transpose()
    try:
        tree = dolfinx.geometry.BoundingBoxTree(mesh, 3)
    except TypeError:
        tree = dolfinx.geometry.bb_tree(mesh, 3)
    midpoint_tree = dolfinx.geometry.create_midpoint_tree(
        mesh, 3, list(range(mesh.topology.connectivity(3, 0).num_nodes))
    )
    entities = []
    for point in fem_points:
        entities.append(dolfinx.geometry.compute_closest_entity(tree, midpoint_tree, mesh, point)[0])
    fem_val = u_sol.eval(fem_points, entities)

    plot_me[np.logical_not(bem_x)] += fem_val.T[0]

    return plot_me
