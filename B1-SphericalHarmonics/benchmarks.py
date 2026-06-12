import os
import sys
import numpy as np
from scipy.special import spherical_jn as jn, spherical_yn as yn, eval_legendre

sys.path.append(os.path.join(os.path.abspath(''), '../'))

from vsie.operators.self_interaction import single_layer 
from vsie.operators.self_interaction import mass_matrix 
from vsie.operators.wave_op import incident_plane 
from vsie.geometry.grid import sphere 
from vsie.geometry.space_fun import discont_physical_functions
from vsie.model.problem import Problem 
from vsie.model.problem import Subdomain 

from scipy.sparse.linalg import gmres

import bempp.api


def analytical_solution(space, parameters):

    h2n = lambda n,a,derivative=False: jn(n, a, derivative=derivative) - 1j * yn(n, a, derivative=derivative)

    def unstack(a, axis=0):
        return np.moveaxis(a, axis, 0)

    def coefficients_for_transmission(
        k_ext           : np.float32,
        k_int           : np.float32,
        rho_ext         : np.float32,
        rho_int         : np.float32,
        r               : np.float32=1,
        epsilon         : np.float32=1E-5,
        max_ite         : int=50,
    ):
        psca_coef = np.zeros(max_ite, dtype=np.complex128)
        pint_coef = np.zeros(max_ite, dtype=np.complex128)

        rho = rho_int/rho_ext
        k = k_ext/k_int

        n = 0
        while n < max_ite:
            jn_int = jn(n, k_int * r)
            jn_ext = jn(n, k_ext * r)
            h2n_ext = h2n(n, k_ext * r)

            d_jn_int = jn(n, k_int * r, derivative=True)
            d_jn_ext = jn(n, k_ext * r, derivative=True)
            d_h2n_ext = h2n(n, k_ext * r, derivative=True)

            tau_num = (2*n + 1) * (-1j)**n * (d_jn_int * jn_ext - rho * k * jn_int * d_jn_ext)
            tau_den = rho * k * jn_int * d_h2n_ext - d_jn_int * h2n_ext

            tau_n = tau_num/tau_den
            ups_n = ((2*n + 1) * (-1j)**n * jn_ext + tau_n * h2n_ext) / jn_int

            psca_coef[n] = tau_n
            pint_coef[n] = ups_n

            n += 1

        return psca_coef, pint_coef

    '''
        Simulation related
    '''
    EPSILON = 1E-5
    RAD     = parameters["RADIUS"]
    lambda_ext = parameters["LAMBDA_EXT"]
    frec = parameters["FREC"]
    rho_0 = parameters["RHO_0"]
    density = parameters["DENSITYFUN"]
    int_wavespeed = parameters["INTWAVESPEEDFUN"]
    ext_wavespeed = parameters["EXTWAVESPEEDFUN"]


    '''
        Visual parameters
    '''
    le = lambda_ext
    li = int_wavespeed(0,0,0) / frec
    ke = 2 * np.pi / le
    ki = 2 * np.pi / li
    re = rho_0
    ri = rho_0
    k = max(ke, ki)


    n_space = int(np.round(space.shape[0]**(1/2)))

    x, y, z = unstack(space, axis=1)

    rs = np.sqrt(x**2 + y**2 + z**2)

    ps = x/rs
  
    for kk in range(len(rs)):
        if rs[kk] == 0:
            ps[kk] = 0


    ext = rs > RAD


    '''
        Analytical Solutions
    '''
    coef = coefficients_for_transmission(ke, ki, re, ri,
                                            r=RAD,
                                            epsilon=EPSILON)

    pinc = np.zeros_like(rs, dtype=np.complex128)
    pinc[ext] = np.exp(-1j * ke * x[ext])

    psca = np.zeros_like(x, dtype=np.complex128)
    pint = np.zeros_like(x, dtype=np.complex128)

    for n, (an, bn) in enumerate(zip(*coef)):
        psca[ext] += an * h2n(n, ke * rs[ext]) * eval_legendre(n, ps[ext])

        pint[~ext] += bn* jn(n, ki * rs[~ext]) * eval_legendre(n, ps[~ext])

    ptot = (psca + pinc + pint)


    return ptot


def VSIE_solve(points_per_wavelength, parameters):
    '''
    Solves the problem using the VSIE method.
    '''
    lambda_ext = parameters["LAMBDA_EXT"]
    rho_0 = parameters["RHO_0"]
    wavespeed = parameters["WAVESPEED"]
    frec = parameters["FREC"]
    dx = lambda_ext/points_per_wavelength

    density = parameters["DENSITYFUN"]
    int_wavespeed = parameters["INTWAVESPEEDFUN"]
    ext_wavespeed = parameters["EXTWAVESPEEDFUN"]

    size = parameters["SIZE"]
    radius = parameters["RADIUS"]

    problem = Problem(rho_0, wavespeed, frec)

    ext_space, int_space, _ = sphere(radius, size, dx, problem)

    ext_space.density = density
    int_space.density = density
    ext_space.wavespeed = ext_wavespeed
    int_space.wavespeed = int_wavespeed 

    problem.set_problem()

    space = Subdomain(
        problem
    )
    weight = np.zeros(problem.mesh.shape[0], dtype=np.complex128)
    alpha = np.zeros(problem.mesh.shape[0], dtype=np.complex128)
    beta = np.zeros(problem.mesh.shape[0], dtype=np.complex128)
    weight[problem.masks[0]] = ext_space.volumes
    weight[problem.masks[1]] = int_space.volumes
    alpha[problem.masks[0]] = ext_space.alpha
    alpha[problem.masks[1]] = int_space.alpha
    beta[problem.masks[0]] = ext_space.beta
    beta[problem.masks[1]] = int_space.beta

    space.set_geometry(problem.mesh, weight)
    space.alpha = alpha
    space.beta = beta
                      

    mass_op = mass_matrix(space)
    sl_op = single_layer(space)
    M = mass_op - sl_op

    direction = np.array((1., 0., 0.))
    direction /= np.linalg.norm(direction)
    uinc = incident_plane(space, direction)

    sol, info = gmres(M, uinc, rtol=1e-5)

    return sol, problem

def bempp_solve(points_per_lambda, parameters):
    lambda_ext = parameters["LAMBDA_EXT"]

    radius = parameters["RADIUS"]
    kappa = 2*np.pi / lambda_ext
    h = 2 * np.pi/(points_per_lambda * kappa)
    grid = bempp.api.shapes.sphere(r=radius, origin=(0,0,0), h=h)

    @bempp.api.complex_callable
    def dirichlet_fun(x, n, domain_index, result):
        result[0] = np.exp(1j * kappa * x[0])

    @bempp.api.complex_callable
    def neumann_fun(x, n, domain_index, result):
        result[0] = 1j * kappa * n[0] * np.exp(1j *kappa * x[0])

    Ai = bempp.api.operators.boundary.helmholtz.multitrace_operator(grid, 1.2 * kappa)
    Ae = bempp.api.operators.boundary.helmholtz.multitrace_operator(grid, kappa)

    op = (Ai + Ae)
    op_squared = op * op

    dirichlet_space = Ai[0, 0].domain
    neumann_space = Ai[0, 1].domain

    dirichlet_grid_fun = bempp.api.GridFunction(dirichlet_space, fun=dirichlet_fun)
    neumann_grid_fun = bempp.api.GridFunction(neumann_space, fun=neumann_fun)

    op_discrete = op.strong_form()
    op_discrete_squared = op_discrete * op_discrete
    rhs = op_discrete * np.concatenate([dirichlet_grid_fun.coefficients,
                                        neumann_grid_fun.coefficients])
    

    from scipy.sparse.linalg import gmres

    x, info = gmres(op_discrete_squared, rhs, rtol=1e-5)

    return x, grid

def get_bempp_pressure(x, parameters, grid, points, Nx, C=2):
    lambda_ext = parameters["LAMBDA_EXT"]
    radius = parameters["RADIUS"]


    kappa = 2*np.pi / lambda_ext

    @bempp.api.complex_callable
    def dirichlet_fun(x, n, domain_index, result):
        result[0] = np.exp(1j * kappa * x[0])

    @bempp.api.complex_callable
    def neumann_fun(x, n, domain_index, result):
        result[0] = 1j * kappa * n[0] * np.exp(1j *kappa * x[0])

    Ai = bempp.api.operators.boundary.helmholtz.multitrace_operator(grid, 1.2 * kappa)
    Ae = bempp.api.operators.boundary.helmholtz.multitrace_operator(grid, kappa)

    dirichlet_space = Ai[0, 0].domain
    neumann_space = Ai[0, 1].domain

    total_field_dirichlet = bempp.api.GridFunction(
        dirichlet_space, coefficients=x[:dirichlet_space.global_dof_count])
    total_field_neumann = bempp.api.GridFunction(
        neumann_space, coefficients=x[dirichlet_space.global_dof_count:])

    x, y, z = points

    eps = 2*C/(Nx)

    idx_ext = np.sqrt(x**2 + y**2+ z**2) > radius
    idx_int = np.sqrt(x**2 + y**2 + z**2) < radius 
    idx_bry = np.abs(np.sqrt(x**2 + y**2 + z**2) - radius) <= eps

    points_exterior = points[:, idx_ext]
    points_interior = points[:, idx_int]
    points_boundary = points[:, idx_bry]

    slp_pot_int = bempp.api.operators.potential.helmholtz.single_layer(
        dirichlet_space, points_interior, 1.2 * kappa)
    slp_pot_ext = bempp.api.operators.potential.helmholtz.single_layer(
        dirichlet_space, points_exterior, kappa)
    dlp_pot_int = bempp.api.operators.potential.helmholtz.double_layer(
        dirichlet_space, points_interior, 1.2 * kappa)
    dlp_pot_ext = bempp.api.operators.potential.helmholtz.double_layer(
        dirichlet_space, points_exterior,  kappa)

    bry_pressure = compute_pressure_boundary(grid, points_boundary, total_field_dirichlet.coefficients)

    total_field_int = (slp_pot_int * total_field_neumann
                    - dlp_pot_int * total_field_dirichlet).ravel()
    total_field_ext = (dlp_pot_ext * total_field_dirichlet
                    - slp_pot_ext * total_field_neumann).ravel() \
        + np.exp(1j * kappa * points_exterior[0])

    total_field = np.zeros(points.shape[1], dtype='complex128')
    total_field[idx_ext] = total_field_ext
    total_field[idx_int] = total_field_int
    total_field[idx_bry] = bry_pressure

    return total_field


# =======================================================================
# OptimUS helper functions
# =======================================================================

def _normalize_vector(vector):
    """Convert a vector into a unit vector.
    For 2D input arrays, the columns will be normalized.

    Parameters
    ----------
    vector : numpy.ndarray
        An array of size (n,) or (n,m) with the m input vectors of dimension n.

    Returns
    -------
    unit_vector : numpy.ndarray
        Array of size (n,) or (n,m) with the m vectors of dimension n scaled
        to unit Euclidean length.
    """

    if not isinstance(vector, np.ndarray):
        raise TypeError("Vector needs to be a Numpy array.")

    if vector.ndim == 1:
        vector_norm = np.linalg.norm(vector)
        if np.isclose(vector_norm, 0):
            raise ValueError("The vector cannot be normalised because it is zero.")
        else:
            return vector / vector_norm
    elif vector.ndim == 2:
        vector_norms = np.linalg.norm(vector, axis=0)
        if np.any(np.isclose(vector_norms, 0)):
            raise ValueError(
                "The vectors cannot be normalised because at least one is zero."
            )
        else:
            return vector / vector_norms
    else:
        raise ValueError("Vector needs to be 1D or 2D.")

def compute_pressure_boundary(grid, boundary_points, dirichlet_solution):
    """Calculate pressure for points near or at the boundary of a domain. When the solid
    angle associated with a boundary vertex is below 0.1, it is assumed to lie on the
    boundary.

    Parameters
    ----------
    grid : bempp.api.Grid
        The surface mesh of bempp.
    boundary_points : numpy.ndarray
        An array of size (3,N) with the coordinates of vertices
        on the domain boundary.
    dirichlet_solution : numpy.ndarray
        An array of size (N,) with the Dirichlet component of the
        solution vector on the boundary.

    Returns
    -------
    total_boundary_pressure : numpy.ndarray
        An array of size (N,) with complex values of the pressure field.

    """

    vertices = grid.vertices
    elements = grid.elements
    centroids = np.mean(vertices[:, elements], axis=1)

    # Initialise arrays with None-values for the element indices
    n = boundary_points.shape[1]
    element_index = np.repeat(None, n)

    # Loop over all centroids and find the elements within which boundary points lie
    for i in range(n):
        eucl_norm = np.linalg.norm(
            centroids - np.atleast_2d(boundary_points[:, i]).transpose(), axis=0
        )

        comp = np.where(eucl_norm == np.min(eucl_norm))[0]

        if comp.size != 0:
            element_index[i] = comp[0]

    space = bempp.api.function_space(grid, "P", 1)
    grid_function = bempp.api.GridFunction(space, coefficients=dirichlet_solution)
    local_coords = np.zeros((2, n), dtype=float)
    total_boundary_pressure = np.zeros(n, dtype="complex128")

    # Loop over elements within which near points lie
    for i in range(n):

        # Obtain vertices of element
        vertices_elem = vertices[:, elements[:, element_index[i]]].transpose()

        # Translate element so that first vertex is global origin
        vertices_translated = vertices_elem - vertices_elem[0, :]
        boundary_point_translated = boundary_points[:, i] - vertices_elem[0, :]

        # Compute element normal
        vector_a = vertices_translated[1, :] - vertices_translated[0, :]
        vector_b = vertices_translated[2, :] - vertices_translated[0, :]
        vector_a_cross_vector_b = np.cross(vector_a, vector_b)

        # if linalg.norm(vector_a_cross_vector_b) < eps: print(vertex) (np.close)
        element_normal = _normalize_vector(vector_a_cross_vector_b)

        # Obtain first rotation matrix for coordinate transformation
        h = np.sqrt(element_normal[0] ** 2 + element_normal[1] ** 2)
        if h != 0:
            r_z = np.array(
                [
                    [element_normal[0] / h, element_normal[1] / h, 0],
                    [-element_normal[1] / h, element_normal[0] / h, 0],
                    [0, 0, 1],
                ]
            )
        else:
            r_z = np.identity(3, dtype=float)

        # Obtain rotated element normal
        element_normal_rotated = np.matmul(r_z, element_normal)

        # Obtain second rotation matrix for coordinate transformation
        r_y = np.array(
            [
                [element_normal_rotated[2], 0, -element_normal_rotated[0]],
                [0, 1, 0],
                [element_normal_rotated[0], 0, element_normal_rotated[2]],
            ]
        )

        # Obtain total rotation matrix
        r_y_mult_r_z = np.matmul(r_y, r_z)
        vertices_0_transformed = np.matmul(r_y_mult_r_z, vertices_translated[0, :])
        vertices_1_transformed = np.matmul(r_y_mult_r_z, vertices_translated[1, :])
        vertices_2_transformed = np.matmul(r_y_mult_r_z, vertices_translated[2, :])
        boundary_point_transformed = np.matmul(r_y_mult_r_z, boundary_point_translated)

        # Extract vertex coordinates in rotated coordinate system in x-y plane
        x = boundary_point_transformed[0]
        y = boundary_point_transformed[1]
        x0 = vertices_0_transformed[0]
        y0 = vertices_0_transformed[1]
        x1 = vertices_1_transformed[0]
        y1 = vertices_1_transformed[1]
        x2 = vertices_2_transformed[0]
        y2 = vertices_2_transformed[1]

        # Obtain local coordinates in orthonormal system for element
        transformation_matrix = np.array([[x1 - x0, x2 - x0], [y1 - y0, y2 - y0]])
        transformation_matrix_inv = np.linalg.inv(transformation_matrix)
        rhs = np.vstack((x - x0, y - y0))
        local_coords[:, i] = np.matmul(transformation_matrix_inv, rhs).transpose()

        # Required format for element and local coordinates for GridFunction.evaluate
        coord = np.array([[local_coords[0, i]], [local_coords[1, i]]])

        # Calculate pressure phase and magnitude at near point
        total_boundary_pressure[i] = grid_function.evaluate(element_index[i], coord)

    return total_boundary_pressure


    