'''
Module has the definition of the Problem class.
'''

import numpy as _np

class Problem(object):
    """
    Class that defines the problem.
    """

    def __init__(self, density, wavespeed, frequency):
        """
        Constructor of the Problem class.

        Parameters
        ----------
        density: float
            Density of the background medium.
        wavespeed: float
            Wavespeed of the background medium.
        frequency: float
            Frequency of the wave.
        """
        
        self.ext_density = density
        self.ext_wavespeed = wavespeed
        self.frequency = frequency
        self.ext_wavenumber = 2 * _np.pi * self.frequency / self.ext_wavespeed   

        self.subdomains = []
        self.mesh = None
        self.masks = []

    def get_minimum_wavelength(self, dom_lims, wavespeed):
        """
        Get the minimum wavelength of the subdomain.
        Parameters
        ----------
        dom_lims: list
            List of the limits of the domain.
            The list has the form [x_min, x_max, y_min, y_max, z_min, z_max].
        wavespeed: function
            Wavespeed of the subdomain.
        Returns
        -------
        l_min: float
            Minimum wavelength of the subdomain.
        """
        x_min, x_max, y_min, y_max, z_min, z_max = dom_lims
        # Create a meshgrid for the domain limits
        x = _np.linspace(x_min, x_max, 1000)
        y = _np.linspace(y_min, y_max, 1000)
        z = _np.linspace(z_min, z_max, 1000)
        X, Y, Z = _np.meshgrid(x, y, z)
        wave_eval = wavespeed(X, Y, Z)

        # Get the minimum wavelength
        c_min = _np.min(wave_eval)
        l_min = c_min / self.frequency

        return l_min

    def set_problem(self):
        """
        Set the problem.
        """
        for subdomain in self.subdomains:
            subdomain.set_up_subdomain()


class Subdomain(object):
    """
    Class that defines a subdomain of the geometry.
    """

    def __init__(self, 
                 problem,
                 name=None, 
                 density=None, 
                 wavespeed=None, 
                 density_gradient=None):
        """
        Constructor of the Subdomain class.

        Parameters
        ----------
        problem: Problem
            Problem object.
        name: str
            Name of the subdomain.
        density: function
            Density of the subdomain.
        wavespeed: function
            Wavespeed of the subdomain.
        density_gradient: function
            Gradient of the density of the subdomain.
        """
        
        self.problem = problem
        self.problem.subdomains.append(self)
        self.name = name
        self.mesh = None
        self.volumes = None
        self.density = density
        self.density_gradient = density_gradient
        self.wavespeed = wavespeed
        self.surface = False

    def set_geometry(self, mesh, volumes):
        """
        Set the geometry of the subdomain.

        Parameters
        ----------
        mesh: ndarray
            Mesh of the subdomain.
        volumes: ndarray
            Value of the volume of the voxels of the subdomain.
        """
        
        self.mesh = mesh
        self.volumes = volumes

    def create_physical_functions(self):
        """
        Creates the physical functions of the subdomain.
        It creates the alpha and beta functions for the subdomain.
        """
        frequency = self.problem.frequency
        ext_density = self.problem.ext_density
        ext_wavenumber = self.problem.ext_wavenumber


        self.wavenumber = lambda x,y,z: 2 * _np.pi * frequency / \
                                        self.wavespeed(x,y,z)
        
        self.alpha = _np.zeros(self.mesh.shape[0], dtype=_np.complex128)
        self.beta = _np.zeros(self.mesh.shape[0], dtype=_np.complex128)
        self.alpha_gradient = _np.zeros(self.mesh.shape, dtype=_np.complex128)

        # Get the mesh points for the subdomain
        mesh_points = self.mesh

        # Calculate alpha and beta for the subdomain
        alpha = ext_density / self.density(*mesh_points.T) - 1
        beta = ext_density * self.wavenumber(*mesh_points.T)**2 \
               /self.density(*mesh_points.T) - ext_wavenumber**2
        
        # Store the results in the alpha and beta arrays
        self.alpha[:] = alpha
        self.beta[:] = beta

        if self.density_gradient is not None:
            # Calculate the alpha gradient for the subdomain
            self.alpha_gradient[:, 0] = -self.density_gradient[0](*mesh_points.T)/(self.density(*mesh_points.T))**2
            self.alpha_gradient[:, 1] = -self.density_gradient[1](*mesh_points.T)/(self.density(*mesh_points.T))**2
            self.alpha_gradient[:, 2] = -self.density_gradient[2](*mesh_points.T)/(self.density(*mesh_points.T))**2

            self.alpha_gradient = ext_density * self.alpha_gradient


    def get_physical_data(self):
        """
        Creates the physical functions of the subdomain from data.
        It creates the alpha and beta functions for the subdomain.
        """

        frequency = self.problem.frequency
        ext_density = self.problem.ext_density
        ext_wavenumber = self.problem.ext_wavenumber


        self.wavenumber = 2 * _np.pi * frequency / self.wavespeed
        
        self.alpha = _np.zeros(self.mesh.shape[0], dtype=_np.complex128)
        self.beta = _np.zeros(self.mesh.shape[0], dtype=_np.complex128)
        self.alpha_gradient = _np.zeros(self.mesh.shape[0], dtype=_np.complex128)

        # Calculate alpha and beta for the subdomain
        alpha = ext_density / self.density - 1
        beta = ext_density * self.wavenumber**2 \
               / self.density - ext_wavenumber**2
        
        # Store the results in the alpha and beta arrays
        self.alpha[:] = alpha
        self.beta[:] = beta

        if self.density_gradient is not None:
            # Calculate the alpha gradient for the subdomain
            self.alpha_gradient[:, 0] = -self.density_gradient[0]/(self.density)**2
            self.alpha_gradient[:, 1] = -self.density_gradient[1]/(self.density)**2
            self.alpha_gradient[:, 2] = -self.density_gradient[2]/(self.density)**2

    def set_up_subdomain(self):
        """
        Set up the subdomain.
        """
        if self.mesh is None:
            raise ValueError("Mesh not set.")
        if callable(self.density):
            self.create_physical_functions()
        else:
            self.get_physical_data()

class SurfDomain(Subdomain):
    """
    Class that defines a surface subdomain of the geometry.
    """

    def __init__(self, problem, name=None, density=None, wavespeed=None):
        """
        Constructor of the SurfDomain class.

        Parameters
        ----------
        problem: Problem
            Problem object.
        name: str
            Name of the subdomain.
        density: function
            Density of the subdomain.
        wavespeed: function
            Wavespeed of the subdomain.
        """
        
        super().__init__(problem, name, density, wavespeed)
        self.surface = True

    def set_geometry(self, mesh, volumes, normals, inner_subdomain, outer_subdomain=None,
                     density_at_surface=None):
        """
        Set the geometry of the surface subdomain.

        Parameters
        ----------
        mesh: ndarray
            Mesh of the subdomain.
        volumes: ndarray
            Value of the volume of the voxels of the subdomain.
        normals: ndarray
            Normals of the surface of the subdomain.
        inner_subdomain: Subdomain
            Inner subdomain of the surface subdomain.
        outer_subdomain: Subdomain
            Outer subdomain of the surface subdomain.
        density_at_surface: list
            List of ndarrays containing the density values at the surface of the subdomain.
            It is assumed that the first value is the density of the inner subdomain 
            and the second value is the density of the outer subdomain.
            This is used to calculate the alpha difference at the surface of the subdomain
            using data.
        """
        
        self.mesh = mesh
        self.volumes = volumes
        self.normals = normals
        self.inner_subdomain = inner_subdomain
        self.outer_subdomain = outer_subdomain
        self.density_at_surface = density_at_surface


    def create_physical_functions(self):
        """
        Creates the physical functions of the subdomain.
        It creates the jump of the alpha for the surface.
        """

        # Get alpha difference at the surface
        ext_density = self.problem.ext_density

        num_voxels = self.mesh.shape[0]
        self.diff_alpha = _np.zeros(num_voxels, dtype=_np.complex128)
        int_aux = _np.zeros(num_voxels, dtype=_np.complex128)
        ext_aux = _np.zeros(num_voxels, dtype=_np.complex128)
        int_aux[:] = ext_density/self.inner_subdomain.density(*self.mesh.T) - 1
        if self.outer_subdomain is not None:
            ext_aux[:] = ext_density/self.outer_subdomain.density(*self.mesh.T) - 1

        self.diff_alpha[:] = int_aux - ext_aux

        self.density = lambda x,y,z: 2*self.inner_subdomain.density(x,y,z)\
            *self.outer_subdomain.density(x,y,z) \
                / (self.inner_subdomain.density(x,y,z) + self.outer_subdomain.density(x,y,z))
        
        self.alpha = _np.zeros(self.mesh.shape[0], dtype=_np.complex128)
        self.alpha[:] = ext_density/self.density(*self.mesh.T) - 1
            

    def get_physical_data(self):
        """
        Creates the physical functions of the subdomain from data.
        It creates the jump of the alpha for the surface.
        """
        # Get alpha difference at the surface
        ext_density = self.problem.ext_density
        
        num_voxels = self.mesh.shape[0]
        self.diff_alpha = _np.zeros(num_voxels, dtype=_np.complex128)
        int_aux = self.density_at_surface[0]
        ext_aux = self.density_at_surface[1]
        self.diff_alpha[:] = ext_density*(int_aux - ext_aux)/int_aux*ext_aux

        outer_alpha = ext_density/self.density_at_surface[1] - 1
        inner_alpha = ext_density/self.density_at_surface[0] - 1

        self.alpha = 0.5*(outer_alpha + inner_alpha)



    def set_up_subdomain(self):
        """
        Set up the subdomain.
        """
        if self.mesh is None:
            raise ValueError("Mesh not set.")
        if self.density_at_surface is None: # revisar bien
            self.create_physical_functions()
        else:
            self.get_physical_data()

