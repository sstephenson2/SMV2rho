#!/usr/bin/env python3

# Functions to include temperature dependence in crustal 
#    density calculation.
# Simple implementation which does not require equation
#    of state.  Just uses constant and independent thermal expansion,
#    compressibility etc.

###################################################################
# import modules
import numpy as np
import sys
import os, os.path
from dataclasses import dataclass

###################################################################

@dataclass
class GeothermConstants:
    """
    Data class to store constants related to geothermal properties.

    Parameters:
    ----------
        tc (float): Crustal thickness, km
        T0 (float): Temperature at the surface, C
        T1 (float): Temperature at the base of the crust, C
        q0 (float): Heat flux at the surface, W/m2
        qm (float): Heat flux at the base, W/m2
        k (float): Thermal conductivity, W/(m K)
        H0 (float): Internal heat production at the surface W/(kg m3)
        hr (float): Decay lengthscale of heat production, km
        rho (float): Density, g/mc3
    """
            
    tc:  float = None    # crustal thickness 
    T0:  float = 10.0    # temperature at surface
    T1:  float = 600.0   # temperature at base of crust
    q0:  float = 59e-3   # heat flux at surface
    qm:  float = 30e-3   # heat flux at base
    k:   float = 2.5     # thermal conductivity
    H0:  float = 7e-10   # internal heat production at surface
    hr:  float = 10.0    # decay lengthscale of heat production
    rho: float = 2.9     # density

@dataclass
class GeothermConstantUncertainties:
    """
    This data class stores uncertainties related to geothermal properties.
    Each property is represented as a float value.

    Parameters:
    ----------
        tc_unc (float): Crustal thickness, km
        T0_unc (float): Temperature at the surface, C
        T1_unc (float): Temperature at the base of the crust, C
        q0_unc (float): Heat flux at the surface, W/m2
        qm_unc (float): Heat flux at the base, W/m2
        k_unc (float): Thermal conductivity, W/(m K)
        H0_unc (float): Internal heat production at the surface W/(kg m3)
        hr_unc (float): Decay lengthscale of heat production, km
        rho_unc (float): Density, g/cm3
    """

    tc_unc:  float = 0.0     # crustal thickness 
    T0_unc:  float = 0.0     # temperature at surface
    T1_unc:  float = 200.0   # temperature at base of crust
    q0_unc:  float = 14e-3   # heat flux at surface
    qm_unc:  float = 10e-3   # heat flux at base
    k_unc:   float = 0       # thermal conductivity
    H0_unc:  float = 2e-10   # internal heat production at surface
    hr_unc:  float = 5.0     # decay lengthscale of heat production
    rho_unc: float = 0.0     # density

class Geotherm(GeothermConstants):
    """
    A class used to represent a Geotherm.

    A Geotherm is a model for calculating the temperature profile in the
    Earth's crust. This class provides several different geotherm models,
    which can be selected using the `geotherm_type` parameter.

    The class inherits from GeothermConstants, which provides default values
    for various geothermal properties. These defaults can be overridden by
    providing keyword arguments when creating a Geotherm instance.

    Parameters
    ----------
    geotherm_type : str, optional
        The type of geotherm model to use. Default is "linear".
    uncertainty_constants : GeothermConstantUncertainties, optional
        An instance of GeothermConstantUncertainties that stores the
        uncertainties related to each geothermal property. Default is an
        instance of GeothermConstantUncertainties with default values.
    **kwargs
        Additional keyword arguments to override constants.

    Attributes
    ----------
    geotherm_type : str
        The type of geotherm model to use.
    uncertainties : GeothermConstantUncertainties
        An instance of GeothermConstantUncertainties that stores the
        uncertainties related to each geothermal property.

    Methods
    -------
    __call__(z)
        Evaluate the geothermal model at a given depth or depths.
    linear(z)
        Calculate the temperature at a given depth or depths using a linear
        geothermal model.
    single_layer_internal_heat(z)
        Calculate the temperature at a given depth or depths using a
        single-layer internal heat geothermal model.
    single_layer_flux_difference(z)
        Calculate the temperature at a given depth or depths using a
        single-layer flux difference geothermal model.
    single_layer_temperature_difference(z)
        Calculate the temperature at a given depth or depths using a
        single-layer temperature difference geothermal model.
    generate_geotherm(z_slices)
        Generate the geotherm for the current set of parameters.
    generate_geotherm_distribution(n_geotherms, z_slices)
        Generate a family of geothermal models based on the mean and 
        uncertainty values of the constants using Monte Carlo sampling
        of parameter uncertainties.  uncertainty_constants must not be set to
        None when calling this method.
    
    Examples
    --------
    # Example 1: Using a linear geothermal model to evaluate temperature at 
    #   10 km depth.
    >>> geotherm = Geotherm(geotherm_type="linear")
    >>> temperature = geotherm(10000)
    >>> print(temperature)

    # Example 2: Using a single-layer internal heat geotherm model to generate
    #   a geotherm with 150 depth slices.
    >>> geotherm = Geotherm(geotherm_type="single_layer_internal_heat")
    >>> temperatures = geotherm.generate_geotherm(z_slices = 150)
    >>> print(temperatures)

    # Example 3: Using a single-layer temperature difference geotherm model to
    #   generate a family of geotherms with 200 depth slices each.
    >>> geotherm = Geotherm(
    ...     geotherm_type="single_layer_temperature_difference",
    ...     uncertainty_constants=GeothermConstantUncertainties(),
    ...     tc=30000)
    >>> geotherm.generate_geotherm_distribution(n_geotherms=100, z_slices=200)
    >>> print(geotherm.T_family)
    """

    def __init__(self, 
                geotherm_type = "linear", 
                uncertainty_constants = None, 
                **kwargs
                ):
        
        super().__init__(**kwargs)
        self.geotherm_type = geotherm_type

        # get the uncertainties object avoiding mutable default arguments
        if uncertainty_constants is None:
            self.uncertainties = GeothermConstantUncertainties()
        else:
            self.uncertainties = uncertainty_constants

    def __call__(self, z):
        """
        Evaluate the geothermal model at a given depth or depths.

        Parameters:
            z (Union[float, np.ndarray]): Depth or depths at which to 
            evaluate the geothermal model.  If a NumPy array is provided, 
            the function supports broadcasting.

        Returns:
            float or np.ndarray: Result of the geothermal model at 
            the given depth or depths.
        """

        if self.geotherm_type == "linear":
            return self.linear(z) 
        elif self.geotherm_type == "single_layer_internal_heat":
            return self.single_layer_internal_heat(z)
        elif self.geotherm_type == "single_layer_flux_difference":
            return self.single_layer_flux_difference(z)
        elif self.geotherm_type == "single_layer_temperature_difference":
            return self.single_layer_temperature_difference(z)
        else:
            raise ValueError(f"Unknown geotherm type: {self.geotherm_type}")

    def linear(self, z):
        """
        Linear geothermal model.

        Parameters:
            z (Union[float, np.ndarray]): Depth or depths at which to 
            evaluate the model.

        Returns:
            Union[float, np.ndarray]: Result of the linear geothermal 
            model at the given depth or depths.
        """

        T0 = self.T0
        T1 = self.T1
        tc = self.tc * 1000

        z = z*1000

        return T0 + ((T1 - T0)/tc) * z
        

    def single_layer_internal_heat(self, z):
        """
        Single-layer model with internal heat generation and heat flux 
        at the moho.

        Parameters:
            z (Union[float, np.ndarray]): Depth or depths at which to 
            evaluate the model.

        Returns:
            Union[float, np.ndarray]: Result of the single-layer internal 
            heat geothermal model at the given depth or depths.
        """

        T0 = self.T0
        H0 = self.H0
        rho = self.rho * 1000
        qm = self.qm
        k = self.k
        hr = self.hr * 1000

        z = z*1000

        return (T0 + (qm * z / k) + (rho * H0 * hr**2 / k) 
                * (1 - np.exp(-z/hr)))

    def single_layer_flux_difference(self, z):
        """
        Single-layer model with internal heat generation based on heat 
        flux at the surface and at the moho.  Does not require internal heat
        production or crustal thickness as arguments.

        Parameters:
            z (Union[float, np.ndarray]): Depth or depths at which to 
            evaluate the model.

        Returns:
            Union[float, np.ndarray]: Result of the single-layer flux 
            difference geothermal model at the given depth or depths.
        """

        T0 = self.T0
        q0 = self.q0
        qm = self.qm
        k = self.k
        hr = self.hr * 1000

        z = z * 1000

        return T0 + (qm * z / k) + ((q0 - qm) * hr / k) * (1 - np.exp(-z/hr))
    
    def single_layer_temperature_difference(self, z):
        """
        Single-layer model including internal heat generation based on the 
        temperature at the surface and at the moho.

        Parameters:
            z (Union[float, np.ndarray]): Depth or depths at which to 
            evaluate the model.

        Returns:
            Union[float, np.ndarray]: Result of the single-layer temperature 
            difference geothermal model at the given depth or depths.
        """
        
        T0 = self.T0
        T1 = self.T1
        tc = self.tc * 1000
        rho = self.rho * 1000
        H0 = self.H0
        hr = self.hr * 1000
        k = self.k

        z = z * 1000

        return (T0 + ((z/tc) * (T1-T0)) + ((rho * H0 * hr**2)/k) 
                * (((z/tc) * (np.exp(-tc/hr) - 1)) + (1 - np.exp(-z/hr))))
    
    def generate_geotherm(self, z_slices=100):
        """
        Generate a geothermal model based on the constants provided.

        Returns:
            np.ndarray: Array of depths at which to evaluate the model.
            np.ndarray: Array of temperatures at the given depths.
        """

        self.z = np.linspace(0, self.tc, z_slices)
        self.T = self(self.z)

        return self.z, self.T
    
    def generate_geotherm_distribution(self, n_geotherms=100, z_slices=100):
        """
        Generate a family of geothermal models based on the mean and 
        uncertainty values of the constants.

        Parameters:
            n_geotherms (int): Number of geotherms to generate.
            z_slices (int): Number of slices to divide the depth into.

        Returns:
            np.ndarray: Array of depths at which to evaluate the models.
            np.ndarray: Array of temperatures at the given depths for each 
            model.
        """

        z = np.linspace(0, self.tc, z_slices)
        T_family = np.empty((n_geotherms, z_slices))

        for i in range(n_geotherms):
            # Initialize a dictionary to store the parameters
            params = {}

            # Iterate over the names of the parameters
            for name in self.__annotations__.keys():
                # Get the mean and standard deviation for the current 
                # parameter
                mean = getattr(self, name)
                std_dev = getattr(self.uncertainties, f"{name}_unc")

                # Generate a random number from a normal distribution with the
                # mean and standard deviation
                random_normal = np.random.normal(mean, std_dev)

                # Take the absolute value of the random number
                param = np.abs(random_normal)

                # Store the generated parameter in the dictionary
                params[name] = param

            # Create a new Geotherm instance with the random parameters
            geotherm = Geotherm(**params)

            # Generate the geotherm and store it in the array
            _, T_family[i] = geotherm.generate_geotherm(z_slices)

        # Assign the results to the instance variables
        self.z = z
        self.T_family = T_family

        return self.z, self.T_family




###################################################################

# define linear continental geotherm
# z, hr must be in metres
def cont_geotherm_linear(z, T1, tc, T0=10):
    return T0 + ((T1 - T0)/tc) * z/1000

# define geotherm with internal heat generation
# (1 layer from Turcotte and Schubert)
# z, hr must be in metres
def cont_geotherm_internal_heat(z, T0=10, qm=30e-3, k=2.5, rho=2800, 
                                H0=7e-10, hr=10000):
    return T0 + (qm * z / k) + (rho * H0 * hr**2 / k) * (1 - np.exp(-z/hr))

# define geotherm with internal heat generation by difference between 
#    basal and surface flux
# (1 layer from Turcotte and Schubert)
# z must be in metres
def cont_geotherm_heat_flux_difference(z, T0=10, qm=30e-3, q0=59e-3, 
                                       k=2.5, hr=10000):
    return T0 + (qm * z / k) + ((q0 - qm) * hr / k) * (1 - np.exp(-z/hr))

# define geotherm with constant basal and surface temperature
# and internal heat generation and given crustal thickness, tc.
# z must be in metres
def cont_geotherm_temperature(z, T1=600, T0=10, rho=2850, 
                              H0=7e-10, k=2.5, hr=10000, tc=30000):
    """
    Calculate geotherm based on constant basal and surface temperature,
    heat production and crustal thickness
    Arguments: - z - depth
               - T1 - basal tmperature (in degrees C).  Default = 600C.
               - T0 - Surface temperature (in degrees C).  Default 10C.
               - rho - rock density (in kg/m3).  Default 2850.
               - H0 - internal heat generation.  Default 7e-7.
               - k - thermal conductivity (in W/m/C).  Default = 2.5
               - hr - decay lengthscale of radioactive heat generation (in m),
                        default = 10000m
               - tc - Crustal thickness (in m).  Default = 30000m
    """
    return (T0 + ((z/tc) * (T1-T0)) + ((rho * H0 * hr**2)/k) * 
            (((z/tc) * (np.exp(-tc/hr) - 1)) + (1 - np.exp(-z/hr))))

###################################################################

# (elastic) temperature dependence of velocity correction
def V_T_correction(T, dvdT=-4e-4):
    return T * dvdT

###################################################################

# thermal expansion (linear) -- value of correction.
# Add to rho_o for absolute rho
def rho_thermal(rho_o, T, alpha=3e-5):
    """
    Calculate simple thermal expansion
    Arguments: - rho_o - surface (reference) density
               - T - temperature (in deg C)
               - T0 - surface temperature (in same units as T)
               - alpha0 - thermal expansion coefficient
    Returns:   - abs_rho - density at temperature, T
               - frac_change - fractional change in density (i.e 1.1 = 10% velocity increase)
               - rel_rho - relative density change (e.g. 0.1 = 10 kg/m3 density increase)
    """
    frac_change = np.exp(-alpha * T)
    abs_rho = rho_o * frac_change
    rel_rho = abs_rho - rho_o
    return abs_rho, frac_change, rel_rho

# thermal expansion (temperature-dependent) -- value of correction.
# Add to rho_o for absolute rho
def rho_thermal2(rho_o, T, alpha0=1.0e-5, alpha1=2.9e-8, T0=10):
    """
    Calculate effect of temperature on density
    Arguments: - rho_o - surface (reference) density
               - T - temperature (in deg C)
               - alpha0 - thermal expansion at 0K (K^{-1})
               - alpha1 - thermal expansion temperature derivative (K^{-2})
               - T0 - surface temperature (in same units as T)
    Returns:   - abs_rho - density at temperature, T
               - frac_change - fractional change in density (i.e 1.1 = 10% velocity increase)
               - rel_rho - relative density change (e.g. 0.1 = 10 kg/m3 density increase)
    """
    frac_change = np.exp( - ((alpha0 * (T - T0)) + ((alpha1/2) * 
                           (((T + 273)**2) - (T0 + 273)**2))))
    abs_rho = rho_o * frac_change
    rel_rho = rho_o * (frac_change - 1)
    
    return abs_rho, frac_change, rel_rho

###################################################################

# compressibility
# reads in P in MPa
# value of correction.
# Add to rho_o for absolute rho
def compressibility(rho_o, P, K=90e9):
    """
    Calculate effect of confining pressure on density
    Arguments: - rho_o - surface (reference) density
               - P - pressure (in MPa)
               - K - Bulk modulus (in Pa)
    Returns:   - abs_rho - density at pressure, P
               - frac_change - fractional change in density 
                    (i.e 1.1 = 10% velocity increase)
               - rel_rho - relative density change (e.t. 0.1 = 10 Mg/m3 
                    density increase)
    """
    frac_change = np.exp(P * 1e6 / K)
    abs_rho = rho_o * np.exp(P * 1e6 / K)
    rel_rho = rho_o * np.exp(P * 1e6 / K) - rho_o
    return abs_rho, frac_change, rel_rho

