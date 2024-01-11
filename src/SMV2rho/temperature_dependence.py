#!/usr/bin/env python3

# Functions to include temperature dependence in crustal 
#    density calculation.
# Simple implementation which does not require equation
#    of state.  Just uses constant and independent thermal expansion etc.

# @author: Simon Stephenson, Nov 2022

###################################################################
# import modules
import numpy as np
import sys
import os, os.path
from dataclasses import dataclass

###################################################################

@dataclass
class GeothermConstants:
    tc:  float = None
    T0:  float = 10.0
    T1:  float = 600.0
    q0:  float = 59e-3
    qm:  float = 30e-3
    k:   float = 2.5
    H0:  float = 7e-10
    hr:  float = 10000.0
    rho: float = 2800

class Geotherm(GeothermConstants):
    def __init__(self, geotherm_type="linear", **kwargs):
        super().__init__(**kwargs)
        self.geotherm_type = geotherm_type

    def __call__(self, z):
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

        T0 = self.T0
        T1 = self.T1
        tc = self.tc

        return T0 + ((T1 - T0)/tc) * z
        

    def single_layer_internal_heat(self, z):

        T0 = self.T0
        H0 = self.H0
        rho = self.rho
        qm = self.qm
        k = self.k
        hr = self.hr

        return (T0 + (qm * z / k) + (rho * H0 * hr**2 / k) 
                * (1 - np.exp(-z/hr)))

    def single_layer_flux_difference(self, z):

        T0 = self.T0
        q0 = self.q0
        qm = self.qm
        k = self.k
        hr = self.hr

        return T0 + (qm * z / k) + ((q0 - qm) * hr / k) * (1 - np.exp(-z/hr))
    
    def single_layer_temperature_difference(self, z):
        
        T0 = self.T0
        T1 = self.T1
        tc = self.tc
        rho = self.rho
        H0 = self.H0
        hr = self.hr
        k = self.k

        return (T0 + ((z/tc) * (T1-T0)) + ((rho * H0 * hr**2)/k) 
                * (((z/tc) * (np.exp(-tc/hr) - 1)) + (1 - np.exp(-z/hr))))

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
               - frac_change - fractional change in density (i.e 1.1 = 10% velocity increase)
               - rel_rho - relative density change (e.t. 0.1 = 10 Mg/m3 density increase)
    """
    frac_change = np.exp(P * 1e6 / K)
    abs_rho = rho_o * np.exp(P * 1e6 / K)
    rel_rho = rho_o * np.exp(P * 1e6 / K) - rho_o
    return abs_rho, frac_change, rel_rho

