#!/usr/bin/env python3

# Functions to convert Vs to Vp and Vp to density

# @Author: Simon Stephenson @Date: Aug 2023

###################################################################
# import modules
import numpy as np
import sys
import scipy.integrate as integrate
from scipy.interpolate import interp1d, CubicSpline
import glob
import os, os.path
import SMV2rho.temperature_dependence as td
import SMV2rho.constants as c
from multiprocessing import Pool, Manager
import pandas as pd
import matplotlib.pyplot as plt

###################################################################
# read/write data

# ensure directory exists
def ensure_dir(filename):
    """
    Ensure that the directory for the given filename exists. If the directory 
    does not exist, it will be created.

    Args:
        filename (str): The path to the file.

    Returns:
        None
    """
    d = os.path.dirname(filename)
    if not os.path.exists(d):
        os.makedirs(d)    

# read in files
def read_file(filename, delimiter = None):
    """
    Read the contents of a file and return them as a list of lines.

    Args:
        filename (str): The path to the file.
        delimiter (str, optional): The delimiter used to split lines into 
            elements (default is None, which means lines are not split).

    Returns:
        list: A list of lines from the file.
    """

    data = []

    with open(filename, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.split(delimiter)
            data.append(line)

    return data

# write out stataion data
def write_profile_to_file(data, filename):
    """
    Write data to a file.

    Args:
        data (list): A list of data to be written to the file.
        filename (str): The path to the output file.

    Returns:
        None
    """

    f = open(filename, 'w')
    f.write(data[0] + "\n")
    i = 0
    for d in data[1:]:
        if i == 0:
            f.write(str(d[0]) + " " + str (d[1]) + "\n")
        elif i == 1:
            f.write(str(d[0]) + "\n")
        elif i > 1:
            for e in d:
                f.write(str(e[1]) + " " + str(e[0]) + " " + "\n")
        i += 1
    f.close()

###################################################################
# calculate velocities/density etc.

# Brocher's (2005) regression fit, converts Vs to Vp
def Vs2Vp_brocher(Vs):
    """
    Convert shear wave velocity (Vs) to compressional wave velocity (Vp) 
    using Brocher's (2005) regression fit.

    Args:
        Vs (float or numpy.ndarray): Shear wave velocity in km/s.

    Returns:
        float or numpy.ndarray: Compressional wave velocity (Vp) in km/s.
    """
    return (0.9409 + 2.0947 * Vs - 0.8206 * 
            Vs**2. + 0.2683 * Vs**3. - 0.0251 * Vs**4.)

# Nafe-Drake relationship - converts Vp to density (g/cm3)
def Vp2rho_brocher(Vp):
    """
    Convert compressional wave velocity (Vp) to density (g/cm³) using the 
    Nafe-Drake relationship.

    Args:
        Vp (float or numpy.ndarray): Compressional wave velocity in km/s.

    Returns:
        float or numpy.ndarray: Density in g/cm³.
    """
    return (1.6612 * Vp - 0.4721 * Vp**2. + 0.0671 * Vp**3. - 
            0.0043 * Vp**4. + 0.000106 * Vp**5.)

# my relationship based on mineral physics as function of 
#   pressure and seismic velocity
def V2rho_stephenson(data, parameters):
    """
    Convert seismic wave velocity (Vp or Vs) into density at standard 
    temperature and pressure (s.t.p.) given pressure. This function is 
    based on mineral physics and accounts for pressure effects on velocity 
    and density.  WARNING: this function returns density at standard 
    temperature and pressure.  Please correct for thermal expansion and 
    compression by lithostatic pressure to obtain rho(z).

    NB. This function can be used for both Vp and Vs conversions, but the 
    parameter values will differ. Ensure you use the correct parameters for 
    the velocity type.

    Args:
        data (array-like): An array of pressure and velocity data where 
            data[0] is pressure (in GPa) and data[1] is velocity (in km/s).
        parameters (class instance): Instance of the Constants or
            TemperatureDependentConstants class containig the parameters
            listed below: 

            - v0 (float): Intercept velocity.
            - b (float): Velocity gradient with respect to density.
            - d0 (float): Partial derivative of velocity gradient (dvdr) with 
                respect to pressure (dp).
            - dp (float): Velocity gradient with respect to pressure (dvdpr).
            - c (float): Amplitude of the velocity drop-off at low pressure.
            - k (float): Lengthscale of the velocity drop-off.

    Returns:
        float or numpy.ndarray: Density at standard temperature and pressure 
        (s.t.p.) in g/cm³.
    """
    p = data[0]
    v = data[1]
    return ((v - parameters.v0 - (parameters.b * p) + 
             (parameters.c * np.exp(-parameters.k * p))) / 
               (parameters.d0 + (parameters.dp * p)))


###################################################################
# processing data...

class Convert:
    """
    Convert seismic velocity profiles to various parameters using different 
    approaches.

    This class provides methods to read seismic velocity profiles, convert 
    them to other parameters, and write the converted data to output files.

    Args:
        profile (str): The file path to the seismic profile data.
        profile_type (str): The type of the seismic profile, either "Vp" or 
            "Vs."
        region_name (str, optional): The geographic location of the profile, 
            e.g., "MADAGASCAR." (default is None)
        seismic_method_name (str): The method used to acquire the vlocity 
            profile.  e.g. "RECEIVER_FUNCTION".  If set to None, the read_data
            metod will pick up the argument from the file string.  Note if 
            set to None the strict file convention must be set (see README.md)
            and tutorial_1.ipynb.
        geotherm (class instance): instance of the Geotherm class containing
            information about the temperature profile at the site of the
            seismic profile.

    Attributes:
        data (dict): A dictionary containing parsed seismic profile data.
        moho (float): The Moho depth parsed from the profile data.

    Methods:
        read_data: Read in data file and parse it into a data dictionary.
        convert_profile_brocher: Convert Vs profile to Vp profile using 
            Brocher's (2005) approach.
        Vp_to_density_brocher: Convert Vp profile to density using Brocher 
            (2005) method.
        V_to_density_stephenson: Convert Vp profile to density using the 
            Stephenson method described in the study.
        write_data: Write the converted data to appropriate file locations 
            based on the specified conversion approach and temperature 
            dependence settings.

    """

    def __init__(self, profile, profile_type = None, region_name = None,
                 seismic_method_name = None, geotherm = None):

        # check whether the profile type has been selected
        if profile_type is None:
            raise NameError("Profile type has not been selected. "
                             "PLEASE SPECIFY Vp OR Vs")

        self.profile = profile
        self.profile_type = profile_type
        self.region_name = region_name
        self.method = seismic_method_name
        self.geotherm = geotherm
        
        # method type from file string (e.g. refraction, reflection, RF etc.)
        # otherwise input argument required
        if self.method is None:
            try:
                self.method = self.profile.split(os.path.sep)[-3]
            except IndexError:
                raise NameError("Error parsing file string. "
                                "Please use the correct file structure or "
                                "set a seismic_method_name and/or "
                                "region_name")
        
        # regional location from file string (e.g. africa, north_america, etc.)
        # otherwise input argument required
        if self.region_name is None:
            try:
                self.method = self.profile.split(os.path.sep)[-5]
            except IndexError:
                raise NameError("Error parsing file string. "
                                "Please use the correct file structure or "
                                "set a seismic_method_name and/or "
                                "region_name")

    def read_data(self):
        """
        Read in data file and parse it into a data dictionary.

        Reads the seismic profile data file specified by `self.profile`, 
        extracts relevant information, and organizes it into a dictionary 
        stored in `self.data`. This method handles the following tasks:
        
        1. Extracts method type from the file path (e.g., refraction, 
            reflection, RF).
        2. Parses the record header data, including station name, location 
            coordinates, and Moho depth.
        3. Converts depth-velocity data into a 2D array and reorders columns.
        4. Ensures the velocity profile extends to the Moho depth.
        5. Checks for any velocity step at the Moho depth and removes it 
            so that we do not sample mantle velocities.
        6. Corrects for potential data capture errors by ensuring depth is 
            monotonically increasing.
        7. Calculates the average velocity for the profile.
        8. Constructs a data dictionary based on the profile type (Vs or Vp) 
            and stores it in `self.data`.

        Returns:
            None: The parsed data is stored in `self.data` for further use.

        Note:
            This method assumes that `self.profile_type` has been set to 
            "Vs" or "Vp" to indicate the type of seismic profile being read.
        """
        data = read_file(self.profile)        

        # record header data as variables
        station = data[0][0]
        loc = np.array([float(data[1][0]), float(data[1][1])])
        moho = float(data[2][0])
        # return moho as self variable - useful later on...
        self.moho = moho
        
        # make z, v(z) array up to moho depth
        # convert to array and switch columns
        v_array = np.array(data[3:]).astype(float)[:,[1, 0]]

        # cutoff v profile at moho
        z_array = v_array[:,0]
        z_array = z_array[z_array >= -moho]
        # stack z and v(z) back together
        v_array = np.column_stack((z_array, v_array[:,1][:len(z_array)]))

        # ensure v profile extends to moho
        if v_array[:,0][-1] != -moho:
            final_entry = np.array([-moho, v_array[:,1][-1]])
            np.append(v_array, final_entry)
            v_array = np.append(v_array, final_entry).reshape(
                int((len(v_array)*2 + 2)/2.), 2)
        
        # check that there is not a velocity step at the moho
        #   (i.e. we don't want mantle velocites!)
        if v_array[:,0][-1] == v_array[:,0][-2]:
            v_array = v_array[0:-1]
            
        # check again just to be sure.....
        if v_array[:,0][-1] == v_array[:,0][-2]:
            v_array = v_array[0:-1]
        
        # ensure z monotonically increasing - correct for data capture errors
        # fix step functions for integration and interpolation
        for i in range(len(v_array) - 1):
            if i > 0:
                if v_array[:,0][i] >=  v_array[:,0][i-1]:
                    # add very small value (10 m) to depth to ensure 
                    # increasing depth
                    # don't do this at the moho! 
                    # NB. (this is the reason for iterating up to 
                    # len(v_array) - 1
                    v_array[:,0][i] = v_array[:,0][i-1] - 0.01
                    if i == len(v_array) - 1:
                        break
        
        
        # calculate average velocity
        av_V  = integrate.trapz(-v_array[:,1], v_array[:,0]) / moho
        
        # return data dictionary to self - check type of profile
        if self.profile_type == "Vs":
            self.data = {"station": station, "Vs_file": self.profile, 
                         "region": self.region_name, 
                         "moho": moho, "location": loc, "av_Vs": av_V,
                         "Vs": v_array, "type": "Vs", "method": self.method}
        elif self.profile_type == "Vp":
            self.data = {"station": station, "Vp_file": self.profile, 
                         "region": self.region_name, 
                         "moho": moho, "location": loc, "av_Vp": av_V,
                         "Vp": v_array, "type": "Vp", "method": self.method}
    
    # convert Vs profile into Vp profile
    # using Brocher's (2005) approach
    def convert_profile_brocher(self):
        """
        Convert Vs profile to Vp profile using Brocher's (2005) approach.

        This method converts a seismic Vs profile to a Vp profile using 
        Brocher's (2005) approach. It updates the `self.data` dictionary to 
        include the `Vp_calc` profile field, average Vp, and Vp/Vs ratio.

        Raises:
            Exception: If the profile type is not "Vs," indicating that you 
                must be using a Vs profile for conversion. In such cases, it 
                advises the user to re-initiate the class.
            NameError: If the `data` dictionary has not been created yet, it 
                reminds the user to run `read_data` first to create the data 
                dictionary.

        Returns:
            None
        """

        # catch if you are trying to convert a Vp profile...
        if self.profile_type != "Vs":
            raise Exception("Must be using a Vs profile to convert! "
                            "Re-initiate class!")
        # catch if data dictionary hasn't been created yet...
        try:
            self.data
        except NameError:
            print("Please run read_data first to create data dictionary!")
        
        # convert profile
        Vp = np.column_stack((self.data["Vs"][:,0], 
                              Vs2Vp_brocher(self.data["Vs"][:,1])))
        # add Vs profile, average Vs and Vp/Vs ratio to dictionary
        av_Vp = integrate.trapz(-Vp[:,1], Vp[:,0]) / self.data["moho"]
        # add new values to data dictionary
        self.data.update(Vp_calc = Vp, av_Vp_calc = av_Vp, 
                         Vp_calc_Vs = (av_Vp/self.data["av_Vs"]))
    
    def Vp_to_density_brocher(self):
        """
        Convert Vp profile to density using Brocher (2005) method.

        This method converts a seismic Vp profile to density using the 
        Brocher (2005) method. It checks whether a Vp or calculated Vp 
        profile exists and performs the conversion accordingly. The resulting 
        density fields are added to the `self.data` dictionary.

        Raises:
            KeyError: If no Vp or calculated Vp profile is found when 
                converting a Vs profile. In such cases, it reminds the user 
                to convert Vs to Vp first.

        Returns:
            None
        """

        # check velocity profile type and carry out the appropriate conversion
        if self.profile_type == "Vp":
            # convert profile
            rho = np.column_stack((self.data["Vp"][:,0], 
                                   Vp2rho_brocher(self.data["Vp"][:,1])))
        elif self.profile_type == "Vs":
            try:
                self.data["Vp_calc"]
            except KeyError:
                raise NameError("You haven't created a Vp array yet! "
                      "Convert Vs to Vp first!")
            rho = np.column_stack((self.data["Vp_calc"][:,0], 
                                   Vp2rho_brocher(self.data["Vp_calc"][:,1])))
        # add Vs profile, average Vs and Vp/Vs ratio to dictionary
        av_rho = integrate.trapz(-rho[:,1], rho[:,0]) / self.data["moho"]
        # add new values to data dictionary
        self.data.update(rho = rho, av_rho = av_rho)

    def V_to_density_stephenson(self, parameters, 
                                dz=0.1,
                                constant_depth = None,
                                constant_density = None,
                                T_dependence = True,
                                plot = False):
        """
        Convert seismic velocity profiles to density using the Stephenson 
        method.

        This function takes seismic velocity profiles and converts them 
        to density profiles using the Stephenson density conversion 
        approach. It provides options for specifying conversion parameters, 
        depth increment, constant density values, temperature dependence, 
        and plotting.  The resulting density fileds are added to the 
        'self.data' dictionary.

        Args:
            parameters (Constants or TemperatureDependentConstants class 
                instance): Parameters for density conversion.
            profile (str, optional): The seismic profile type, "Vp" (default) 
                or "Vs".
            dz (float, optional): Depth increment for density calculation 
                (default is 0.1 km).
            constant_depth (float, optional): Depth for constant density (km).
            constant_density (float, optional): Constant density value.
            T_dependence (bool, optional): Include temperature dependence 
                (default is True).
            plot (bool, optional): Whether to plot the density and pressure 
                profiles (default is False).

        Returns:
            None
        """
        
        check_arguments(T_dependence, constant_depth, constant_density,
                            "stephenson", parameters, self.geotherm)

        # instantiate convert profile calss
        ProfileConvert = V2RhoStephenson(self.data,
                                         parameters, 
                                         self.profile_type,
                                         constant_depth,
                                         constant_density,
                                         T_dependence,
                                         self.geotherm)
            
        
        data_converted = ProfileConvert.calculate_density_profile(dz)

    # update class with new fields
        # update class instance with new variables
        # check for T dependence so that we only append T array if it exists
        if T_dependence is True:
            if self.profile_type == "Vp":
                self.data.update(rho = data_converted["rho"], 
                            av_rho = data_converted["av_rho"], 
                            rho_hi_res = data_converted["rho_hi_res"],
                            Vp_hi_res = data_converted["Vp_hi_res"], 
                            p = data_converted["p"], 
                            T = data_converted["T"])
            else:
                self.data.update(rho = data_converted["rho"], 
                            av_rho = data_converted["av_rho"], 
                            rho_hi_res = data_converted["rho_hi_res"],
                            Vs_hi_res = data_converted["Vs_hi_res"], 
                            p = data_converted["p"], 
                            T = data_converted["T"])
        else:
            if self.profile_type == "Vp":
                self.data.update(rho = data_converted["rho"], 
                            av_rho = data_converted["av_rho"], 
                            rho_hi_res = data_converted["rho_hi_res"],
                            Vp_hi_res = data_converted["Vp_hi_res"], 
                            p = data_converted["p"])
            else:
                self.data.update(rho = data_converted["rho"], 
                            av_rho = data_converted["av_rho"], 
                            rho_hi_res = data_converted["rho_hi_res"],
                            Vs_hi_res = data_converted["Vs_hi_res"],
                            p = data_converted["p"])    
    
        # plot profiles?
        if plot == True:
            plt.plot(Z_arr, p_arr)
            plt.show()
                
            plt.plot(rho[:,1], rho[:,0], color='blue')
            plt.plot(rho_arr, -Z_arr, color='orange')
            plt.show()

    # write out data
    def write_data(self, path, file_structure = None, approach="stephenson",
                   T_dependence = False):
        """
        Write seismic profile data to appropriate file locations.

        This method takes the seismic profile data stored in the class 
        instance and writes it to the correct file locations based on the 
        specified density conversion approach and temperature dependence 
        settings.

        Args:
            path (str): The path to the directory containing all velocity 
                data.  See README for directory structure details.
            file_structure (str): If set to None (Default) then we will 
                construct the path from the information in 
                the metadata (i.e. following the default file structure).  
                Otherwise a manual outpath needs to be used that leads to the
                output location.  We will then append relevant method 
                information to the output filename to bookmark the output 
                profiles.
            approach (str, optional): The density (and Vp-Vs if used) 
                conversion approach, which is needed for the file path 
                (default is "stephenson").
            T_dependence (bool, optional): Specifies whether temperature 
                dependence is included (default is False).

        Returns:
            None
        """
        
        # create lists to write out
        outlist_Vp = []
        outlist_rho = []
        
        if T_dependence is True:
            approach = approach + "_T_DEPENDENT"

        # check data type tp save in correct place...
        if self.profile_type == "Vs":
            # check if calculated Vp array exists
            if "Vp_calc" in self.data:
                outlist_Vp.append(self.data["station"])
                outlist_Vp.append(list(self.data["location"]))
                outlist_Vp.append([self.data["moho"]])
                outlist_Vp.append(self.data["Vp_calc"])
                # write output file
                if file_structure is None:
                    outpath = (path + self.data["region"] 
                               + "/Vp/" + self.data["method"] 
                               + "/CALCULATED_" + approach 
                               + os.path.sep 
                               + self.data["station"] 
                               + ".dat")
                else:
                    outpath = (path + os.path.sep 
                               + self.data["station"]
                               + "_Vp_CALCULATED_"
                               + self.data["method"] 
                               + ".dat")
                # check the output directory exists - make it if not
                ensure_dir(outpath)
                write_profile_to_file(outlist_Vp, outpath)
                
            # check if density array exists
            if "rho" in self.data:
                outlist_rho.append(self.data["station"])
                outlist_rho.append(list(self.data["location"]))
                outlist_rho.append([self.data["moho"]])
                outlist_rho.append(self.data["rho"])
                # write output file
                if file_structure is None:
                    outpath = (path + self.data["region"] 
                               + "/vs_rho_" + approach 
                               + os.path.sep 
                               + self.data["method"] 
                               + os.path.sep + self.data["station"] 
                               + ".dat")
                else:
                    outpath = (path + os.path.sep 
                               + self.data["station"]
                               + "_Vs_rho_"
                               + approach + ".dat")
                # check the output directory exists - make it if not
                ensure_dir(outpath)
                write_profile_to_file(outlist_rho, outpath)
            
        if self.profile_type == "Vp":
            if "rho" in self.data:
                outlist_rho.append(self.data["station"])
                outlist_rho.append(list(self.data["location"]))
                outlist_rho.append([self.data["moho"]])
                outlist_rho.append(self.data["rho"])
                # write output file
                if file_structure is None:
                    outpath = (path + self.data["region"] 
                               + "/vp_rho_" + approach 
                               + os.path.sep 
                               + self.data["method"] 
                               + os.path.sep + self.data["station"] 
                               + ".dat")
                else:
                    outpath = (path + os.path.sep 
                               + self.data["station"]
                               + "_Vp_rho_"
                               + approach + ".dat")
                # check the output directory exists - make it if not
                ensure_dir(outpath)
                write_profile_to_file(outlist_rho, outpath)

###################################################################

class V2RhoStephenson:
    """
    Class for density conversion of a given seismic profile using the 
    stephenson method.  Only handles single profiles.
    If you are converting a single profile, it is easiest to use the 
    functionality in this class through the convert_V_profile function.
    If you are converting multiple profiles, then it is easiest to access
    the functionality in this class through the MultiConversion class.

    Args:
        data (dict): Seismic profile data containing depth, velocity,
            moho depth etc.  This is the output from the read_data method
            of the Convert class.
        parameters (numpy.ndarray): Parameters for density conversion.
        profile (str, optional): Profile type, "Vp" or "Vs" (default is "Vp").
        constant_depth (float, optional): Depth for constant density for
            uppermost x km (km).  Must be set if constant_density is set.
        constant_density (float, optional): Constant density value to be 
            assigned to uppermost x km.  Must be set if constant_depth 
            is set.
        T_dependence (bool, optional): Include temperature dependence.
        T_parameters (list, optional): Parameters for temperature-dependent
            density.  Must be set if T_dependence is True.

    Attributes:
        data (dict): Seismic profile data containing depth, velocity, and 
            density.  This is the output from the read_data method
            of the Convert class.
        parameters (numpy.ndarray): Parameters for velocity-to-density 
            conversion.
        profile (str): Profile type, "Vp" or "Vs."
        constant_depth (float): Depth for constant density.
        constant_density (float): Constant density value.
        T_dependence (bool): Temperature dependence inclusion.
        T_parameters (list): Parameters for temperature-dependent density.

    Methods:
        calculate_density_profile(dz=0.1): Calculate density profile.
        _set_up_arrays(dz): Set up depth and velocity arrays.
        _overburden_density(rho_arr, z_arr): Calculate overburden density.
        _pressure(rho_overburden, z): Calculate pressure using overburden 
            density.
        _thermal_expansion_compression(rho_0, T, P): Calculate thermal 
            expansion and compression.
        _calculate_density_pressure(z_arr, V_arr=None, rho_arr=None, 
            T_arr=None, dz=0.1):
            Calculate density as a function of depth.

    """

    def __init__(self, data, parameters, 
                profile="Vp",
                constant_depth = None,
                constant_density = None,
                T_dependence = True,
                geotherm = None):

        self.data = data
        self.profile = profile
        self.constant_depth = constant_depth
        self.constant_density = constant_density
        self.T_dependence = T_dependence
        self.geotherm = geotherm

        # check that constants class instance matches the profile type
        #  i.e. make sure that Vp profile is matched with Vp constants.
        if isinstance(parameters, c.VpConstants):
            self.parameters = parameters
            if self.T_dependence is True:
                raise ValueError("You selected T_dependence = True but have"
                            " not created a material_constants instance"
                            " of the Constants class.")
            if self.profile != "Vp":
                raise ValueError("incompatible constants and profile types"
                                 " please check that profile argument and"
                                 " constants are both Vp.")
        
        # check the same for VsConstants instance
        elif isinstance(parameters, c.VsConstants):
            self.parameters = parameters
            if self.T_dependence is True:
                raise ValueError("You selected T_dependence = True but have"
                            " not created a material_constants instance"
                            " of the Constants class.")
            if self.profile != "Vs":
                raise ValueError("incompatible constants and profile types"
                                 " please check that profile argument and"
                                 " constants are both Vs.")
        
        # check that if we are using the Constants class that we select 
        #   the correct set of constants
        if isinstance(parameters, c.Constants):
            if self.profile == "Vp":
                self.parameters = parameters.vp_constants
            elif self.profile == "Vs":
                self.parameters = parameters.vs_constants

        # if using the Constants class and material constants are set
        # then set materials_constants as this class instance
        if (
            self.T_dependence is True 
            and parameters.material_constants is None
            ):
            raise ValueError("You selected T_dependence = True but have"
                            " not created a material_constants instance"
                            " of the Constants class.")
        elif (
            self.T_dependence is True 
            and parameters.material_constants is not None
            ):
            self.material_constants = parameters.material_constants

    def calculate_density_profile(self, dz=0.1):
        """
        Calculate the density profile.

        This method calculates the density profile based on the provided 
        seismic data and parameters, taking into account temperature 
        dependence if specified.

        Parameters:
            dz (float, optional): Depth interval for calculations (default 
                is 0.1 km).

        Returns:
            data (dict): Seismic profile data including density.  Also
                returns as an attribute of the calss.
        """

        # run the density calculation
        z_v_arr = np.array(self._set_up_arrays(dz))
        # get temperature array if needed
        if self.T_dependence is True:
            try:
                T_arr = self.geotherm(z_v_arr[:,0]*1000)
            except ValueError:
                print("You have selected T_dependence = True, but you "
                      "have not passed V2rhoStephenson an instance of the "
                      "Geotherm class.")
            except TypeError:
                if self.geotherm.tc is None:
                    print("Please set the tc attribute when creating "
                          "the geotherm object")
                else:
                    print("unknown error")
        
        # set up P_rho array
        P_rho = np.zeros(np.shape(z_v_arr))

        # loop through depth, velocity and density 
        for i, value in enumerate(z_v_arr):
            # check that we haven't run off the end of the array
            if i == len(z_v_arr):
                break
            else:
                if self.T_dependence is True:
                    P_rho[i] = self._calculate_density_pressure(
                        z_v_arr[:,0][0:i+1], z_v_arr[:,1][0:i+1],
                        P_rho[:,1][0:i+1], T_arr[0:i+1], dz=dz)
                else:
                    P_rho[i] = self._calculate_density_pressure(
                        z_v_arr[:,0][0:i+1], z_v_arr[:,1][0:i+1],
                        P_rho[:,1][0:i+1], dz=dz)

        # bin density array to make it the same resolution as velocity arrays
        rhoz_arr = np.column_stack((-z_v_arr[:,0], P_rho[:,1]))
        rho_binned = convert_to_same_depth_intervals(rhoz_arr,
                                              self.data[self.profile])
        
        # make high resolution velocity and pressure arrays
        Pz_arr = np.column_stack((-z_v_arr[:,0], P_rho[:,0]))
        Vz_arr = np.column_stack((-z_v_arr[:,0], z_v_arr[:,1]))
        
        # calculate average density
        av_rho = integrate.trapz(P_rho[:,1], z_v_arr[:,0]) / self.data["moho"]

        # update class instance with new variables
        # check for T dependence so that we only append T array if it exists
        if self.T_dependence is True:
            Tz_arr = np.column_stack((-z_v_arr[:,0], T_arr))
            if self.profile == "Vp":
                self.data.update(rho = rho_binned, av_rho = av_rho, 
                                 rho_hi_res = rhoz_arr,
                                 Vp_hi_res = Vz_arr, p = Pz_arr, 
                                 T = Tz_arr)
            else:
                self.data.update(rho = rho_binned, av_rho = av_rho, 
                                 rho_hi_res = rhoz_arr,
                                 Vs_hi_res = Vz_arr, p = Pz_arr, 
                                 T = Tz_arr)
        else:
            if self.profile == "Vp":
                self.data.update(rho = rho_binned, av_rho = av_rho, 
                                 rho_hi_res = rhoz_arr,
                                 Vp_hi_res = Vz_arr, p = Pz_arr)
            else:
                self.data.update(rho = rho_binned, av_rho = av_rho, 
                                 rho_hi_res = rhoz_arr,
                                 Vs_hi_res = Vz_arr, p = Pz_arr)
        return self.data

    def _set_up_arrays(self, dz):
        """
        Set up depth and velocity arrays for density calculations.

        Parameters:
            dz (float): Depth interval for calculations.

        Returns:
            numpy.ndarray: Arrays of depth and velocity.
        """
        # extract depth and velocity arrays
        # depth array needs to be increasing for use in this function
        depth = -self.data[self.profile][:,0]
        V = self.data[self.profile][:,1]
        
        # calculate density by downward integration of pressure
        # first interpolate profile
        V_func = interp1d(depth, V, fill_value="extrapolate")
        # create high-resolution depth array to integrate
        Z_arr = np.arange(0, depth[-1] + dz, dz)
        # interpolate high-resolution velocity array
        V_interp = V_func(Z_arr)
        return np.column_stack((Z_arr, V_interp))

    # calculate overburden density
    def _overburden_density(self, rho_arr, z_arr):
        """
        Calculate overburden density.

        Parameters:
            rho_arr (numpy.ndarray): Density array.
            z_arr (numpy.ndarray): Depth array (km).

        Returns:
            float: Overburden density.
        """
        return integrate.trapz(rho_arr, z_arr) / z_arr[-1]

    # calculate pressure using overburden density (in MPa)
    def _pressure(self, rho_overburden, z):
        """
        Calculate pressure using overburden density (in MPa).

        Parameters:
            rho_overburden (float): Overburden density (in Mg/m3).
            z (float): Depth (in km).

        Returns:
            float: Pressure in MPa.
        """
        return p(rho_overburden, z) / 1e6
    
    # calculate thermal expansion and compression using rho_0
    # and thermal expansivity and compressibility parameters
    def _thermal_expansion_compression(self, rho_0, T, P):
        """
        Calculate thermal expansion and compression.

        Parameters:
            rho_0 (float): Initial density (arbitrary units).
            T (float): Temperature (in C).
            P (float): Pressure (in MPa).

        Returns:
            float: density as a function of pressure and temperature.
        """
        return (rho_0 * td.rho_thermal2(rho_0, T, 
                            self.material_constants.alpha0, 
                            self.material_constants.alpha1)[1] 
                      * td.compressibility(rho_0, P, 
                            self.material_constants.K)[1])

    def _calculate_density_pressure(self, z_arr, V_arr = None, 
                                    rho_arr = None, T_arr = None,
                                    dz=0.1):
        """
        Calculate density as a function of depth.

        Parameters:
            z_arr (numpy.ndarray): Depth array (in km).
            V_arr (numpy.ndarray, optional): Velocity array (default is None).
            rho_arr (numpy.ndarray, optional): Density array (default 
                is None).
            T_arr (numpy.ndarray, optional): Temperature array (default 
                is None).
            dz (float, optional): Depth interval for discretisation
                (default is 0.1 km).

        Returns:
            numpy.ndarray: Pressure and density values.
        """
                
        # if first entry return 0.1 for P and calculate surface 
        # density using V
        if len(z_arr) == 1 and self.constant_density is None:
            density_0 = V2rho_stephenson(np.array([0.1, V_arr[0]]), 
                                                self.parameters)
            return np.array([0.1, density_0])
        # else return pre-determined density value for upper x km
        elif len(z_arr) == 1 and self.constant_density is not None:
            return np.array([0.1, self.constant_density])

        # if not using T dependence then set pressure 
        # using constant density
        if (
            self.constant_density is not None 
            and z_arr[-1] < self.constant_depth
        ):
            if self.T_dependence is not True:
                P = self._pressure(self.constant_density, z_arr[-1])
                return np.array([P, self.constant_density])
            elif self.T_dependence is True:
                P = self._pressure(
                    self._overburden_density(rho_arr, z_arr), z_arr[-1])
                # return pressure and density at depth i
                density_i = (self._thermal_expansion_compression(
                             self.constant_density, T_arr[-1], P))
                return np.array([P, density_i])

        elif (self.constant_depth is None 
              or (self.constant_density is not None 
                  and z_arr[-1] >= self.constant_depth)):
            # integration returns zero if array is only 1 in length
            if len(rho_arr) == 2:
                P = self._pressure(rho_arr[-1], dz)
            else:
                # calculate pressure at bottom of column given 
                # overburden density
                P = self._pressure(
                    self._overburden_density(rho_arr, z_arr), z_arr[-1])
        
            if self.T_dependence is True:
                # correct for temperature
                Vc = V_arr[-1] - td.V_T_correction(T_arr[-1], self.parameters.m)
                # calculate surface density of next depth given pressure
                density_0 = V2rho_stephenson(np.array([P, Vc]), 
                                                self.parameters)
                # return pressure and density at depth i
                density_i = (self._thermal_expansion_compression(
                             density_0, T_arr[-1], P))
                return np.array([P, density_i])
            else:
                # calculate surface density uncorrected for temperature
                density_0 = V2rho_stephenson(np.array([P, V_arr[-1]]), 
                                                self.parameters)
                return np.array([P, density_0])

###################################################################

class MultiConversion:
    """
    Wrapper class to extract multiple files from path to send to 
    Convert class using specified density conversion approach.
    Check that all required arguments have been provided.
    
    Attributes:
        path (str): The master directory where all data are stored in 
            directories named after their location.
        which_location (str or list, optional): Determines which 
            locations the user wants to convert. Defaults to "ALL," 
            indicating that all locations will be converted. If specific 
            locations are desired, provide a list of location names.
        write_data (bool, optional): Specifies whether to write the 
            converted data to files. Defaults to False.
        approach (str, optional): The density conversion approach to use. 
            Options are "stephenson" or "brocher." Defaults to 
            "stephenson."
        parameters (class instance, optional): Class instance of the 
            Constants or TemperatureDependentConstants class.  Must be 
            provided if using approach 'stephenson'.  Must be an instance
            of the TemperatureDependentConstants class if T_dependence is
            set to True.
        constant_depth (float, optional): The depth (from the surface) 
            over which to use a constant density value (in kilometers). 
            Defaults to None.
        constant_density (float, optional): The value of the constant 
            density to use for the uppermost few kilometers (in Mg/m3),
            if `constant_depth` is set.  Defaults to None.
        T_dependence (bool, optional): Determines whether to include 
            temperature dependence of velocity to density conversion, 
            including thermal expansion and compressibility. Defaults 
            to False.

    Methods:
        assemble_file_lists(): Assembles file lists and necessary parameters 
            for density conversion based on provided settings.
        send_to_conversion_function(parallel=False): Initiates the density 
            conversion process for each profile and assembles a list of data 
            dictionaries containing velocity, Moho, density information, etc. 
            The method provides an option for parallel processing but 
            currently has limitations due to pickling issues.
    """

    def __init__(self, path, which_location="ALL", 
                           write_data = False, 
                           approach = "stephenson",
                           parameters = None,
                           constant_depth = None,
                           constant_density = None,
                           T_dependence = False):
        """
        Initialize a MultiConversion instance with specified parameters.
        """
        
        self.path = path
        self.which_location = which_location
        self.write_data = write_data
        self.approach = approach
        self.parameters = parameters
        self.constant_depth = constant_depth
        self.constant_density = constant_density
        self.T_dependence = T_dependence

    def assemble_file_lists(self):
        """
        Assemble file lists and necessary parameters for density conversion.

        This method prepares the necessary data and parameters for density 
        conversion by assembling file lists, checking provided arguments, and 
        setting up parameters based on the chosen conversion approach. It 
        also initializes the `convert_metadata` attribute for storing 
        conversion metadata.

        Parameters:
            None

        Returns:
            None

        """
        path = self.path
        which_location = self.which_location
        approach = self.approach
        parameters = self.parameters
        constant_depth = self.constant_depth
        constant_density = self.constant_density
        T_dependence = self.T_dependence

        # check that all the necessary information has been provided
        check_arguments(T_dependence, constant_depth, constant_density,
                        approach, parameters)

        # check you are running the right script
        if which_location == "ALL":
            print("ALL selected, assembling file lists for all profiles...")
            # obtain file list
            locations = glob.glob(path + "/*")
        else:
            print(f"{which_location} selected, assembling file lists "
                  "for all profiles...")
            # set location variable - convert to one element 
            # list if not already a list
            if type(which_location) == str:
                locations = [path + os.path.sep + which_location]
            else:
                locations = which_location

        vs_files_all = []
        vp_files_all = []

        # loop through locations and append data to lists of vp and vs data
        for location in locations:
            print(f"   -- assembling lists for "
                  f"{location.split(os.path.sep)[-1]}")
            # get all observed vs profiles
            vs_files = glob.glob(location + "/Vs/*/DATA/*")
            vs_files_all.append(vs_files)
            # get all observed vp profiles
            vp_files = glob.glob(location + "/Vp/*/DATA/*")
            vp_files_all.append(vp_files)
            
        if approach == "stephenson":
            # set Vp and Vs parameters for density conversion
            # using Constants class in the constants module
            if not parameters.vp_constants:
                raise ValueError("No vp_constants class instance.  Please"
                                  "create a Constants class instance for Vp.")
            if not parameters.vs_constants:
                raise ValueError("No vs_constants class instance.  Please"
                                  "create a Constants class instance for Vs.")

            # get material property parameters from Constants class instance
            if T_dependence is True:
                if not parameters.material_constants:
                    raise ValueError("No material_constants class instance."
                                     " Please create a Constants class"
                                     " instance for material properties.")
            # set material property parameters to None if not using T 
            #    dependence
            else:
                parameters.material_constants = None
        
        # if not using stephenson approach then set all
        # density-related parameters to None.  Note these parameters
        # are None by default, but we can override them internally here
        # if they are set by accident since we do not require them.
        else:
            parameters.vp_constants = None
            parameters.vs_constants = None
            parameters.material_constants = None

        # loop through Vs profiles and convert to Vp and then to density
        print(f"Reading in data and converting to density "
              f"using {approach} approach...")
        
        # set up convert_metadata list that is appended to 
        # in process_file_list function (note lists are mutable so
        # will be updated by _process_file_list method)
        self.convert_metadata = []
        
        # Process Vp files
        self._process_file_list(vp_files_all, "Vp", 
                                parameters, constant_depth, 
                                constant_density, 
                                T_dependence)

        # Process Vs files
        self._process_file_list(vs_files_all, "Vs", 
                                parameters, constant_depth, 
                                constant_density, 
                                T_dependence)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def send_to_conversion_function(self, parallel=False):
        """
        Send data to the velocity conversion function and obtain station 
        profiles.

        This method initiates the velocity conversion process for each 
        profile and assembles a list of data dictionaries containing velocity,
        Moho, density information, etc. The method provides an option for 
        parallel processing but currently has limitations due
        to pickling issues. It works with both parallel and 
        single-processor modes.

        Parameters:
            parallel (bool, optional): Whether to use parallel processing 
                    (default is False).

        Returns:
            np.ndarray: An array containing station profiles for further use.
        
    """
        # carry out velocity conversion for each profile
        # make list of data dictionaries containing velocity, 
        # moho, density info etc.
        # check if we want to use parallel processing
        # doesn't currently work because of pickling bug.  
        # Something not pickleable!
        if parallel is True:
            # make local varable because can't pass class instances to 
            # multiprocessing programs
            convert_metadata_local = self.convert_metadata
            print(convert_metadata_local[0])
            num_proc = os.cpu_count()
            with Manager() as manager:
                station_profiles = manager.list()
                with manager.Pool(num_proc - 1) as p:
                    p.starmap(convert_V_profile, 
                            [(v_profile, station_profiles) 
                                for v_profile in convert_metadata_local])
                station_profiles = list(station_profiles)
        # if only using one processor
        else:
            # convert profiles with progress bar to show progress
            station_profiles = [convert_V_profile(*v_profile) 
                                for v_profile 
                                in progress_bar(self.convert_metadata)]
        
        # convert station dictionary list to array for ease of use later...
        # make station_profiles object that is returned to class as an
        # attribute
        self.station_profiles = np.array(station_profiles)

        return self.station_profiles

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _assemble_params(self, file, profile_type, 
                         location_name, *extra_params):
        """
        Assemble parameters for density conversion.

        This function assembles a list of parameters required for density
        conversion, including file information, profile type, write data 
        option, path, approach, and location name. If the approach is 
        "stephenson," additional extra parameters are included.

        Parameters:
            file (str): The file path being processed.
            profile_type (str): The type of profile being processed 
                    (e.g., 'Vp' or 'Vs').
            location_name (str): The name of the location associated with 
                    the file.
            *extra_params: Variable number of extra parameters specific to 
                    the approach.

        Returns:
            list: A list of assembled parameters for density conversion.

        """
        # assemble parameter list
        params = [file, profile_type, self.write_data, 
                        self.path,
                        self.approach,
                        location_name]

        # add extra parameters if using "stephenson" approach
        if self.approach == "stephenson":
            params.extend(extra_params)
        return params

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _process_file_list(self, file_list, profile_type, 
                           *extra_params):
        """
        Process a list of files to assemble necessary data and metadata.
        
        This function iterates through a list of files and extracts 
        location names from file paths.
        It assembles parameter lists using the provided `profile_type` 
        and any extra parameters.
        The assembled parameters are appended to the `convert_metadata` 
        attribute of the class.

        Parameters:
            file_list (list): A list of file paths to be processed.
            profile_type (str): The type of profile being processed 
                    (e.g., 'Vp' or 'Vs').
            *extra_params: Variable number of extra parameters 
                    specific to the profile type.

        Returns:
            None

    """
        # loop through files in file_list and assemble necessary data 
        # and metadata
        for files in file_list:
            if len(files) > 0:
                # extract location name from file string
                location_name = files[0].split(os.path.sep)[-5]
                # assemble paramter list
                for file in files:
                    params = self._assemble_params(file, profile_type,
                                            location_name,
                                            *extra_params)
                    self.convert_metadata.append(params)

###################################################################

def check_arguments(T_dependence, 
                    constant_depth, 
                    constant_density,
                    approach, 
                    parameters,
                    geotherm = None):
    """
    Check if required arguments are provided and raise errors if not.

    This function checks the provided arguments for the conversion process 
    and raises errors if any required information is missing based on the 
    selected conversion approach.  Prevents conflicting options.

    Args:
        T_dependence (bool): Flag indicating whether to correct for 
            temperature and pressure.
        constant_depth (float): Depth range for constant density from the 
            surface.
        constant_density (float): Constant density value for the uppermost 
            x kilometers.
        approach (str): Density conversion scheme, "brocher" or "stephenson".
        parameters (str): Constants class instance if using the Stephenson 
            scheme.

    Raises:
        ValueError: If required arguments are missing or incompatible with 
            the sefcheck_argumentslected approach.
    """
    # check that all the necessary information has been provided
    if T_dependence is True:
        if isinstance(parameters, c.Constants) is not True:
            raise ValueError("T_dependence is set to True. "
                         "Please use the Constants class and not the "
                         "VpConstants or VsConstants classes to store "
                         "the constants.")
        elif isinstance(parameters, c.Constants) is True:
            if parameters.material_constants is None:
                raise ValueError("T_dependence is set to True but "
                            "material_constants instance of the Constants "
                            "class has not been set.")
        
        if not geotherm:
            raise ValueError("Please create a geotherm object using the `Geotherm` "
                  "class")

    if constant_depth is not None and constant_density is None:
        raise ValueError("constant_depth is set but not \
                         constant_density")
    if constant_density is not None and constant_depth is None:
        raise ValueError("constant_density is set but "
                         "not constant_depth")
    if approach == "stephenson" and parameters is None:
        raise ValueError(f"method: {approach} selected but "
                         "parameter file not provided")

###################################################################

def convert_V_profile(
        file, 
        profile_type, 
        write_data=False, 
        path = None,
        approach = "stephenson",
        location = None,
        parameters = None,
        constant_depth = None,
        constant_density = None,
        T_dependence = False,
        geotherm = None,
        print_working_file = False
        ):

    """
    Convert a single Vp or Vs velocity profile using the chosen scheme.

    This function takes a velocity profile file and converts it to density values
    using the specified approach, which can be "brocher" or "stephenson".

    Args:
        file (str): Path to the input velocity profile file.
        profile_type (str): Type of velocity profile, 'Vp' or 'Vs'.
        write_data (bool, optional): Write out the converted data to files. 
            Default is False.
        path (str, optional): Location to write out the results. Default 
            is None.
        approach (str, optional): Density conversion scheme to use, "brocher" 
            or "stephenson".
        location (str, optional): Regional location of the profile. Default 
            is "None".
        parameters (np.ndarray, optional): Parameter array if using the 
            Stephenson scheme for velocity-density conversion.  A single 
            column (1D) array.  Not Vp and Vs parameters combined.   
            Default is None.
        constant_depth (float, optional): Depth range for constant density 
            from the surface. Default is None.
        constant_density (float, optional): Constant density value for the 
            uppermost x kilometers. Default is None.
        T_dependence (bool, optional): Correct velocity and density for 
            temperature and pressure. Default is False.
        T_parameters (tuple, optional): Parameters needed to correct for 
            temperature effects (dV/dT, alpha0, alpha1).  Default is None.
        print_working_file (bool, optional): Print the file that is being converted.
            Default is True.

    Returns:
        pd.DataFrame: Pandas DataFrame containing the converted velocity and 
            density data.

    Notes:
        - The function converts velocity profiles to density profiles using 
            the specified approach.
        - It can write out the converted data to files if 'write_data' is 
            set to True.

    Example:
        >>> convert_V_profile('velocity_profile.dat', 'Vp', write_data=True,
        ...                   path='output/', approach='stephenson',
        ...                   location='Region1', parameters=[p1, p2, ...],
        ...                   constant_depth=10.0, constant_density=2.7,
        ...                   T_dependence=True, 
        ...                   T_parameters=(0.01, 1.0, 0.002, 0.00001, 90e9))

    """

    # check that the program will run -- are all options provided?
    check_arguments(T_dependence, constant_depth, constant_density,
                    approach, parameters, geotherm)

    # keep tabs on the profile file path
    if print_working_file is True:
        print(f"working on {file}")
    else:
        pass

    # initiate conversion class
    Data = Convert(file, 
                   profile_type, 
                   region_name=location,
                   geotherm = geotherm)
    
    # read in data
    Data.read_data()
    
    # check density conversion scheme to use
    # if brocher
    if approach == "brocher":
        # convert vs to vp first
        if profile_type == 'Vs':
            Data.convert_profile_brocher()
        
        # convert the profile using Brocher
        Data.Vp_to_density_brocher()
    
    # if stephenson    
    elif approach == "stephenson":        
        # convert using Stephenson approach
        Data.V_to_density_stephenson(
            parameters, 
            constant_depth = constant_depth,
            constant_density = constant_density,
            T_dependence = T_dependence)
    
    # write out files
    if write_data is True:
        if T_dependence is True:
            Data.write_data(path, approach=approach,
                            T_dependence = True)
        else:
            Data.write_data(path, approach=approach,
                            T_dependence = False)   
    # return station data and information
    return Data.data

###################################################################

def av_profile(profiles, base_depth, average_type = 'median'):
    """
    Calculate the average profile and bulk property profile given a family 
    of profiles.

    This function takes a list of profiles, where each profile is represented 
    as a NumPy array with columns [depth, property]. It calculates the 
    average profile and the bulk property profile as a function of depth.

    Args:
        profiles (list of np.ndarray): List of profiles 
            (e.g., Vp, Vs, P, T, rho, etc.). Each profile must contain at 
            least two columns: depth and property.
        base_depth (float): Depth to which the bulk property should be 
            averaged.
        average_type (str, optional): Method of averaging ('median' or 
            'mean'). Default is 'median'.

    Returns:
        av_z_f (scipy.interpolate.CubicSpline): Average profile as a function 
            of depth.
        bulk_base_depth_f (scipy.interpolate.CubicSpline): Bulk property 
            profile as a function of base depth.

    Example:
        >>> profiles = [np.array([[0.0, 1.5], 
                                  [5.0, 2.0], 
                                  [10.0, 2.5]]),
                        np.array([[0.0, 1.6], 
                                  [5.0, 2.1], 
                                  [10.0, 2.6]])]
        >>> av_z_f, bulk_base_depth_f = av_profile(profiles, base_depth=15.0, 
                                            average_type='mean')

    Notes:
        - This function interpolates the input profiles and calculates the 
            average profile at each depth.
        - It also computes the bulk property profile as a function of base 
            depth.  (e.g. bulk crustal velocity as a function of crustal 
            thickness)
    """
    
    # interpolate profiles and make list of functions
    funcs = []
    for p in profiles:
        p[0][0] = 0.
        func = interp1d(-p[:,0], p[:,1])
        funcs.append(func)
    
    # interpolated z array (make profiles sampled at same frequency)
    z_arr = np.linspace(0, base_depth, int(base_depth*3))
    
    # calculate as a function of crustal thickness
    # first interpolate profiles to make functions for each
    profiles_interp = []
    for f in funcs:
        profile_interp = []
        for z in z_arr:
            try:
                profile_interp.append(f(z))
            except ValueError:
                break
        profiles_interp.append(profile_interp)
    
    # next evaluate profiles and average at each depth in z_arr
    average_list = []
    for zi in range(len(z_arr)):
        z_list = []
        for prof in profiles_interp:
            if len(prof) > zi:
                z_list.append(prof[zi])
        if average_type == "median":
            average_list.append(np.nanmedian(z_list))
        else:
            average_list.append(np.nanmean(z_list))
    
    # average function as function of depth
    av_z_f = CubicSpline(z_arr, np.array(average_list))
    
    # create bulk curve down to base_depth
    bulk_base_depth = [average_list[0]]
    for i in range(len(z_arr)):
        if i < len(z_arr) - 1:
            zi = z_arr[i+1]
            bulk_base_depth.append(
                integrate.trapz(av_z_f(z_arr[0:i+2]), z_arr[0:i+2])/zi)
    
    # interpolate bulk function as function base_depth
    bulk_base_depth_f = CubicSpline(z_arr, bulk_base_depth)
    
    return av_z_f, bulk_base_depth_f


###################################################################

def save_bulk_profiles(outpath, filename, data_function, start, stop, step):
    """
    Save bulk profiles data to a file.

    Args:
    outpath (str): The output directory where the data file will be saved.
    filename (str): The name of the output data file.
    data_function (function): A function that computes the data values 
        based on x-values.
    start (float): The start value for the x-values.
    stop (float): The stop value for the x-values.
    step (float): The step size for generating x-values.

    Returns:
    None

    This function generates x-values using numpy.arange() based on the 
    specified start, stop, and step values.
    It computes the data values using the provided data_function and saves 
    the data to a file in the specified outpath.

    Example usage:
    save_bulk_profiles("/output/directory/", 
                       "bulk_data.txt", bulk_function, 0, 50.5, 0.5)
    """
    x = np.arange(start, stop, step)
    data = np.column_stack((x, data_function(x)))
    np.savetxt(outpath + filename, data)


###################################################################

def convert_to_same_depth_intervals(profile1, profile2):
    """
    Convert profiles to have the same depth intervals.  Takes depth
    as first column and any arbitrary y value as second.  profiles 1 and 2
    need not have the same y field.
    
    This function bins velocity profiles so that they have the same depth 
    ranges.It is suitable for profiles with distinct layers rather than 
    those with gradients, although it is designed to produce a reasonable
    solution in both cases.

    Args:
        profile1 (np.ndarray): The first profile as a NumPy array 
            with columns [depth, velocity].
        profile2 (np.ndarray): The second profile as a NumPy array 
            with columns [depth, velocity].

    Returns:
        np.ndarray: A binned velocity profile aligned with the depth 
            intervals of the lowest-resolution profile.

    Example:
        >>> binned_profile = convert_to_same_depth_intervals(profile1, profile2)
        >>> print(binned_profile)
        array([[ 0. , 1600. ],
               [ 2.5, 1600. ],
               [ 2.5 , 2050. ],
               [10. , 2050. ]])

    Notes:
        - This function finds the lowest-resolution profile and interpolates the 
            other profile accordingly.
        - It returns a binned profile with the same depth intervals as the 
            lowest-resolution profile.
        - It works best when lowest-resoltion profile has discrete layers 
            rather than gradients, although it will return reasonable values
            in both cases.
    """

    # Extract depth and velocity arrays from profile1 and profile2
    bins1, y1 = profile1[:, 0], profile1[:, 1]
    bins2, y2 = profile2[:, 0], profile2[:, 1]

    # Find the lowest-resolution profile
    # set bins_low_res variable based on the lowest resolution profile
    if len(profile1) < len(profile2):
        bins_low_res, y_low_res = bins1, y1
        interp_profile = interp1d(bins2, y2, fill_value="extrapolate")
    else:
        bins_low_res, y_low_res = bins2, y2
        interp_profile = interp1d(bins1, y1, fill_value="extrapolate")

    # start binning profiles based on intervals in lowest-resolution profile
    binned_profile = []
    e = 0
    while e < len(bins_low_res) - 1:
        # Check whether we have single velocity layers or 
        # layers with increasing velocity.
        if np.round(bins_low_res[e], 1) == np.round(bins_low_res[e + 1], 1):
            e += 1
        else:
            # bin high-resultion profile
            h = bins_low_res[e + 1] - bins_low_res[e]
            int_val = integrate.quad(interp_profile, bins_low_res[e], 
                                     bins_low_res[e + 1])[0] / h
            if not np.isnan(int_val):
                binned_profile.append([bins_low_res[e], int_val])
                binned_profile.append([bins_low_res[e + 1], int_val])
            e += 1

    binned_profile = np.array(binned_profile)
    return binned_profile

###################################################################

# split data with no velocity profile into list of 
# dictionaries with data for each station
def read_no_profile_data(data_file, region_name = None):
    """
    Read crustal data from a database of crustal thickness estimates and 
    vp/vs ratios.
    
    Args:
        data_file (str): Path to the data file to be read.
        region_name (str, optional): Optional regional location of the 
        profile.

    Returns:
        pd.DataFrame: A Pandas DataFrame containing the following columns:
        
        - 'region' (str): Regional location of the profile (if provided, 
            otherwise None).
        - 'lon' (float): Longitude of the station.
        - 'lat' (float): Latitude of the station.
        - 'moho' (float): Crustal thickness estimate.
        - 'vp_vs' (float): Ratio of seismic velocities (vp/vs).

    Example:
        >>> df = read_no_profile_data('crustal_data.csv', 
                region_name='Example_Region')
        >>> print(df.head())
            region     lon     lat   moho   vp_vs
        0  Example Region  -75.56   40.21  36.58  1.73
        1  Example Region  -72.89   42.17  38.12  1.68
        2  Example Region  -80.12   35.69  33.45  1.80
        3  Example Region  -76.55   39.29  40.27  1.64
        4  Example Region  -74.21   41.54  35.78  1.75
    """
    data_raw = read_file(data_file, delimiter = ",")
    
    data = []
    # loop through entries starting from second entry (skip file header)
    for i in data_raw[1:]:
        for e in range(len(i)):
            if i[e] == "-" or i[e] == "-\n" or i[e] == "":
                i[e] = np.nan

        station_data = {"region": region_name,
                            "lon": float(i[0]), 'lat': float(i[1]), 
                            "moho": float(i[2]), 
                            "vp_vs":float(i[3])}
        data.append(station_data)
    return pd.DataFrame(data)

###################################################################

def p(rho, h, g=9.81):
    """
    Calculate lithostatic pressure given average density of
    overburden and thickness of the overburden.

    Parameters:
        - rho (float): Average density of the overburden in Mg/m³.
        - h (float): Thickness of the overburden in kilometers.
        - g (float, optional): Acceleration due to gravity in m/s². 
            Default is 9.81 m/s².

    Returns:
        float: Lithostatic pressure in Pascals (Pa).
        
    Formula:
        The lithostatic pressure (P) is calculated using the formula:
        P = rho * g * h * 1e6
    """
    
    return rho * g * h * 1e6

###################################################################

class Corrections:
    """
    Calculate corrections for variable crustal densities
    
    Arguments: tc: crustal thicknesses
               rho_av: average observed density of the crust
               rho_a = asthenospheric mantle density
               rho_o = reference density to correct to, 
                       default = 2.85 g/cm^3
    """
    def __init__(self, tc, rho_av, rho_a, rho_o=2.85):
        self.tc, self.rho_av = tc, rho_av
        self.rho_a, self.rho_o = rho_a, rho_o

    def correct_thickness(self):
        """
        Calculate magnitude of the crustal thickness correction for
        a given reference density.   Returns magnitude of correction
        and the corrected thickness.  Accommodates numpy arrays.
        """
        
        tc, rho_av = self.tc, self.rho_av
        rho_a, rho_o = self.rho_a, self.rho_o
        
        drho = rho_av - rho_o
        correction_factor = drho / (rho_a - drho - rho_av)
        
        return -tc * correction_factor, (-tc * correction_factor) + tc

    def correct_elevation(self):
        """
        Calculate magnitude of the elevation correction for
        a given reference density.   Returns magnitude of correction.
        Accommodates numpy arrays.
        
        Arguments: e: elevation array for each station
        """
        
        tc, rho_av = self.tc, self.rho_av
        rho_a, rho_o = self.rho_a, self.rho_o        
        
        de = tc * (rho_av - rho_o) / rho_a
        
        return de

###################################################################

def progress_bar(iterable, length=50):
    """
    Create a text-based progress bar to track the progress of an iterable.

    Parameters:
    iterable (iterable): The iterable object (e.g., list, range) 
    to iterate through.
    length (int, optional): The length of the progress bar. Defaults to 50.

    Yields:
    item: The current item from the iterable.

    Example:
    >>> for item in progress_bar(range(100)):
    ...     # Simulate some processing
    ...     time.sleep(0.1)
    ...
    [=====================>                   ] 50%
    """
    total = len(iterable)  # Get the total number of items in the iterable
    bar_length = length   # Define the length of the progress bar

    # Iterate through the items in the iterable
    for i, item in enumerate(iterable):
        progress = (i + 1) / total  # Calculate the progress as a ratio
        # Create the progress arrow
        arrow = '=' * int(round(progress * bar_length))  
        # Calculate spaces to fill the progress bar
        spaces = ' ' * (bar_length - len(arrow))  

        # Update the progress bar in the same line using carriage return (\r)
        sys.stdout.write(f'\r[{arrow}{spaces}] {int(progress * 100)}%')
        # Flush the standard output to display the progress immediately
        sys.stdout.flush() 
        # Yield the current item from the iterable
        yield item