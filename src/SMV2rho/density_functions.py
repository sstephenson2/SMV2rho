#!/usr/bin/env python3

# Functions to convert Vs to Vp and Vp to density

# @Author: Simon Stephenson @Date: Aug 2023

###################################################################
# import modules
import numpy as np
import sys
import copy
import scipy.integrate as integrate
from scipy.interpolate import interp1d, CubicSpline
import glob
import os, os.path
import SMV2rho.temperature_dependence as td
import SMV2rho.constants as c
import pandas as pd
import matplotlib.pyplot as plt

###################################################################
# read/write data

# ensure directory exists
def ensure_dir(filename):
    """
    Ensure that the directory for the given filename exists. If the directory 
    does not exist, it will be created.

    Parameters
    ----------
    filename : str
        The path to the file.

    Returns
    -------
    None
    """
    d = os.path.dirname(filename)
    if not os.path.exists(d):
        os.makedirs(d)    

# read in files
def read_file(filename, delimiter = None):
    """
    Read the contents of a file and return them as a list of lines.

    Parameters
    ----------
    filename : str
        The path to the file.
    delimiter : str, optional
        The delimiter used to split lines into elements (default is None, 
        which means lines are not split).

    Returns
    -------
    list
        A list of lines from the file.
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

    Parameters
    ----------
    data : list
        A list of data to be written to the file.
    filename : str
        The path to the output file.

    Returns
    -------
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

    Parameters
    ----------
    Vs : float or numpy.ndarray
        Shear wave velocity in km/s.

    Returns
    -------
    float or numpy.ndarray
        Compressional wave velocity (Vp) in km/s.
    """

    return (0.9409 + 2.0947 * Vs - 0.8206 * 
            Vs**2. + 0.2683 * Vs**3. - 0.0251 * Vs**4.)

# Nafe-Drake relationship - converts Vp to density (g/cm3)
def Vp2rho_brocher(Vp):
    """
    Convert compressional wave velocity (Vp) to density (g/cm³) using the 
    Nafe-Drake relationship.

    Parameters
    ----------
    Vp : float or numpy.ndarray
        Compressional wave velocity in km/s.

    Returns
    -------
    float or numpy.ndarray
        Density in g/cm³.
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

    Parameters
    ----------
    data : array-like
        An array of pressure and velocity data where data[0] is pressure 
        (in GPa) and data[1] is velocity (in km/s).
    parameters : class instance
        Instance of the VpConstants or VsConstants classes containing the 
        attributes listed below: 

        - v0 (float): Intercept velocity.
        - b (float): Velocity gradient with respect to density.
        - d0 (float): Partial derivative of velocity gradient (dvdr) with 
            respect to pressure (dp).
        - dp (float): Velocity gradient with respect to pressure (dvdpr).
        - c (float): Amplitude of the velocity drop-off at low pressure.
        - k (float): Lengthscale of the velocity drop-off.

    Returns
    -------
    float or numpy.ndarray
        Density at standard temperature and pressure (s.t.p.) in g/cm³.
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
    Convert seismic velocity profiles to various parameters.

    This class provides methods to read seismic velocity profiles, convert 
    them to other parameters using different approaches, and write the 
    converted data to output files.

    Parameters
    ----------
    profile : str
        File path to the seismic profile data.
    profile_type : str
        Type of the seismic profile, "Vp" or "Vs."
    region_name : str, optional
        Geographic location of the profile. Default is None.
    seismic_method_name : str
        Method used to acquire the velocity profile. If set to None, the 
        read_data method will pick up the argument from the file string. 
        Note if set to None the strict file convention must be set 
        (see README.md) and tutorial_1.ipynb.
    geotherm : class instance
        Instance of the Geotherm class containing information about the 
        temperature profile at the site of the seismic profile.

    Attributes
    ----------
    data : dict
        Dictionary containing parsed seismic profile data.
    profile_type : str
        Type of profile.  "Vp" or "Vs".
    moho : float
        Moho depth parsed from the profile data.
    geotherm : class instance
        Instance of the Geotherm class (if using temperature-dependent 
        conversion. Default None)
    region_name : str
        Geographic location of the file (parsed from file string or given 
        as argument)
    method : str
        Method used to collect the velocity profile.  Parsed from file 
        string or given as argument (e.g. 'REFRACTION').
    rho, av_rho, rho_hi_res : various types
        Attributes generated by the Vs_to_Vp_brocher, Vp_to_density_brocher, 
        and V_to_density_stephenson methods.
    vp_calc : np.ndarray
        Calculated vp profile generated by Vs_to_Vp_brocher method.
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

        This method reads the seismic profile data file specified by `self.profile`, 
        extracts relevant information, and organizes it into a dictionary 
        stored in `self.data`. 

        The method handles the following tasks:
        
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

        Returns
        -------
        None
            The parsed data is stored in `self.data` for further use.

        Notes
        -----
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

        # ensure tc in the geotherm class matches moho value
        if self.geotherm is not None:
            if (hasattr(self.geotherm, 'master') 
            and self.geotherm.master is True
            ):
                # deepcopy in order to make sure the geotherm.tc values do not
                # end up referencing each other for all profiles converted in
                # the same batch.  Only a problem if using the 
                # MultiConversion class.
                self.geotherm = copy.deepcopy(self.geotherm)
                self.geotherm.tc = copy.deepcopy(self.moho)

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
                         "Vs": v_array, "type": "Vs", "method": self.method,
                         "geotherm": self.geotherm}
        elif self.profile_type == "Vp":
            self.data = {"station": station, "Vp_file": self.profile, 
                         "region": self.region_name, 
                         "moho": moho, "location": loc, "av_Vp": av_V,
                         "Vp": v_array, "type": "Vp", "method": self.method,
                         "geotherm": self.geotherm}
    
    # convert Vs profile into Vp profile
    # using Brocher's (2005) approach
    def Vs_to_Vp_brocher(self):
        """
        Convert Vs profile to Vp profile using Brocher's (2005) approach.

        This method converts a seismic Vs profile to a Vp profile using 
        Brocher's (2005) approach. It updates the `self.data` dictionary to 
        include the `Vp_calc` profile field, average Vp, and Vp/Vs ratio.

        Raises
        ------
        Exception
            If the profile type is not "Vs," indicating that you must be using 
            a Vs profile for conversion. In such cases, it advises the user to 
            re-initiate the class.
        NameError
            If the `data` dictionary has not been created yet, it reminds the 
            user to run `read_data` first to create the data dictionary.

        Returns
        -------
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

        Raises
        ------
        KeyError
            If no Vp or calculated Vp profile is found when converting a Vs 
            profile. In such cases, it reminds the user to convert Vs to Vp 
            first.

        Returns
        -------
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

    def V_to_density_stephenson(
            self, 
            parameters, 
            dz=0.1,
            constant_depth = None,
            constant_density = None,
            T_dependence = True,
            plot = False
            ):
        """
        Convert seismic velocity profiles to density using the Stephenson 
        method.

        This method takes seismic velocity profiles and converts them to 
        density profiles using the Stephenson density conversion approach. It 
        provides options for specifying conversion parameters, depth 
        increment, constant density values, temperature dependence, and 
        plotting. The resulting density fields are added to the 'self.data' 
        dictionary.

        Parameters
        ----------
        parameters : Constants or TemperatureDependentConstants class instance
            Parameters for density conversion.
        dz : float, optional
            Depth increment for density calculation (default is 0.1 km).
        constant_depth : float, optional
            Depth for constant density (km).
        constant_density : float, optional
            Constant density value.
        T_dependence : bool, optional
            Include temperature dependence (default is True).
        plot : bool, optional
            Whether to plot the density and pressure profiles (default is 
            False).

        Returns
        -------
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
            plt.plot(data_converted['p'][:, 1], 
                     data_converted['p'][:, 0])
            plt.show()
                
            plt.plot(data_converted['rho'][:, 1], 
                     data_converted['rho'][:, 0], 
                     color='blue')
            plt.plot(data_converted['rho_hi_res'][:, 1], 
                     data_converted['rho_hi_res'][:, 0], 
                     color='orange')
            plt.show()

    # write out data
    def write_data(self, 
                   path, 
                   file_structure=None, 
                   approach="stephenson", 
                   T_dependence=False):
        """
        Write seismic profile data to appropriate file locations.

        This method takes the seismic profile data stored in the class 
        instance and writes it to the correct file locations based on the 
        specified density conversion approach and temperature dependence 
        settings.

        Parameters
        ----------
        path : str
            The path to the directory containing all velocity data. See README 
            for directory structure details.
        file_structure : str, optional
            If set to None (Default) then we will construct the path from the 
            information in the metadata (i.e. following the default file 
            structure). Otherwise a manual outpath needs to be used that leads 
            to the output location. We will then append relevant method 
            information to the output filename to bookmark the output 
            profiles.
        approach : str, optional
            The density (and Vp-Vs if used) conversion approach, which is 
            needed for the file path (default is "stephenson").
        T_dependence : bool, optional
            Specifies whether temperature dependence is included (default is 
            False).

        Returns
        -------
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
    Stephenson method. Only handles single profiles.

    If you are converting a single profile, it is easiest to use the 
    functionality in this class through the convert_V_profile function.
    If you are converting multiple profiles, then it is easiest to access
    the functionality in this class through the MultiConversion class.

    Parameters
    ----------
    data : dict
        Seismic profile data containing depth, velocity, and density. This 
        is the output from the read_data method of the Convert class.
    parameters : numpy.ndarray
        Instance of the Constants, VpConstants or VsConstants class. If 
        T_dependence is True, then this must be an instance of Constants 
        class and must contain material_constants attribute.
    profile : str
        Profile type, "Vp" or "Vs."
    constant_depth : float
        Depth for constant density for uppermost x km.
    constant_density : float
        Constant density value for uppermost x km.
    T_dependence : bool
        True for temperature dependence inclusion (default True).
    geotherm : list
        Instance of the Geotherm class, must be set if T_dependence is True 
        (default None).

    Attributes
    ----------
    data : dict
        Seismic profile data containing depth, velocity, and density. This 
        is the output from the read_data method of the Convert class.
    parameters : numpy.ndarray
        Instance of the Constants, VpConstants or VsConstants class. If 
        T_dependence is True, then this must be an instance of Constants 
        class and must contain material_constants attribute.
    profile : str
        Profile type, "Vp" or "Vs."
    constant_depth : float
        Depth for constant density for uppermost x km.
    constant_density : float
        Constant density value for uppermost x km.
    T_dependence : bool
        True for temperature dependence inclusion (default True).
    geotherm : list
        Instance of the Geotherm class, must be set if T_dependence is True 
        (default None).
    """

    def __init__(
        self, 
        data, 
        parameters, 
        profile="Vp",
        constant_depth = None,
        constant_density = None,
        T_dependence = True,
        geotherm = None
        ):

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

        Parameters
        ----------
        dz : float, optional
            Depth interval for calculations (default is 0.1 km).

        Returns
        -------
        data : dict
            Seismic profile data including density. Also returns as an 
            attribute of the class.
        """

        # run the density calculation
        z_v_arr = np.array(self._set_up_arrays(dz))
        # get temperature array if needed
        if self.T_dependence is True:
            try:
                T_arr = self.geotherm(z_v_arr[:,0])
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

        Parameters
        ----------
        dz : float
            Depth interval for calculations.

        Returns
        -------
        numpy.ndarray
            Arrays of depth and velocity.
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

        Parameters
        ----------
        rho_arr : numpy.ndarray
            Density array.
        z_arr : numpy.ndarray
            Depth array (km).

        Returns
        -------
        float
            Overburden density.
        """
        return integrate.trapz(rho_arr, z_arr) / z_arr[-1]

    # calculate pressure using overburden density (in MPa)
    def _pressure(self, rho_overburden, z):
        """
        Calculate pressure using overburden density (in MPa).

        Parameters
        ----------
        rho_overburden : float
            Overburden density (in Mg/m3).
        z : float
            Depth (in km).

        Returns
        -------
        float
            Pressure in MPa.
        """
        return p(rho_overburden, z) / 1e6
    
    # calculate thermal expansion and compression using rho_0
    # and thermal expansivity and compressibility parameters
    def _thermal_expansion_compression(self, rho_0, T, P):
        """
        Calculate thermal expansion and compression.

        Parameters
        ----------
        rho_0 : float
            Initial density (arbitrary units).
        T : float
            Temperature (in C).
        P : float
            Pressure (in MPa).

        Returns
        -------
        float
            Density as a function of pressure and temperature.
        """
        return (rho_0 * td.rho_thermal2(rho_0, T, 
                            self.material_constants.alpha0, 
                            self.material_constants.alpha1)[1] 
                      * td.compressibility(rho_0, P, 
                            self.material_constants.K)[1])

    def _calculate_density_pressure(self, 
                                    z_arr, 
                                    V_arr=None, 
                                    rho_arr=None, 
                                    T_arr=None, 
                                    dz=0.1):
        """
        Calculate density as a function of depth.

        Parameters
        ----------
        z_arr : numpy.ndarray
            Depth array (in km).
        V_arr : numpy.ndarray, optional
            Velocity array (default is None).
        rho_arr : numpy.ndarray, optional
            Density array (default is None).
        T_arr : numpy.ndarray, optional
            Temperature array (default is None).
        dz : float, optional
            Depth interval for discretisation (default is 0.1 km).

        Returns
        -------
        numpy.ndarray
            Pressure and density values.
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
    
    Parameters
    ----------
    path : str
        The master directory where all data are stored in 
        directories named after their location.
    which_location : str or list, optional
        Determines which locations the user wants to convert. 
        Defaults to "ALL," indicating that all locations will be 
        converted. If specific locations are desired, provide a 
        list of location names.
    write_data : bool, optional
        Specifies whether to write the converted data to files. 
        Defaults to False.
    approach : str, optional
        The density conversion approach to use. Options are 
        "stephenson" or "brocher." Defaults to "stephenson."
    parameters : class instance, optional
        Class instance of the Constants class. Must be provided 
        if using approach 'stephenson'. Must contain a 
        material_constants attribute if T_dependence is True.
    master_geotherm : instance of Geotherm class, optional
        Used as a reference or template for other operations. 
        When the `master` attribute of `master_geotherm` is True, 
        deep copies are made and parameters are updated for all 
        individual profiles.
    constant_depth : float, optional
        The depth (from the surface) over which to use a constant 
        density value (in kilometers). Defaults to None.
    constant_density : float, optional
        The value of the constant density to use for the uppermost 
        few kilometers (in Mg/m3), if `constant_depth` is set.  
        Defaults to None.
    T_dependence : bool, optional
        Determines whether to include temperature dependence of 
        velocity to density conversion, including thermal 
        expansion and compressibility. Defaults to False.

    Attributes
    ----------
    path : str
        The master directory where all data are stored.
    which_location : str or list
        The locations to convert.
    write_data : bool
        Whether to write the converted data to files.
    approach : str
        The density conversion approach to use.
    parameters : class instance
        Instance of the Constants class.
    master_geotherm : instance of Geotherm class
        Used as a reference or template for other operations.
    constant_depth : float
        The depth over which to use a constant density value.
    constant_density : float
        The value of the constant density to use for the uppermost 
        few kilometers.
    T_dependence : bool
        Whether to include temperature dependence of velocity to 
        density conversion.
    convert_metadata : list
        Metadata for the conversion process.  This is a list of
        dictionaries that is appended to in the process_file_list
        method.  It is used to store information about the conversion
        process for each profile.
    """

    def __init__(
        self, 
        path, 
        which_location = "ALL", 
        write_data = False, 
        approach = "stephenson",
        parameters = None,
        master_geotherm = None,
        constant_depth = None,
        constant_density = None,
        T_dependence = False
        ):

        """
        Initialize a MultiConversion instance with specified parameters.
        """
        
        self.path = path
        self.which_location = which_location
        self.write_data = write_data
        self.approach = approach
        self.parameters = parameters
        self.master_geotherm = master_geotherm
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
        master_geotherm = self.master_geotherm
        constant_depth = self.constant_depth
        constant_density = self.constant_density
        T_dependence = self.T_dependence

        # flag geotherm class instance as a master geotherm so
        #    that we make deep copies and update parameters for all 
        #    individual profiles.  Note this will update the geotherm_master
        #    attribute outide of the class instance if the master attribute is
        #    not already set to True since class attributes are mutable 
        #    objects.  This behaviour is intended.
        if isinstance(master_geotherm, td.Geotherm):
            master_geotherm.master = True

        # check that all the necessary information has been provided
        check_arguments(T_dependence, constant_depth, constant_density,
                        approach, parameters, master_geotherm)

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
            parameters = None

        # loop through Vs profiles and convert to Vp and then to density
        print(f"Reading in data and converting to density "
              f"using {approach} approach...")
        
        # set up convert_metadata list that is appended to 
        # in process_file_list function (note lists are mutable so
        # will be updated by _process_file_list method)
        self.convert_metadata = []
        
        # Process Vp files
        self._process_file_list(vp_files_all, 
                                "Vp", 
                                parameters, 
                                constant_depth, 
                                constant_density, 
                                T_dependence,
                                master_geotherm)

        # Process Vs files
        self._process_file_list(vs_files_all, 
                                "Vs", 
                                parameters, 
                                constant_depth, 
                                constant_density, 
                                T_dependence,
                                master_geotherm)

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

        Parameters
        ----------
        parallel : bool, optional
            Whether to use parallel processing (default is False). Note that 
            this functionality is not currently available owing to issues with 
            the class-based architecture for handling constants and some
            non-pickleable objects. This bug will be fixed in a later 
            release. 
            Setting parallel=True will trigger an exception.

        Returns
        -------
        np.ndarray
            An array containing station profiles for further use.
        """
        # carry out velocity conversion for each profile
        # make list of data dictionaries containing velocity, 
        # moho, density info etc.
        # check if we want to use parallel processing
        # Functionality not currently available due to current
        # that uses class instances and non-pickleable objects.
        if parallel is True:
            raise NotImplementedError("Parallel processing is currently not "
                                      "supported due to issues with pickling "
                                      "class instances.")
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

    def _assemble_params(self, 
                         file, 
                         profile_type, 
                         location_name, 
                         *extra_params):
        """
        Assemble parameters for density conversion.

        This function assembles a list of parameters required for density
        conversion, including file information, profile type, write data 
        option, path, approach, and location name. If the approach is 
        "stephenson," additional extra parameters are included.

        Parameters
        ----------
        file : str
            The file path being processed.
        profile_type : str
            The type of profile being processed (e.g., 'Vp' or 'Vs').
        location_name : str
            The name of the location associated with the file.
        *extra_params : 
            Variable number of extra parameters specific to the approach.

        Returns
        -------
        list
            A list of assembled parameters for density conversion.
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

    def _process_file_list(self, 
                           file_list, 
                           profile_type, 
                           *extra_params):
        """
        Process a list of files to assemble necessary data and metadata.
        
        This function iterates through a list of files and extracts 
        location names from file paths.
        It assembles parameter lists using the provided `profile_type` 
        and any extra parameters.
        The assembled parameters are appended to the `convert_metadata` 
        attribute of the class.

        Parameters
        ----------
        file_list : list
            A list of file paths to be processed.
        profile_type : str
            The type of profile being processed (e.g., 'Vp' or 'Vs').
        *extra_params : 
            Variable number of extra parameters specific to the profile type.

        """
        # loop through files in file_list and assemble necessary data 
        # and metadata
        for files in file_list:
            if len(files) > 0:
                # extract location name from file string
                location_name = files[0].split(os.path.sep)[-5]
                # assemble paramter list
                for file in files:
                    params = self._assemble_params(
                        file, profile_type,
                        location_name,
                        *extra_params
                    )
                    self.convert_metadata.append(params)
    
    def profiles_to_dataframe(self):
        """
        Convert station profiles to a pandas DataFrame for bulk information.

        This method iterates over the station profiles stored in the 
        `station_profiles` attribute of the class instance. For each profile, 
        it extracts the 'station', 'location', 'av_rho', 'moho', 'region', 
        'av_Vp', 'av_Vs', and 'av_Vp_calc' attributes (if they exist), and 
        stores them in separate lists. These lists are then converted to 
        numpy arrays.

        The method then creates a pandas DataFrame from these arrays, with 
        each array forming a column in the DataFrame. The DataFrame is stored 
        in the `station_profiles_df` attribute of the class instance, and is 
        also returned by the method.

        Returns
        -------
        pd.DataFrame
            A pandas DataFrame where each row represents a station profile, 
            and the columns represent the 'station', 'lon', 'lat', 'moho', 
            'av_vp', 'av_vs', 'av_rho', and 'region' attributes of the profile. 
            'lon' and 'lat' are derived from the 'location' attribute of the 
            profile, which is assumed to be a two-element array or list. If 
            'av_Vp', 'av_Vs', or 'av_Vp_calc' do not exist in a profile, their 
            values in the DataFrame are set to NaN.
        """

        try:
            # Attempt to use self.station_profiles
            self.station_profiles
        except AttributeError:
            print("The station_profiles attribute does not exist. Please "
                  "first run the send_to_conversion_function method.")
            return

        # Initialize lists to hold data
        stations = []
        lon_lats = []
        av_rhos = []
        mohos = []
        regions = []
        av_vp = []
        av_vs = []
        av_vp_calc = []

        # Iterate once over filtered_station_profiles
        for profile in self.station_profiles:
            stations.append(profile['station'])
            lon_lats.append(profile['location'])
            av_rhos.append(profile['av_rho'] 
                         if 'av_rho' in profile else np.nan)
            mohos.append(profile['moho'])
            regions.append(profile['region'])
            av_vp.append(profile['av_Vp'] 
                         if 'av_Vp' in profile else np.nan)
            av_vs.append(profile['av_Vs'] 
                         if 'av_Vs' in profile else np.nan)
            av_vp_calc.append(profile['av_Vp_calc'] 
                         if 'av_Vp_calc' in profile else np.nan)

        # Convert lists to numpy arrays
        station_array = np.array(stations)
        lon_lat_array = np.array(lon_lats)
        av_rho_array = np.array(av_rhos)
        moho_array = np.array(mohos)
        region_array = np.array(regions)
        av_vp_array = np.array(av_vp)
        av_vs_array = np.array(av_vs)
        av_vp_calc_array = np.array(av_vp_calc)

        # Create output dataframe of bulk information.
        # include average vp calculated if using brocher approach
        if self.approach == 'brocher':
            # Create output dataframe of bulk information.
            self.station_profiles_df = pd.DataFrame({
                'station': station_array,
                'lon': lon_lat_array[:, 0], 
                'lat': lon_lat_array[:, 1],
                'moho': moho_array, 
                'av_vp': av_vp_array,
                'av_vs': av_vs_array, 
                'av_rho': av_rho_array,
                'region': region_array,
                'av_vp_calc': av_vp_calc_array
            })

        else:
            # Create output dataframe of bulk information.
            self.station_profiles_df = pd.DataFrame({
                'station': station_array,
                'lon': lon_lat_array[:, 0], 
                'lat': lon_lat_array[:, 1],
                'moho': moho_array, 
                'av_vp': av_vp_array,
                'av_vs': av_vs_array, 
                'av_rho': av_rho_array,
                'region': region_array
            })

        return self.station_profiles_df


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

    Parameters
    ----------
    T_dependence : bool
        Flag indicating whether to correct for temperature and pressure.
    constant_depth : float
        Depth range for constant density from the surface.
    constant_density : float
        Constant density value for the uppermost x kilometers.
    approach : str
        Density conversion scheme, "brocher" or "stephenson".
    parameters : str
        Constants class instance if using the Stephenson scheme.
    geotherm : Geotherm, optional
        Geotherm class instance object for temperature dependence. 
        Defaults to None.

    Raises
    ------
    ValueError
        If required arguments are missing or incompatible with the selected 
        approach.
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
    write_data = False, 
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
    Convert a single Vp or Vs velocity profile to a density profile.

    Parameters
    ----------
    file : str
        Path to the input velocity profile file.
    profile_type : str
        Type of velocity profile, 'Vp' or 'Vs'.
    write_data : bool, optional
        If True, writes the converted data to files. Default is False.
    path : str, optional
        Path to write the results. Default is None.
    approach : str, optional
        Density conversion scheme to use, "brocher" or "stephenson". 
        Default is "stephenson".
    location : str, optional
        Regional location of the profile. Default is None.
    parameters : np.ndarray, optional
        Instance of the Constants, VpConstants or VsConstants classes.  
        Must be instance of the Constants class with a material_constants 
        object if T_dependence is set to True.
    constant_depth : float, optional
        Depth range for constant density from the surface. Default is None.
    constant_density : float, optional
        Constant density value for the uppermost x kilometers. Default is None.
    T_dependence : bool, optional
        If True, corrects velocity and density for temperature and pressure. 
        Default is False.
    geotherm : object, optional
        Instance of the Geotherm class for temperature dependence.  
        Must be provided if T_dependence is set to True.  Default is None.
    print_working_file : bool, optional
        If True, prints the file that is being converted. Default is False.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the converted velocity and density data.

    Examples
    --------
    >>> convert_V_profile(
    ...     'velocity_profile.dat', 
    ...     'Vp', 
    ...     write_data=True, 
    ...     path='output/', 
    ...     approach='stephenson', 
    ...     location='Region1', 
    ...     parameters=constants_object, 
    ...     constant_depth=10.0, 
    ...     constant_density=2.7, 
    ...     T_dependence=True, 
    ...     geotherm=geotherm_object, 
    ...     print_working_file=True
    ... )
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
    Data = Convert(
        file, 
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
            Data.Vs_to_Vp_brocher()
        
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

def av_profile(profiles, 
               base_depth, 
               average_type = 'median'):
    """
    Calculate the average profile and bulk property profile given a family 
    of profiles.

    This function takes a list of profiles, where each profile is represented 
    as a NumPy array with columns [depth, property]. It calculates the 
    average profile and the bulk property profile as a function of depth.

    Parameters
    ----------
    profiles : list of np.ndarray
        List of profiles (e.g., Vp, Vs, P, T, rho, etc.). Each profile must 
        contain at least two columns: depth and property.
    base_depth : float
        Depth to which the bulk property should be averaged.
    average_type : str, optional
        Method of averaging ('median' or 'mean'). Default is 'median'.

    Returns
    -------
    av_z_f : scipy.interpolate.CubicSpline
        Average profile as a function of depth.
    bulk_base_depth_f : scipy.interpolate.CubicSpline
        Bulk property profile as a function of base depth.

    Examples
    --------
    >>> profiles = [
    ...     np.array([
    ...         [0.0, 1.5], 
    ...         [5.0, 2.0], 
    ...         [10.0, 2.5]
    ...     ]),
    ...     np.array([
    ...         [0.0, 1.6], 
    ...         [5.0, 2.1], 
    ...         [10.0, 2.6]
    ...     ])
    ... ]
    >>> av_z_f, bulk_base_depth_f = av_profile(
    ...     profiles, 
    ...     base_depth=15.0, 
    ...     average_type='mean'
    ... )
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

def save_bulk_profiles(
        outpath, 
        filename, 
        data_function, 
        start, 
        stop, 
        step
        ):
    """
    Save bulk profiles data to a file.

    This function generates x-values using numpy.arange() based on the 
    specified start, stop, and step values.
    It computes the data values using the provided data_function and saves 
    the data to a file in the specified outpath.

    Parameters
    ----------
    outpath : str
        The output directory where the data file will be saved.
    filename : str
        The name of the output data file.
    data_function : function
        A function that computes the average property at depth z.
    start : float
        The start value for the x-values.
    stop : float
        The stop value for the x-values.
    step : float
        The step size for generating x-values.

    Returns
    -------
    None

    Examples
    --------    
    >>> save_bulk_profiles(
    ...     '/output/directory/', 
    ...     'bulk_data.txt', 
    ...     function, 
    ...     0, 
    ...     50.5, 
    ...     0.5
    ... )
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
    ranges. It is suitable for profiles with distinct layers rather than 
    those with gradients, although it is designed to produce a reasonable
    solution in both cases.

    Parameters
    ----------
    profile1 : np.ndarray
        The first profile as a NumPy array with columns [depth, velocity].
    profile2 : np.ndarray
        The second profile as a NumPy array with columns [depth, velocity].

    Returns
    -------
    np.ndarray
        A binned velocity profile aligned with the depth intervals of the 
        lowest-resolution profile.

    Examples
    --------
    >>> binned_profile = convert_to_same_depth_intervals(profile1, profile2)
    >>> print(binned_profile)
    array([[ 0. , 1600. ],
           [ 2.5, 1600. ],
           [ 2.5 , 2050. ],
           [10. , 2050. ]])

    Notes
    -----
    - This function finds the lowest-resolution profile and interpolates the 
        other profile accordingly.
    - It returns a binned profile with the same depth intervals as the 
        lowest-resolution profile.
    - It works best when lowest-resolution profile has discrete layers 
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

    Parameters
    ----------
    data_file : str
        Path to the data file to be read.
    region_name : str, optional
        Optional regional location of the profile.

    Returns
    -------
    pd.DataFrame
        A Pandas DataFrame containing the following columns:
        
        - 'region' (str): Regional location of the profile (if provided, 
            otherwise None).
        - 'lon' (float): Longitude of the station.
        - 'lat' (float): Latitude of the station.
        - 'moho' (float): Crustal thickness estimate.
        - 'vp_vs' (float): Ratio of seismic velocities (vp/vs).

    Examples
    --------
    >>> df = read_no_profile_data(
    ...         'crustal_data.csv', 
    ...         region_name='Example_Region')
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

    Parameters
    ----------
    rho : float
        Average density of the overburden in Mg/m³.
    h : float
        Thickness of the overburden in kilometers.
    g : float, optional
        Acceleration due to gravity in m/s². Default is 9.81 m/s².

    Returns
    -------
    float
        Lithostatic pressure in Pascals (Pa).

    Notes
    -----
    The lithostatic pressure (P) is calculated using the formula:
    P = rho * g * h * 1e6
    """

    return rho * g * h * 1e6


###################################################################

def progress_bar(iterable, length=50):
    """
    Create a text-based progress bar to track the progress of an iterable.

    Parameters
    ----------
    iterable : iterable
        The iterable object (e.g., list, range) to iterate through.
    length : int, optional
        The length of the progress bar. Defaults to 50.

    Yields
    ------
    item
        The current item from the iterable.

    Examples
    --------
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