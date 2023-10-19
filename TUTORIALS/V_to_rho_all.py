#!/usr/bin/env python3

# Convert from seismic velocities to density.

# Choose between Brocher's empirical (i.e. Nafe-Drake) fit
#   documented in Brocher (2005), after Ludwig (1970) and
#   new pressure-and temperature dependent scheme (this study).

# If using "brocher", Vs profiles are first converted to Vp 
#   and then to density.

# If using "stephenson", then density conversion is carried out using
#   parameters determined by inverse modelling of lab data.
#   Requires a parameter file.

# Processing requires strict file structure as laid out in the tutorial
#   jupyter notebook.

# also finds coincident vp and vs profiles within given distance, buff.

# @Author: Started -- Simon Stephenson @Date: Mar 2019
#          Processing and brocher scheme included: Mar 2019
#          Updated density conversion scheme added Feb 2021
#          Full pressure dependence added: May 2022
#          Temperature dependence included: Jan 2023

### file format:
#     Station name
#     lon lat
#     moho_depth
#     Vs    depth
#     .     .
#     .     .
#     .     .

########################################################################################
# import modules...

import numpy as np
import matplotlib.pyplot as plt
import sys
from density_functions import *
from spatial_functions import *
from coincident_profile_functions import *
import pandas as pd
from plotting import *

########################################################################################
# inputs

#density conversion approach

approach = "stephenson"
#approach = "brocher"

# do you want to use full T and P dependent conversion?
T_dependence = True

# select profiles to calculate 
#   (i.e. name of directory containing velocity profiles)
#   set to "ALL" to do everywhere
which_profiles = "ALL"

# location of crustal velocity files
path = "../../SEISCRUST"

# path to output data
outpath = "../COMPARISONS/DATA/CRUSTAL_STRUCTURE"

# output velocity and density file name (note _[approach].dat will be appended)
av_vp_vs_rho_file = "av_vp_vs_rho_all"

# buffer distance between coincident vp and vs profiles
buff = 50.

# path to density conversion parameters (if using stephenson approach)
parameter_dir = "../DENSITY_MODELS/STEPHENSON_PARAM_FILES/"
# standard parameters with exponential dropoff
parameters_type = "parameters"
# parameters omitting exponential dropoff in velocity
#parameters_type = "parameters_c0"

# Hk data file
#all_crustal_thickness_file = "/Users/eart0518/Work/RESIDUAL_ELEVATION/GLOBAL_RECEIVER_FUNCS/DATA/all_crust_data_onshore.csv"
all_crustal_thickness_file = "/Users/eart0518/Work/RESIDUAL_ELEVATION/GLOBAL_RECEIVER_FUNCS/DATA/all_crust_data_onshore_no_EARS.csv"

# depth and density values for constant density for uppermost part of crust
#   Prevents spurious densities in the exponential part
#   of the relationship since this exponential dropoff is
#   rarely honoured in the seismic data.
#    NB. ONLY APPLICABLE IF USING approach = "stephenson"

rho_const = 2.75        # constant density of uppermost crust Mg/m^3
d_const = 7.            # depth range for constant density, km

#rho_const = None        # constant density of uppermost crust Mg/m^3
#d_const = None            # depth range for constant density, km

# maximum depth down to which bulk crustal properties will be calculated
# ie. average and bulk vp, va, rho
bulk_depth_limit = 70

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# empirical average density as function of depth function
average_rho_z_file = "av_dens_depth_function_T_DEPENDENT.dat"
average_vp_z_file = "av_vp_depth_function_T_DEPENDENT.dat"
average_vs_z_file = "av_vs_depth_function_T_DEPENDENT.dat"

# empirial bulk density as function of crustal thickness function
bulk_rho_tc_file = "bulk_rho_tc_function_T_DEPENDENT.dat"
bulk_vp_tc_file = "bulk_vp_tc_function_T_DEPENDENT.dat"
bulk_vs_tc_file = "bulk_vs_tc_function_T_DEPENDENT.dat"

# averaging approach (i.e. mean or median)
#average_type = "median"
average_type = "mean"

########################################################################################

# DENSITY CONVERSION

# load parameter files for density conversion
Vp_params = np.loadtxt(parameter_dir + "Vp_" + parameters_type + ".dat")
Vs_params = np.loadtxt(parameter_dir + "Vs_" + parameters_type + ".dat")

# load parameters for temperature dependence
T_parameters = np.loadtxt(parameter_dir + "T_parameters.dat")

# stack parameters together [Vp, Vs]
parameters = np.column_stack((Vp_params, Vs_params))

# set up the class instance to convert multiple profiles at once
Profiles = MultiConversion(path, 
                        which_location = which_profiles,
                        write_data = True, 
                        approach = "stephenson",
                        parameters = parameters,
                        constant_depth = 7,
                        constant_density = 2.75,
                        T_dependence = T_dependence,
                        T_parameters = T_parameters)

# assemble necessary information to carry out conversion
#  e.g. file paths, parameters, other metadata etc.
Profiles.assemble_file_lists()

# run density conversion for all profiles
station_profiles = Profiles.send_to_conversion_function()

# get crustal thickness datafile as Pandas DataFrame (i.e. data without velocity profiles)
all_crustal_thickness_data = read_no_profile_data(all_crustal_thickness_file)

# collate bulk information for each profile
station_array = np.array([s['station'] for s in station_profiles])
lon_lat_array = np.array([ll['location'] for ll in station_profiles])
av_rho_array = np.array([r['av_rho'] for r in station_profiles])
moho_array = np.array([m['moho'] for m in station_profiles])
region_array = np.array([rgn['region'] for rgn in station_profiles])

# get arrays of average Vp and Vs
av_vp_array = [station['av_Vp'] if 'av_Vp' in station else np.nan 
               for station in station_profiles]
av_vs_array = [station['av_Vs'] if 'av_Vs' in station else np.nan 
               for station in station_profiles]

# create output dataframe of bulk information...
#   [station_name, lon, lat, crust_thickness, 
#    average_vp, average_vs, average_density]
#   if original velocity file is vp, write vs = np.nan and vice versa
#output_data = np.column_stack((station_array, lon_lat_array, moho_array, 
#                                av_vp_array, av_vs_array, av_rho_array))
output_data_converted = pd.DataFrame({'station': station_array,
                                      'lon': lon_lat_array[:,0], 
                                      'lat': lon_lat_array[:,1],
                                      'moho': moho_array, 
                                      'av_vp': av_vp_array,
                                      'av_vs': av_vs_array, 
                                      'av_rho': av_rho_array,
                                      'region': region_array})

# get average density as function of depth and bulk density as function 
# of crustal thickness
print("Getting average density, Vp and Vs profiles")
rho_z, rho_tc = av_profile([profile['rho_hi_res'] 
                             for profile in station_profiles], 
                             bulk_depth_limit,
                             average_type = average_type)
Vp_z, Vp_tc = av_profile([profile['Vp_hi_res'] 
                          for profile in station_profiles 
                          if 'Vp' in profile], 
                          bulk_depth_limit, average_type = average_type)
Vs_z, Vs_tc = av_profile([profile['Vs_hi_res'] 
                          for profile in station_profiles 
                          if 'Vs' in profile], 
                          bulk_depth_limit, average_type = average_type)

# calculate bulk density from bulk crustal thickness for places without V profiles
# write nan if the density is outside the depth limit 
# (I.e. on really thick crust where bulk profile is unreliable)
all_crustal_thickness_data['av_rho'] = np.where(
                                all_crustal_thickness_data['moho'] < bulk_depth_limit, 
                                rho_tc(all_crustal_thickness_data['moho']), np.nan)

# merge dataframes to create single output file (how='outer' to preserve all keys)
output_data_all = pd.merge(output_data_converted, 
                           all_crustal_thickness_data, how='outer')

########################################################################################
# WRITE OUTPUT DATA FILES

# check approach and whether T dependence is being used for outfile path names
if approach == 'stephenson':
    if Profiles.T_dependence is True:
        print("Saving T dependent bulk density file...")
        approach = approach + "_T_DEPENDENT"
        outfile = outpath + av_vp_vs_rho_file + "_" + approach + ".dat"
    else:
        print("Saving T bulk density file...")
        outfile = outpath + av_vp_vs_rho_file + "_" + approach + ".dat"

# save global average velocity and density
print("Saving concatenated, high-resolution velocity and density files...")
np.savetxt(outpath + "density_high_res_global" + "_" + approach + ".dat", 
            np.concatenate([profile['rho_hi_res'] 
                            for profile in station_profiles]))
np.savetxt(outpath + "Vp_high_res_global.dat", 
            np.concatenate([profile['Vp_hi_res'] 
                            for profile in station_profiles if
                            'Vp_hi_res' in profile]))
np.savetxt(outpath + "Vs_high_res_global.dat", 
            np.concatenate([profile['Vs_hi_res'] 
                            for profile in station_profiles if
                            'Vs_hi_res' in profile]))

# save average density, Vp and Vs as function of depth and bulk density as function
#   of crustal thickness.  Ensure NaNs appear as 'nan' and separator is space.
print("Saving bulk density data file...")
output_data_all.to_csv(outfile, sep=' ', index=False, na_rep='nan',
                        columns=['station', 'lon', 'lat', 'moho', 
                                'av_vp', 'av_vs', 'av_rho'])

save_bulk_profiles(outpath, average_rho_z_file, rho_z, 0, 50, 0.5)
save_bulk_profiles(outpath, bulk_rho_tc_file, rho_tc, 0, 50, 0.5)
save_bulk_profiles(outpath, average_vp_z_file, Vp_z, 0, 50, 0.5)
save_bulk_profiles(outpath, bulk_vp_tc_file, Vp_tc, 0, 50, 0.5)
save_bulk_profiles(outpath, average_vs_z_file, Vs_z, 0, 50, 0.5)
save_bulk_profiles(outpath, bulk_vs_tc_file, Vs_tc, 0, 50, 0.5)

########################################################################################
# LOCATE COINCIDENT PROFILES...

# extract vs profiles and vp profiles from station_profiles
vs_profiles = np.array(list(filter(lambda station: 
                            station['type'] == 'Vs', station_profiles)))
vp_profiles = np.array(list(filter(lambda station: 
                            station['type'] == 'Vp', station_profiles)))

# calculate coincident vp and vs profiles within buffer distance, buff
coincident_profiles, vp_profiles, vs_profiles = \
    get_coincident_profiles(vp_profiles, vs_profiles, buff)

# get some bulk crustal properties for the coincident profiles,
#  e.g. compare densities or moho depth etc.
# also return the location of coincident profiles.
vp_vs_vpcalc, rho_rho, moho_moho, lon_lat = \
    compare_adjacent_profiles(vp_profiles, coincident_profiles, approach)

# save files
np.savetxt(outpath + "bulk_vp_vs_" + approach + ".dat", 
            np.column_stack((vp_vs_vpcalc[:,0], vp_vs_vpcalc[:,1])))
np.savetxt(outpath + "obs_vp_calc_vp_" + approach + ".dat", 
            np.column_stack((vp_vs_vpcalc[:,0], vp_vs_vpcalc[:,2])))
np.savetxt(outpath + "rho_vp_rho_vs_" + approach + ".dat", rho_rho)
np.savetxt(outpath + "moho_moho_" + approach + ".dat", moho_moho)
np.savetxt(outpath + "lon_lat_" + approach + ".dat", lon_lat)
np.savetxt(outpath + "vpo_vso_vpc_" + approach + ".dat", vp_vs_vpcalc)

########################################################################################
# PLOTTING

# Define data
vp_observed = vp_vs_vpcalc[:, 0]
vs_observed = vp_vs_vpcalc[:, 1]
vp_calculated = vp_vs_vpcalc[:, 2]
rho_vp = rho_rho[:, 0]
rho_vs = rho_rho[:, 1]
vp_moho = moho_moho[:, 0]
vs_moho = moho_moho[:, 1]

plot_three_scatter(vp_vs_vpcalc[:, :2], 
                   rho_rho, moho_moho, 
                   x_labels=["Vp observed", "Vp density", "Vp, moho"], 
                   y_labels=["Vs observed", "Vs density", "Vs moho"])