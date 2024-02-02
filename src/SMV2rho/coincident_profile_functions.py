#!/usr/bin/env python3

########################################################################################
# functions to locate coincident velocity profiles

########################################################################################

import numpy as np
from SMV2rho.spatial_functions import *

# locate coincident profiles
def get_coincident_profiles(profiles1, profiles2, buff):
    """
    Find all surveys that are in the same place.

    Parameters
    ----------
    profiles1 : list
        List of vp profiles.
    profiles2 : list
        List of vs profiles.
    buff : float
        Buffer distance in km (i.e. search distance between different profiles 
        to look for coincident profiles).

    Returns
    -------
    coincident_profiles_all : list
        List of profiles where Vp and Vs are coincident. If no coincident 
        profiles are available then None is written as the list entry.
    profiles1 : list
        List of vp profiles.
    profiles2 : list
        List of vs profiles.
    """
    
    print("Finding stations with coincident Vp and Vs surveys...")
    print("   -- this may take a few minutes...!")
    # first find all vs profiles and all vp profiles

    print(f"   -- taking locations < {buff:.1f} km from vp measurement")
    i = 1
    coincident_profiles_all = []
    for p1 in profiles1:
        # print progress to command line
        if i%100 == 0:
            print(f"       - currently on item {i:.0f} of {len(profiles1):.0f}")
        dists = []
        for p2 in profiles2:
            # calculate distance between two seismic profiles
            dist = haversine(p1['location'], p2['location'])
            dists.append(dist)
            dists_array = np.array(dists)
            # locate coincident profiles
            coinc_profiles_id = np.array(np.where(dists_array < buff))#
        coincident_profiles_all.append(profiles2[coinc_profiles_id[0]])
        i += 1
    return coincident_profiles_all, profiles1, profiles2


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def compare_adjacent_profiles(
        profiles, 
        coincident_profiles, 
        approach
        ):
    """
    Calculate and compare bulk properties of overlapping/coincident velocity 
    profiles.

    This function takes velocity profiles and coincident profiles, and 
    calculates and compares various properties between them.

    Parameters
    ----------
    profiles : list of dictionaries
        Velocity profiles containing location, velocity, density, etc.
        Output from run_convert_velocities.
    coincident_profiles : list of lists
        Output from get_coincident_profiles. It's a list in the same
        format as v_profiles with empty entries where no coincident
        profile exists and an entry where there is a profile that
        coincides with a v_profile with a corresponding index.
    approach : str
        Method used to calculate velocities. Required since there 
        may not be a calculated vp profile. It can be 'brocher' or 
        'stephenson'.

    Returns
    -------
    vp_vs_vpcalc : numpy.ndarray
        Comparison between vp, vs, and calculated vp. If calculated 
        vp is not available, it writes np.nan.
    rho_rho : numpy.ndarray
        Comparison of density estimates between vp and vs profiles.
    moho_moho : numpy.ndarray
        Comparison of moho depth estimates between vp and vs profiles.
    lon_lat : numpy.ndarray
        Locations of coincident profiles.
    """
    # Initialize empty lists to store results
    vp_vs_vpcalc = []
    rho_rho = []
    moho_moho = []
    lon_lat = []

    # Iterate through v_profiles and coincident_profiles
    for vp, vs_list in zip(profiles, coincident_profiles):
        for vs in vs_list:
            # Extract location information
            lon_lat_i = vs['location']
            
            # Initialize arrays to store velocity and density data
            vpo_vso_vpc = np.zeros(3)
            rho_rho_i = np.zeros(2)
            moho_moho_i = np.zeros(2)

            # Populate vpo_vso_vpc array with velocity values
            vpo_vso_vpc[1] = vs['av_Vs']  # observed vs
            if approach == 'brocher':
                vpo_vso_vpc[2] = vs['av_Vp']  # calculated vp using Brocher (2005)
            elif approach == 'stephenson':
                vpo_vso_vpc[2] = np.nan  # calculated vp not available for this approach
            vpo_vso_vpc[0] = vp['av_Vp']  # observed vp

            # Populate rho_rho_i array with density values
            rho_rho_i[0] = vp['av_rho']  # density from vp
            rho_rho_i[1] = vs['av_rho']  # density from vs

            # Populate moho_moho_i array with moho depth values
            moho_moho_i[0] = vp['moho']  # moho depth from vp
            moho_moho_i[1] = vs['moho']  # moho depth from vs

            # Append arrays to result lists
            vp_vs_vpcalc.append(vpo_vso_vpc)
            rho_rho.append(rho_rho_i)
            moho_moho.append(moho_moho_i)
            lon_lat.append(lon_lat_i)

    # Convert lists to NumPy arrays
    vp_vs_vpcalc = np.array(vp_vs_vpcalc)
    rho_rho = np.array(rho_rho)
    moho_moho = np.array(moho_moho)
    lon_lat = np.array(lon_lat)

    return vp_vs_vpcalc, rho_rho, moho_moho, lon_lat