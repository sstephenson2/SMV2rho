#!/usr/bin/env python3

# calculate and propagate uncertainties through density calculation

from distutils.config import DEFAULT_PYPIRC
import numpy as np
import sys
import scipy.stats as stats
import matplotlib.pyplot as plt
from SMV2rho.density_functions import V2rho_stephenson as V2rho
from SMV2rho.temperature_dependence import rho_thermal2, compressibility

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# absolute precision error (for combining uncertainties)
def abs_precision_error(x, x_mean, N):
    return np.sqrt(np.sum((x - x_mean)**2) / (N-1))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# crustal density uncertainty
def rho_err(
        constants, 
        geotherm,
        profile_type='Vp',
        N=1000, 
        z_slices=100, 
        make_plots=False,
        save_plots=False, 
        outpath="../UNCERTAINTY_PLOTS"
        ):
    """
    Calculate crustal density uncertainty by Monte Carlo sampling of 
    thermal and compressibility parameters. Returns drho(z) for mean 
    correction, and standard deviation. Note that this does not include 
    the uncertainty associated with the velocity-pressure calibration, 
    which should be combined separately if necessary.

    This function can only handle a single profile at a time.  Note that
    the Geotherm class must be initialised with the correct parameters
    including the geotherm parameters (including tc) and uncertainties.

    Args:
        z (float): Moho depth in kilometers.
        bulk_rho (float): Bulk density of the crust to approximate pressure 
            gradient.
        q0 (float): Thermal ad material parameters (1 sigma).
        dq0 (float): Uncertainty in q0.
        qm (float): Thermal material parameter (1 sigma).
        dqm (float): Uncertainty in qm.
        hr (float): Depth parameter (1 sigma), converted to meters.
        dhr (float): Uncertainty in hr.
        dvdT (float): Pressure-velocity conversion parameter (1 sigma).
        ddvdT (float): Uncertainty in dvdT.
        alpha0 (float): Thermal expansion parameter (1 sigma).
        dalpha0 (float): Uncertainty in alpha0.
        alpha1 (float): Thermal expansion parameter (1 sigma).
        dalpha1 (float): Uncertainty in alpha1.
        K (float): Compressibility parameter (1 sigma).
        dK (float): Uncertainty in K.
        dens_parameters (str): File path to the pressure-velocity-density 
            conversion parameters.
        N (int, optional): Number of random samples to draw from each 
        distribution. Default is 1000.
        z_slices (int, optional): Number of depth slices. Default is 50.
        make_plots (bool, optional): Whether to generate and display plots. 
            Default is False.
        save_plots (bool, optional): Whether to save plots to specified 
            outpath. Default is False.
        outpath (str, optional): Path to save plots. Default is 
            "../UNCERTAINTY_PLOTS".

    Returns:
        tuple: A tuple containing the average fractional error and 
        average absolute error in density.
    """

    # unpack error parameters
    if profile_type == 'Vp':
        v_constants = constants.vp_constants
    elif profile_type == 'Vs':
        v_constants = constants.vs_constants
    

    # set variable names locally
    m = abs(v_constants.m)
    m_unc = v_constants.m_unc
    alpha0 = constants.material_constants.alpha0
    alpha0_unc = constants.material_constants.alpha0_unc
    alpha1 = constants.material_constants.alpha1
    alpha1_unc = constants.material_constants.alpha1_unc
    K = constants.material_constants.K
    K_unc = constants.material_constants.K_unc
    bulk_rho = geotherm.rho

    # sample normal distributions for thermal expansion, 
    # compressibility, and velocity parameters
    # take absolute value to avoid negative draws.  Note that
    # this adjustments means that the parameter distribution 
    # is not perfectly normal, but it is close enough for our purposes. 
    dvdT_gauss = generate_gauss(abs(m), m_unc, N)
    alpha0_gauss = generate_gauss(alpha0, alpha0_unc, N)
    alpha1_gauss = generate_gauss(alpha1, alpha1_unc, N)

    # stop bulk modulus approaching zero to avoid crazy uncertainties
    # avoids dividing P by a very small (or negative) number
    K_gauss = generate_gauss(K - (K/3), K_unc, N) + (K/3)

    # generate family of geotherms by sampling distributions
    z_arr, Tz_all = geotherm.generate_geotherm_distribution(
        n_geotherms = N, 
        z_slices = z_slices
        )
    
    # calculate average error in temperature
    T_errors = np.column_stack(
        (np.mean(Tz_all, axis=0), 
        np.std(Tz_all, axis=0))
        )

    # calculate v correction error using random family of 
    # geotherms and dvdT estimates
    v_err_all = Tz_all * np.transpose(np.repeat([dvdT_gauss], 
                                                z_slices, axis=0))

    # calculate mean and standard deviation of v correction
    vc_err = np.column_stack((np.mean(v_err_all, axis=0), 
                                          np.std(v_err_all, axis=0)))
    
    # get error in density at s.t.p. by propagating v 
    # correction error through density conversion
    # note that absolute value of v doesn't matter 
    # since at given P relationship is linear.
    # approximate pressure using bulk density and depth

    overburden = 9.81 * bulk_rho * z_arr

    rho_0_error = (
        V2rho([overburden, vc_err[:, 1]], v_constants) 
        - V2rho([overburden, 0], v_constants)
    )

    rho_0_mean = (
        V2rho([overburden, vc_err[:, 0]], v_constants) 
        - V2rho([overburden, 0], v_constants)
    )
    
    # stack together mean and errors
    rho_0_errors = np.column_stack((rho_0_mean, rho_0_error))
    
    # calculate errors due to thermal expansion
    thermal_expans_profiles = rho_thermal2(
        bulk_rho, np.transpose(Tz_all),
        alpha0=np.repeat([alpha0_gauss], 
                          z_slices, axis=0),
        alpha1=np.repeat([alpha1_gauss], 
                          z_slices, axis=0))[2]
    thermal_expans_profiles_frac = rho_thermal2(
        bulk_rho, np.transpose(Tz_all),
        alpha0=np.repeat([alpha0_gauss], 
                          z_slices, axis=0),
        alpha1=np.repeat([alpha1_gauss], 
                          z_slices, axis=0))[1]
    
    thermal_expans_abs_err = np.column_stack(
        (np.mean(thermal_expans_profiles, axis=1), 
        np.std(thermal_expans_profiles, axis=1)))
    thermal_expans_frac_err = np.column_stack(
        (np.mean(thermal_expans_profiles_frac , axis=1), 
        np.std(thermal_expans_profiles_frac , axis=1)))                                      
    
    # calculate errors due to compressibility
    Pz = np.transpose(np.repeat([z_arr], N, axis=0)) * 9.81 * bulk_rho
    compress_profiles = compressibility(
        bulk_rho, Pz, K=np.repeat([K_gauss], 
                                   z_slices, axis=0))[2]
    compress_profiles_frac = compressibility(
        bulk_rho, Pz, K=np.repeat([K_gauss], 
                                  z_slices, axis=0))[1]
    
    compress_abs_err = np.column_stack(
        (np.mean(compress_profiles, axis=1), 
         np.std(compress_profiles, axis=1)))
    compress_frac_err = np.column_stack(
        (np.mean(compress_profiles_frac, axis=1), 
         np.std(compress_profiles_frac, axis=1)))
    
    # combine errors assuming they are uncorrelated (not completely correct 
    # approach for velocity correction and thermal expansion as they both 
    # depend on particular geotherm but a reasonable approximation)

    # error and correction by adding terms
    total_correction_add = (rho_0_errors[:,0] 
                            + thermal_expans_abs_err[:,0] 
                            + compress_abs_err[:,0])
    error_total_add = np.sqrt(rho_0_errors[:,1]**2 
                              + thermal_expans_abs_err[:,1]**2 
                              + compress_abs_err[:,1]**2)
    
    # fractional error and correction by multiplication,
    # preferred way of calculating uncertainty.  Still assumes
    # errors are uncorrelated (not completely correct assumption
    # but a reasomnable approximation).
    # fractional change in rho_0
    compress_therm_frac = (compress_frac_err[:,0] 
                           * thermal_expans_frac_err[:,0])
    # corrected mean rho(z)
    total_correction_frac = ((bulk_rho + rho_0_errors[:,0]) 
                             * (compress_therm_frac) - bulk_rho)
    # uncertainty in density correction as function of z
    error_total_frac = np.sqrt((rho_0_errors[:,1] 
                                   * compress_therm_frac)**2
                                + (compress_frac_err[:,1] 
                                   * (bulk_rho + rho_0_errors[:,0]) 
                                   * compress_therm_frac)**2
                                + (thermal_expans_frac_err[:,1] 
                                   * (bulk_rho + rho_0_errors[:,0]) 
                                   * compress_therm_frac)**2)
    
    # Tz array formatted for saving
    Tz_out = np.array([np.column_stack((z_arr, i)) for i in Tz_all])

    if save_plots is True:
        
        np.savetxt(outpath + f"/Tz_all_{z}_{bulk_rho}.dat", 
                    np.concatenate(Tz_out))
        np.savetxt(outpath + f"/T_errors_{z}_{bulk_rho}.dat", 
                    np.column_stack((z_arr, T_errors)))
        np.savetxt(outpath + f"/rho_o_error_{z}_{bulk_rho}.dat", 
                    np.column_stack((z_arr, rho_0_errors)))
        np.savetxt(outpath + f"/thermal_expans_error_{z}_{bulk_rho}.dat", 
                    np.column_stack((z_arr, thermal_expans_abs_err)))
        np.savetxt(outpath + f"/compress_error_{z}_{bulk_rho}.dat", 
                    np.column_stack((z_arr, compress_abs_err)))
        np.savetxt(outpath + f"/total_error_{z}_{bulk_rho}.dat", 
                    np.column_stack((z_arr, error_total_add)))
        np.savetxt(outpath + f"/total_error_frac_{z}_{bulk_rho}.dat", 
                    np.column_stack((z_arr, error_total_frac)))

        np.savetxt(outpath + f"/total_correction_add_{z}_{bulk_rho}.dat", 
                    np.column_stack((z_arr, total_correction_add)))
        np.savetxt(outpath + f"/total_correction_frac_{z}_{bulk_rho}.dat", 
                    np.column_stack((z_arr, total_correction_frac)))
        

    if make_plots is True:    
        # Create a multi-panel figure
        fig, axs = plt.subplots(2, 4, figsize=(12, 8))

        # Set common labels
        fig.text(0.5, 0.04, 'Value', ha='center', va='center', fontsize=12)
        fig.text(0.06, 0.5, 'Depth (km)', ha='center', 
                 va='center', rotation='vertical', fontsize=12)

        # Plotting

        # Panel 1
        axs[0, 0].plot(total_correction_add, -z_arr, label='Average')
        axs[0, 0].plot(total_correction_add + error_total_add, 
                       -z_arr, label=r'Total Corr $+ 1\sigma$')
        axs[0, 0].plot(total_correction_add - error_total_add, 
                       -z_arr, label=r'Total Corr $- 1\sigma$')
        axs[0, 0].set_title('Total correction \nby addition')

        # Panel 2
        axs[0, 1].plot(total_correction_frac, -z_arr, label='Average')
        axs[0, 1].plot(total_correction_frac + error_total_frac, 
                       -z_arr, label=r'$+ 1\sigma$')
        axs[0, 1].plot(total_correction_frac - error_total_frac, 
                       -z_arr, label=r'$- 1\sigma$')
        axs[0, 1].set_title('Total correction and \nfractional error')


        geotherm.generate_geotherm()

        # Panel 3
        axs[0, 2].hist2d(np.concatenate(Tz_out)[:,1], 
                         -np.concatenate(Tz_out)[:,0], bins=25)
        axs[0, 2].plot(geotherm.T, -geotherm.z)
        axs[0, 2].plot(T_errors[:,0], -z_arr, label='Average')
        axs[0, 2].plot(T_errors[:,0] 
                       + T_errors[:,1], -z_arr, label=r'$+ 1\sigma$')
        axs[0, 2].plot(T_errors[:,0] 
                       - T_errors[:,1], -z_arr, label=r'$- 1\sigma$')
        axs[0, 2].set_title('T error')

        # Panel 4
        axs[0, 3].plot(compress_abs_err[:,0], -z_arr, label='Average')
        axs[0, 3].plot(compress_abs_err[:,0] + compress_abs_err[:,1], 
                       -z_arr, label=r'$+ 1\sigma$')
        axs[0, 3].plot(compress_abs_err[:,0] - compress_abs_err[:,1], 
                       -z_arr, label=r'$- 1\sigma$')
        axs[0, 3].set_title('Compression \nabsolute error')

        # Panel 5
        axs[1, 0].plot(thermal_expans_abs_err[:,0], 
                       -z_arr, label='Average')
        axs[1, 0].plot(thermal_expans_abs_err[:,0] 
                       + thermal_expans_abs_err[:,1], 
                       -z_arr, label=r'$+ 1\sigma$')
        axs[1, 0].plot(thermal_expans_abs_err[:,0] 
                       - thermal_expans_abs_err[:,1], 
                       -z_arr, label=r'$- 1\sigma$')
        axs[1, 0].set_title('Thermal expansion \nabsolute error')

        # Panel 6
        axs[1, 1].plot(compress_frac_err[:,0], -z_arr, label='Average')
        axs[1, 1].plot(compress_frac_err[:,0] 
                       + compress_frac_err[:,1], 
                       -z_arr, label=r'$+ 1\sigma$')
        axs[1, 1].plot(compress_frac_err[:,0] 
                       - compress_frac_err[:,1], 
                       -z_arr, label=r'$- 1\sigma$')
        axs[1, 1].set_title('Compression \nfractional error')

        # Panel 7
        axs[1, 2].plot(thermal_expans_frac_err[:,0], 
                       -z_arr, label='Average')
        axs[1, 2].plot(thermal_expans_frac_err[:,0] 
                       + thermal_expans_frac_err[:,1], 
                       -z_arr, label=r'$+ 1\sigma$')
        axs[1, 2].plot(thermal_expans_frac_err[:,0] 
                       - thermal_expans_frac_err[:,1], 
                       -z_arr, label=r'$- 1\sigma$')
        axs[1, 2].set_title('Thermal expansion \nfractional error')

        # Panel 8
        axs[1, 3].plot(rho_0_errors[:,0], -z_arr, label='Average')
        axs[1, 3].plot(rho_0_errors[:,0] - rho_0_errors[:,1], 
                       -z_arr, label=r'$+ 1\sigma$')
        axs[1, 3].plot(rho_0_errors[:,0] + rho_0_errors[:,1], 
                       -z_arr, label=r'$- 1\sigma$')
        axs[1, 3].set_title(r"$\rho_\circ(z)$ error")

        # Adjust spacing between subplots
        plt.subplots_adjust(wspace=0.3, hspace=0.5)

        # Create a single legend
        legend_labels = ['Average', r'$+ 1\sigma$', r'$- 1\sigma$']
        axs[1, 3].legend(legend_labels, loc='center', 
                         bbox_to_anchor=(0.2, -0.3), ncol=3)

        # Show the plot
        plt.show()
    
    error_average_add = (np.sqrt(np.sum(error_total_add**2)
                                 / (len(error_total_add)-1)))
    error_average_frac = (np.sqrt(np.sum(error_total_frac**2)
                                  / (len(error_total_frac)-1)))
    
    return error_average_frac, error_average_add





def generate_gauss(abs_mean, mean_unc, N):
    """
    Generate an array of absolute Gaussian random numbers.

    Parameters:
    abs_mean (float): The absolute mean of the Gaussian distribution.
    mean_unc (float): The standard deviation (uncertainty) of the 
                      Gaussian distribution.
    N (int): The number of random numbers to generate.

    Returns:
    numpy.ndarray: An array of absolute Gaussian random numbers.
    """
    # Generate N random numbers from a Gaussian distribution with mean 
    # abs_mean and standard deviation mean_unc
    # The np.random.normal function generates these random numbers
    # The np.abs function takes the absolute value of these numbers, so 
    # the result is an array of absolute Gaussian random numbers
    return np.abs(np.random.normal(abs_mean, mean_unc, N))
