import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
import os
from os.path import basename
from copy import copy
import shutil

from ppxf.ppxf import ppxf
import ppxf.ppxf_util as util
import ppxf.sps_util as lib

from .der_snr import DER_SNR
from .ned_calculator import NedCalculator

def bootstrap_residuals(model, resid, wild=True):
    '''
    https://en.wikipedia.org/wiki/Bootstrapping_(statistics)#Resampling_residuals
    https://en.wikipedia.org/wiki/Bootstrapping_(statistics)#Wild_bootstrap

    Davidson & Flachaire (2008) eq.(12) gives the recommended form
    of the wild bootstrapping probability used here.

    https://doi.org/10.1016/j.jeconom.2008.08.003

    :param spec: model (e.g. best fitting spectrum)
    :param res: residuals (best_fit - observed)
    :param wild: use wild bootstrap to allow for variable errors
    :return: new model with bootstrapped residuals

    '''
    if wild:  # Wild Bootstrapping: generates -resid or resid with prob=1/2
        eps = resid * (2 * np.random.randint(2, size=resid.size) - 1)
    else:  # Standard Bootstrapping: random selection with repetition
        eps = np.random.choice(resid, size=resid.size)

    return model + eps

def read_fits_summary(fitsfile):
    hdu = fits.open(fitsfile)

    age_grid = hdu['age_grid'].data
    weights = hdu['pp_weights'].data.reshape(hdu['reg_dim'].data)

    name = hdu[0].header['name']
    z = hdu[0].header['z']
    hdu.close()
    del hdu

    wei1 = weights.sum(axis=1)
    wei1 /= wei1.sum()

    wei1_rev = copy(wei1[::-1])

    ages = age_grid[:, 0]
    ages1 = (ages[-1] - ages)[::-1] + (ages[1] - ages[0])

    agesplot = np.concatenate([np.array([0.]), ages1])
    weiplot = np.concatenate([np.array([0.]), np.cumsum(wei1_rev)])

    agesplot = np.concatenate([agesplot, np.array([agesplot[-1] + (agesplot[-1] - agesplot[-2])])])
    weiplot = np.concatenate([weiplot, np.array([weiplot[-1]])])

    nedcalc = NedCalculator(z)
    univ_age = nedcalc.zage_Gyr

    agesplot = [a if a < univ_age else univ_age for a in agesplot]

    return name, z, agesplot, weiplot, univ_age

def plot_sfh(ax, fitsfile, col_line, legend_on=False):
    name, z, agesplot, weiplot, univ_age = read_fits_summary(fitsfile)

    ax.set_title(name, fontsize=18, weight='bold')
    ax.set_xlim(0, 13.5)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('Time since BB (Gyr)', fontsize=15)
    ax.set_ylabel('Cumulative mass %', fontsize=15)
    ax.minorticks_on()
    ax.tick_params(axis='both', which='both', direction='in', labelsize=15)

    ax.axhline(0.75, color='gray', alpha=0.2)
    ax.text(12, 0.76, '75%', style='italic', color='gray')
    ax.axhline(0.95, color='gray', alpha=0.2)
    ax.text(12, 0.96, '95%', style='italic', color='gray')
    ax.axvline(3., color='gray', alpha=0.2, linestyle='-.', linewidth=2.)
    ax.axvline(univ_age, color='gray', linestyle='-.', linewidth=2.)

    ax.plot(agesplot, weiplot, color=col_line, linewidth=3.)

    ax.text(univ_age - 0.1, 0., 'today', color='gray', style='italic', rotation=90, horizontalalignment='right')
    ax.text(2.9, 0., 'z~2', style='italic', color='gray', rotation=90, horizontalalignment='right')
    ax.set_xticks([0., 2., 4., 6., 8., 10., 12.])

    if legend_on == True:
        ax.legend(frameon=False, bbox_to_anchor=(univ_age / ax.get_xlim()[1], -0.05), loc='lower right',
                  prop={'size': 12, 'weight': 'bold'}, labelspacing=0.1, labelcolor='linecolor')

    return agesplot, weiplot

def line2p(p1, p2, x):
    x1, y1 = p1
    x2, y2 = p2

    m = (y2 - y1) / (x2 - x1)
    q = -(y2 - y1) / (x2 - x1) * x1 + y1

    return m * x + q

def line2p_rev(p1, p2, y):
    x1, y1 = p1
    x2, y2 = p2

    return (y - y1) / (y2 - y1) * (x2 - x1) + x1 if y2 != y1 else x2

def get_values_from_sfh(univ_age, sfh_table, ycol):
    '''
    INPUTS:
        univ_age: the age of the universe at the redshift of the galaxy in Gyr
        sfh_table: the plotted sfh in the form of a pandas DataFrame. It needs to have a column named "time"
        ycol: is the name of the column of the sfh_table we want to use in order to retrieve the different values we want to compute

    OUTPUTS:
        y_z2: mass formed at redshift~2
        x_075: time to form 75% of the mass (t_75)
        x_090: time to form 90% of the mass (t_90)
        x_100: time to form 100% of the mass (t_fin)
        dor_90: Degree of Relicness using x_090
        dor_100: : Degree of Relicness using x_100
    '''

    tt = []
    tt90 = []
    tt100 = []

    ii = 0
    while sfh_table.iloc[ii][ycol] < 0.75:
        tt.append((ii, sfh_table.iloc[ii][ycol]))
        ii += 1
    tt.append((ii, sfh_table.iloc[ii][ycol]))

    yy = sfh_table.iloc[np.array(tt[-2:])[:, 0]]

    xx = line2p_rev(yy[['time', ycol]].iloc[0].values, yy[['time', ycol]].iloc[1].values, 0.75)

    ii = 0
    while sfh_table.iloc[ii][ycol] < 0.9:
        tt90.append((ii, sfh_table.iloc[ii][ycol]))
        ii += 1
    tt90.append((ii, sfh_table.iloc[ii][ycol]))

    yy90 = sfh_table.iloc[np.array(tt90[-2:])[:, 0]]

    xx90 = line2p_rev(yy90[['time', ycol]].iloc[0].values, yy90[['time', ycol]].iloc[1].values, 0.9)

    ii = 0
    while sfh_table.iloc[ii][ycol] < 0.998:
        tt100.append((ii, sfh_table.iloc[ii][ycol]))
        ii += 1
    tt100.append((ii, sfh_table.iloc[ii][ycol]))

    yy100 = sfh_table.iloc[np.array(tt100[-2:])[:, 0]]

    xx100 = line2p_rev(yy100[['time', ycol]].iloc[0].values, yy100[['time', ycol]].iloc[1].values, 0.998)

    tt_rev = [(0, 0)]

    for i in range(1, len(sfh_table['time'])):

        p1 = sfh_table['time'].iloc[i - 1], sfh_table[ycol].iloc[i - 1]
        p2 = sfh_table['time'].iloc[i], sfh_table[ycol].iloc[i]

        xs = np.arange(sfh_table['time'].iloc[i - 1], sfh_table['time'].iloc[i] + 0.1, 0.1)
        ys = line2p(p1, p2, xs)

        for x, y in zip(xs, ys):
            tt_rev.append((round(x, 2), y))

    tt_rev = np.array(tt_rev)

    # this is the mass formed at redshift ~2
    y_z2 = round(tt_rev[:, 1][np.where(tt_rev[:, 0] == 2.90)[0]][0], 5)

    # these are the times at 75%, 90%, and 100% formed mass
    x_075 = round(xx, 5)
    x_090 = round(xx90, 5)
    x_100 = round(xx100, 5)

    dor_90 = (y_z2 + 0.5 / x_075 + (0.7 + (univ_age - x_090) / univ_age)) / 3
    dor_100 = (y_z2 + 0.5 / x_075 + (0.7 + (univ_age - x_100) / univ_age)) / 3

    return y_z2, x_075, x_090, x_100, dor_90, dor_100

def make_catalogue(file_names, method, nrand=9):
    tie_balmer = True
    limit_doublets = True
    metal_range = [-2,0.5]

    c = 299792.458  # speed of light in km/s

    regul_err = 0.1  # Large regularization error

    vel = 0  # eq.(8) of Cappellari (2017)

    moments = [4, 2, 2]

    gas_reddening = 0 if tie_balmer else None

    logAges = []
    metals = []
    snrs = []

    # nrand = 9

    col_line1 = 'black'  # gdago: here you choose the color of the line
    col_line2 = 'green'  # gdago: here you choose the color of the line

    # Some lists to add into the catalogue later
    mass_fracs = []
    times_75 = []
    times_90 = []
    times_100 = []
    dors_100 = []
    univ_ages = []

    mass_fracsu = []
    times_75u = []
    times_90u = []
    times_100u = []
    dors_100u = []

    mass_fracsr = []
    times_75r = []
    times_90r = []
    times_100r = []
    dors_100r = []

    # New lists for alpha+0.1 results
    logAges_plus = []
    metals_plus = []
    snrs_plus = []
    # New
    mass_fracs_plus = []
    times_75_plus = []
    times_90_plus = []
    times_100_plus = []
    dors_100_plus = []

    # New lists for alpha-0.1 results
    logAges_minus = []
    metals_minus = []
    snrs_minus = []
    # New
    mass_fracs_minus = []
    times_75_minus = []
    times_90_minus = []
    times_100_minus = []
    dors_100_minus = []

    # Alpha flags to track boundary cases
    alpha_flags = []

    sigmas = []
    names = []
    alphas = []
    sigma_maxs = []
    sigma_errs=[]

    masses = []
    mass_errs = []

    radii = []
    radii_errs = []


    for i, filename in enumerate(file_names):
        print('\n')
        print("Doing:", filename)
        hdu = fits.open(filename, ignore_missing_simple=True)

        t = hdu['COADD'].data
        galaxy = t['flux']
        wave = t['wave']

        galaxy = galaxy[(wave > 3600) & (wave < 6500)]
        wave = wave[(wave > 3600) & (wave < 6500)]

        sigma = hdu[0].header['SIGMA']  # or hdu[0].header['HIERARCH SIGMA']
        alpha = hdu[0].header['ALPHA']  # or hdu[0].header['HIERARCH ALPHA']
        sigma_max = hdu[0].header['SIGMA_MAX']
        sigma_err = hdu[0].header['SIGMA_ERR']

        mass = hdu[0].header['logM']
        mass_err = hdu[0].header['errlogM']

        radius = hdu[0].header['meanRadkpc_r']
        radius_err = hdu[0].header['meanRadErrkpc_r']

        redshift = 0

        # Set alpha flag (-1 for at lower bound, 1 for at upper bound, 0 for neither)
        alpha_flag = 0
        if alpha <= 0:
            alpha_flag = -1
        elif alpha >= 0.4:
            alpha_flag = 1
        alpha_flags.append(alpha_flag)

        if alpha < 0:
            alpha = '0'
        elif alpha > 0.4:
            alpha = '4'
        else:
            alpha = str(int(alpha * 10))

        # wave = wave/(1 + redshift)  # Compute wave in the galaxy rest frame
        # Restrict wavelength range
        # galaxy = galaxy[(wave > 3600) & (wave < 6500)]
        # wave = wave[(wave > 3600) & (wave < 6500)]

        snr = DER_SNR(galaxy)  # Compute SNR
        snrs.append(snr)
        # print(snrs)
        wave *= np.median(util.vac_to_air(wave) / wave)

        noise = np.full_like(galaxy, fill_value=0.012)  # Assume constant noise per pixel here

        d_ln_lam = np.log(wave[-1] / wave[0]) / (wave.size - 1)  # Average ln_lam step
        velscale = c * d_ln_lam  # eq. (8) of Cappellari (2017)
        FWHM_gal = 2.76 / (1 + redshift)  # SDSS has an approximate instrumental resolution FWHM of 2.76A.

        # Load the SSP models
        # ssp_file = f'data/MILES_SSP/alpha{alpha}.npz'
        ssp_file = f'data/MILES_SSP_SAFE/alpha{alpha}_safe.npz'
        sps = lib.sps_lib(ssp_file, velscale, FWHM_gal, age_range=[0, NedCalculator(redshift).zage_Gyr],
                          metal_range=metal_range)

        reg_dim = sps.templates.shape[1:]
        stars_templates = sps.templates.reshape(sps.templates.shape[0], -1)

        lam_range_gal = np.array([np.min(wave), np.max(wave)])

        gas_templates, gas_names, line_wave = util.emission_lines(
            sps.ln_lam_temp, lam_range_gal, FWHM_gal, tie_balmer=tie_balmer,
            limit_doublets=limit_doublets)

        templates = np.column_stack([stars_templates, gas_templates])

        # start = [vel, sigma]     # (km/s), starting guess for [V, sigma]
        start = [vel, sigma]

        n_temps = stars_templates.shape[1]
        n_forbidden = np.sum(["[" in a for a in gas_names])  # forbidden lines contain "[*]"
        n_balmer = len(gas_names) - n_forbidden

        component = [0] * n_temps + [1] * n_balmer + [2] * n_forbidden
        gas_component = np.array(component) > 0  # gas_component=True for gas templates

        start = [start, start, start]

        # First run is just to compute a good estimate of the noise (it isn't used in the final fit)
        pp = ppxf(templates, galaxy, noise, velscale, start, moments=moments,
                  degree=-1, mdegree=8, lam=wave, lam_temp=sps.lam_temp,
                  regul=1 / regul_err, reg_dim=reg_dim, component=component,
                  gas_component=gas_component, gas_names=gas_names,
                  gas_reddening=gas_reddening, quiet=True)

        noise = noise * np.sqrt(pp.chi2)

        print("chi2 = ", pp.chi2)

        pp = ppxf(templates, galaxy, noise, velscale, start, moments=moments,
                  degree=-1, mdegree=8, lam=wave, lam_temp=sps.lam_temp,
                  regul=1 / regul_err, reg_dim=reg_dim, component=component,
                  gas_component=gas_component, gas_names=gas_names,
                  gas_reddening=gas_reddening, clean=True, quiet=True)

        weights = pp.weights[~gas_component]  # Exclude weights of the gas templates
        weights = weights.reshape(reg_dim) / weights.sum()  # Normalized

        mean_age = sps.mean_age_metal(weights, quiet=True)

        hdu_wei = fits.ImageHDU(data=weights, name='pp_weights')  # save the grid before reshaping the weights
        hdu_regdim = fits.ImageHDU(data=reg_dim, name='reg_dim')
        hdu_age = fits.ImageHDU(data=sps.age_grid, name='age_grid')
        hdu_metal = fits.ImageHDU(data=sps.metal_grid, name='metal_grid')
        bestfit = fits.ImageHDU(data=pp.bestfit, name='bestfit_spectrum')
        lam = fits.ImageHDU(data=wave, name='wavelength')
        orig = fits.ImageHDU(data=galaxy, name='original_spectrum')
        gas = fits.ImageHDU(data=pp.gas_bestfit, name='gas_bestfit')

        # Store the results in a way that is suitable for Peppe's SFH plotting

        # name = row['sexa_id'] #This is just the ID
        name = filename.replace('data/stacked_fits/', '').replace('.fits', '')
        z_orig = redshift  # This should be the original redshift of each galaxy
        # Write outputs in fits files
        hdr = fits.Header()
        # hdr['HIERARCH SDSS_ID'] = row['objid']
        hdr['HIERARCH NAME'] = name
        hdr['HIERARCH z'] = z_orig
        hdr['HIERARCH mean_age_unr'] = mean_age[0]
        hdr['HIERARCH mean_metal_unr'] = mean_age[1]
        hdr['HIERARCH Mg/Fe'] = alpha

        hdr['HIERARCH velDisp'] = sigma
        """ this is new! ^^^"""

        hdr['HIERARCH SNR'] = snr
        primary_hdu = fits.PrimaryHDU(header=hdr)
        hdulist = [primary_hdu, hdu_wei, hdu_regdim, hdu_age, hdu_metal, bestfit, lam, orig, gas]
        hdulis = fits.HDUList(hdulist)
        savepath = 'outputs/ppxf_fits/'
        if not os.path.exists(savepath):
            os.makedirs(savepath)

        hdulis.writeto(savepath + name + '_ppxfout_UNR.fits', overwrite=True)

        # Bootstrapping
        bestfit0 = pp.bestfit.copy()
        resid = galaxy - bestfit0
        start = pp.sol.copy()

        np.random.seed(123)

        weights_array = np.empty((nrand, pp.weights.size))
        for j in range(nrand):
            galaxy1 = bootstrap_residuals(bestfit0, resid)

            pp = ppxf(templates, galaxy1, noise, velscale, start, moments=moments,
                      degree=-1, mdegree=8, lam=wave, lam_temp=sps.lam_temp,
                      component=component,
                      gas_component=gas_component, gas_names=gas_names,
                      gas_reddening=gas_reddening, quiet=True)

            noise = noise * np.sqrt(pp.chi2)

            print('chi2 in loop: ',pp.chi2)

            pp = ppxf(templates, galaxy1, noise, velscale, start, moments=moments,
                      degree=-1, mdegree=8, lam=wave, lam_temp=sps.lam_temp,
                      component=component,
                      gas_component=gas_component, gas_names=gas_names,
                      gas_reddening=gas_reddening, clean=True, quiet=True)

            weights_array[j] = pp.weights

            weights = pp.weights[~gas_component]  # Exclude weights of the gas templates
            weights = weights.reshape(reg_dim) / weights.sum()  # Normalized

        plt.figure(figsize=(12, 6))
        plt.plot(wave, galaxy, 'k-', label='Data')  # Original data in black
        plt.plot(wave, pp.bestfit, 'r-', label='PPXF fit')  # Best fit in red
        plt.plot(wave, galaxy - pp.bestfit, 'g-', label='Residuals')  # Residuals in green
        plt.xlabel('Wavelength ($\AA$)')
        plt.ylabel('Flux')
        plt.title(f'PPXF Fit for {name}')
        plt.legend()
        plt.grid(True, alpha=0.2)
        plt.savefig(savepath + name + '_ppxfout_UNR_fit.png')
        plt.close()

        pp.weights = weights_array.sum(0)
        weights_err = weights_array.std(0)

        weights = pp.weights[~gas_component]  # E+fxclude weights of the gas templates
        weights = weights.reshape(reg_dim) / weights.sum()  # Normalized

        mean_age = sps.mean_age_metal(weights, quiet=True)
        sps.mean_age_metal(weights, quiet=True)
        logAges.append(mean_age[0])
        metals.append(mean_age[1])

        hdu_wei = fits.ImageHDU(data=weights, name='pp_weights')  # save the grid before reshaping the weights
        hdu_regdim = fits.ImageHDU(data=reg_dim, name='reg_dim')
        hdu_age = fits.ImageHDU(data=sps.age_grid, name='age_grid')
        hdu_metal = fits.ImageHDU(data=sps.metal_grid, name='metal_grid')
        bestfit = fits.ImageHDU(data=pp.bestfit, name='bestfit_spectrum')
        lam = fits.ImageHDU(data=wave, name='wavelength')
        orig = fits.ImageHDU(data=galaxy, name='original_spectrum')
        gas = fits.ImageHDU(data=pp.gas_bestfit, name='gas_bestfit')

        # Write outputs in fits files
        hdr = fits.Header()
        # hdr['HIERARCH SDSS_ID'] = row['objid']
        hdr['HIERARCH NAME'] = name
        hdr['HIERARCH z'] = z_orig
        hdr['HIERARCH mean_age_unr'] = mean_age[0]
        hdr['HIERARCH mean_metal_unr'] = mean_age[1]
        hdr['HIERARCH Mg/Fe'] = alpha

        hdr['HIERARCH velDisp'] = sigma  # Use the sigma value you already have
        """^^^^"""

        hdr['HIERARCH SNR'] = snr
        primary_hdu = fits.PrimaryHDU(header=hdr)
        hdulist = [primary_hdu, hdu_wei, hdu_regdim, hdu_age, hdu_metal, bestfit, lam, orig, gas]
        hdulis = fits.HDUList(hdulist)
        hdulis.writeto(savepath + name + '_ppxfout_REGUL.fits', overwrite=True)

        ## Plotting SFHs for each galaxy (but this section not needed for fitting stel pop parameters)

        plt.figure(figsize=(6, 4))
        ax1 = plt.subplot(1, 1, 1)

        ppxfout_file_unr = savepath + name + '_ppxfout_UNR.fits'
        ppxfout_file = savepath + name + '_ppxfout_REGUL.fits'

        agesplot, weiplot = plot_sfh(ax1, ppxfout_file_unr, col_line1, legend_on=False)
        agesplot, weiplot = plot_sfh(ax1, ppxfout_file, col_line2, legend_on=False)

        savepath = 'outputs/sfh_plots/'
        if not os.path.exists(savepath):
            os.makedirs(savepath)

        plt.tight_layout()
        plt.subplots_adjust(wspace=0.1)
        plt.savefig(savepath + basename(ppxfout_file).replace('.fits', '.png'), dpi=120)
        plt.close()

        ## Computing DoR values (also not needed for fitting stel pop parameters)

        name1, z1, agesplot1, weiplot1, univ_age1 = read_fits_summary(ppxfout_file_unr)
        name2, z2, agesplot2, weiplot2, univ_age2 = read_fits_summary(ppxfout_file)
        df_out = pd.DataFrame({'time': agesplot1, 'regul0': weiplot1, 'regul_max': weiplot2})
        y_z2u, x_075u, x_090u, x_100u, dor_90u, dor_100u = get_values_from_sfh(univ_age1, df_out, "regul0")
        y_z2r, x_075r, x_090r, x_100r, dor_90r, dor_100r = get_values_from_sfh(univ_age2, df_out, "regul_max")

        mass_fracsu.append(y_z2u)
        times_75u.append(x_075u)
        times_90u.append(x_090u)
        times_100u.append(x_100u)
        dors_100u.append(dor_100u)

        mass_fracsr.append(y_z2r)
        times_75r.append(x_075r)
        times_90r.append(x_090r)
        times_100r.append(x_100r)
        dors_100r.append(dor_100r)

        y_z2 = min(y_z2u, y_z2r)
        x_075 = max(x_075u, x_075r)
        x_090 = max(x_090u, x_090r)
        x_100 = max(x_100u, x_100r)
        dor_100 = (y_z2 + 0.5 / x_075 + (0.7 + (univ_age1 - x_100)) / univ_age1) / 3

        mass_fracs.append(y_z2)
        times_75.append(x_075)
        times_90.append(x_090)
        times_100.append(x_100)
        dors_100.append(dor_100)
        univ_ages.append(univ_age1)

        sigmas.append(sigma)
        alphas.append(int(alpha) / 10)
        names.append(filename)
        sigma_maxs.append(sigma_max)
        sigma_errs.append(sigma_err)

        masses.append(mass)
        mass_errs.append(mass_err)

        radii.append(radius)
        radii_errs.append(radius_err)

        ## Now add the alpha+0.1 and alpha-0.1 calculations
        # Get the original alpha value as a float for calculations
        alpha_orig = int(alpha) / 10

        # First, alpha+0.1 calculation (if applicable)
        if alpha_orig < 0.4:  # Only do this if we're not already at the upper bound
            # Calculate new alpha value
            alpha_plus = alpha_orig + 0.1

            # If at lower bound with flag, set to upper bound
            if alpha_orig == 0 and alpha_flag == -1:
                alpha_plus = 0.4

            alpha_plus_str = str(int(alpha_plus * 10))

            # Load the SSP models with new alpha
            # ssp_file_plus = f'data/MILES_SSP/alpha{alpha_plus_str}.npz'
            ssp_file_plus = f'data/MILES_SSP_SAFE/alpha{alpha_plus_str}_safe.npz'

            try:
                sps_plus = lib.sps_lib(ssp_file_plus, velscale, FWHM_gal,
                                       age_range=[0, NedCalculator(redshift).zage_Gyr],
                                       metal_range=metal_range)

                reg_dim_plus = sps_plus.templates.shape[1:]
                stars_templates_plus = sps_plus.templates.reshape(sps_plus.templates.shape[0], -1)

                gas_templates_plus, gas_names_plus, line_wave_plus = util.emission_lines(
                    sps_plus.ln_lam_temp, lam_range_gal, FWHM_gal, tie_balmer=tie_balmer,
                    limit_doublets=limit_doublets)

                templates_plus = np.column_stack([stars_templates_plus, gas_templates_plus])

                # start_plus = [start, start, start]
                start_plus = start
                component_plus = [0] * stars_templates_plus.shape[1] + [1] * n_balmer + [2] * n_forbidden
                gas_component_plus = np.array(component_plus) > 0

                # First calculate noise for alpha+0.1
                pp_plus = ppxf(templates_plus, galaxy, noise, velscale, start_plus, moments=moments,
                               degree=-1, mdegree=8, lam=wave, lam_temp=sps_plus.lam_temp,
                               regul=1 / regul_err, reg_dim=reg_dim_plus, component=component_plus,
                               gas_component=gas_component_plus, gas_names=gas_names_plus,
                               gas_reddening=gas_reddening, quiet=True)

                noise_plus = noise * np.sqrt(pp_plus.chi2)

                # Then full fit with corrected noise
                pp_plus = ppxf(templates_plus, galaxy, noise_plus, velscale, start_plus, moments=moments,
                               degree=-1, mdegree=8, lam=wave, lam_temp=sps_plus.lam_temp,
                               regul=1 / regul_err, reg_dim=reg_dim_plus, component=component_plus,
                               gas_component=gas_component_plus, gas_names=gas_names_plus,
                               gas_reddening=gas_reddening, clean=True, quiet=True)

                weights_plus = pp_plus.weights[~gas_component_plus]
                weights_plus = weights_plus.reshape(reg_dim_plus) / weights_plus.sum()

                mean_age_plus = sps_plus.mean_age_metal(weights_plus, quiet=True)
                logAges_plus.append(mean_age_plus[0])
                metals_plus.append(mean_age_plus[1])
                snrs_plus.append(snr)  # Using the same SNR as original fit

                # Save the results
                hdu_wei = fits.ImageHDU(data=weights_plus, name='pp_weights')
                hdu_regdim = fits.ImageHDU(data=reg_dim_plus, name='reg_dim')
                hdu_age = fits.ImageHDU(data=sps_plus.age_grid, name='age_grid')
                hdu_metal = fits.ImageHDU(data=sps_plus.metal_grid, name='metal_grid')
                bestfit = fits.ImageHDU(data=pp_plus.bestfit, name='bestfit_spectrum')
                lam = fits.ImageHDU(data=wave, name='wavelength')
                orig = fits.ImageHDU(data=galaxy, name='original_spectrum')
                gas = fits.ImageHDU(data=pp_plus.gas_bestfit, name='gas_bestfit')

                hdr = fits.Header()
                hdr['HIERARCH NAME'] = name
                hdr['HIERARCH z'] = z_orig
                hdr['HIERARCH mean_age_unr'] = mean_age_plus[0]
                hdr['HIERARCH mean_metal_unr'] = mean_age_plus[1]
                hdr['HIERARCH Mg/Fe'] = alpha_plus_str  # Using string version for consistency
                hdr['HIERARCH velDisp'] = sigma
                hdr['HIERARCH SNR'] = snr

                primary_hdu = fits.PrimaryHDU(header=hdr)
                hdulist = [primary_hdu, hdu_wei, hdu_regdim, hdu_age, hdu_metal, bestfit, lam, orig, gas]
                hdulis = fits.HDUList(hdulist)

                savepath = 'outputs/ppxf_fits/'
                hdulis.writeto(savepath + name + '_ppxfout_plus.fits', overwrite=True)

                # Now compute SFH parameters for alpha+0.1
                try:
                    ppxfout_file_plus = savepath + name + '_ppxfout_plus.fits'
                    name_plus, z_plus, agesplot_plus, weiplot_plus, univ_age_plus = read_fits_summary(ppxfout_file_plus)
                    df_out_plus = pd.DataFrame({'time': agesplot_plus, 'regul0': weiplot_plus})
                    y_z2_plus, x_075_plus, x_090_plus, x_100_plus, dor_90_plus, dor_100_plus = get_values_from_sfh(
                        univ_age_plus, df_out_plus, "regul0")

                    # Store the values
                    mass_fracs_plus.append(y_z2_plus)
                    times_75_plus.append(x_075_plus)
                    times_90_plus.append(x_090_plus)
                    times_100_plus.append(x_100_plus)
                    dors_100_plus.append(dor_100_plus)
                except Exception as e:
                    print(f"Error in SFH calculation for alpha+0.1 {name}: {e}")
                    mass_fracs_plus.append(None)
                    times_75_plus.append(None)
                    times_90_plus.append(None)
                    times_100_plus.append(None)
                    dors_100_plus.append(None)

            except Exception as e:
                print(f"Error in alpha+0.1 calculation for {name}: {e}")
                # Append None values if calculation fails
                logAges_plus.append(None)
                metals_plus.append(None)
                snrs_plus.append(None)
                mass_fracs_plus.append(None)
                times_75_plus.append(None)
                times_90_plus.append(None)
                times_100_plus.append(None)
                dors_100_plus.append(None)
        else:
            # If already at upper bound, just append None or duplicate values
            logAges_plus.append(None)
            metals_plus.append(None)
            snrs_plus.append(None)
            mass_fracs_plus.append(None)
            times_75_plus.append(None)
            times_90_plus.append(None)
            times_100_plus.append(None)
            dors_100_plus.append(None)

        # Second, alpha-0.1 calculation (if applicable)
        if alpha_orig > 0:  # Only do this if we're not already at the lower bound
            # Calculate new alpha value
            alpha_minus = alpha_orig - 0.1

            # If at upper bound with flag, set to lower bound
            if alpha_orig == 0.4 and alpha_flag == 1:
                alpha_minus = 0

            alpha_minus_str = str(int(alpha_minus * 10))

            # Load the SSP models with new alpha
            # ssp_file_minus = f'data/MILES_SSP/alpha{alpha_minus_str}.npz'
            ssp_file_minus = f'data/MILES_SSP_SAFE/alpha{alpha_minus_str}_safe.npz'

            try:
                sps_minus = lib.sps_lib(ssp_file_minus, velscale, FWHM_gal,
                                        age_range=[0, NedCalculator(redshift).zage_Gyr],
                                        metal_range=metal_range)

                reg_dim_minus = sps_minus.templates.shape[1:]
                stars_templates_minus = sps_minus.templates.reshape(sps_minus.templates.shape[0], -1)

                gas_templates_minus, gas_names_minus, line_wave_minus = util.emission_lines(
                    sps_minus.ln_lam_temp, lam_range_gal, FWHM_gal, tie_balmer=tie_balmer,
                    limit_doublets=limit_doublets)

                templates_minus = np.column_stack([stars_templates_minus, gas_templates_minus])

                # start_minus = [start, start, start]
                start_minus = start
                component_minus = [0] * stars_templates_minus.shape[1] + [1] * n_balmer + [2] * n_forbidden
                gas_component_minus = np.array(component_minus) > 0

                # First calculate noise for alpha-0.1
                pp_minus = ppxf(templates_minus, galaxy, noise, velscale, start_minus, moments=moments,
                                degree=-1, mdegree=8, lam=wave, lam_temp=sps_minus.lam_temp,
                                regul=1 / regul_err, reg_dim=reg_dim_minus, component=component_minus,
                                gas_component=gas_component_minus, gas_names=gas_names_minus,
                                gas_reddening=gas_reddening, quiet=True)

                noise_minus = noise * np.sqrt(pp_minus.chi2)

                # Then full fit with corrected noise
                pp_minus = ppxf(templates_minus, galaxy, noise_minus, velscale, start_minus, moments=moments,
                                degree=-1, mdegree=8, lam=wave, lam_temp=sps_minus.lam_temp,
                                regul=1 / regul_err, reg_dim=reg_dim_minus, component=component_minus,
                                gas_component=gas_component_minus, gas_names=gas_names_minus,
                                gas_reddening=gas_reddening, clean=True, quiet=True)

                weights_minus = pp_minus.weights[~gas_component_minus]
                weights_minus = weights_minus.reshape(reg_dim_minus) / weights_minus.sum()

                mean_age_minus = sps_minus.mean_age_metal(weights_minus, quiet=True)
                logAges_minus.append(mean_age_minus[0])
                metals_minus.append(mean_age_minus[1])
                snrs_minus.append(snr)  # Using the same SNR as original fit

                # Save the results
                hdu_wei = fits.ImageHDU(data=weights_minus, name='pp_weights')
                hdu_regdim = fits.ImageHDU(data=reg_dim_minus, name='reg_dim')
                hdu_age = fits.ImageHDU(data=sps_minus.age_grid, name='age_grid')
                hdu_metal = fits.ImageHDU(data=sps_minus.metal_grid, name='metal_grid')
                bestfit = fits.ImageHDU(data=pp_minus.bestfit, name='bestfit_spectrum')
                lam = fits.ImageHDU(data=wave, name='wavelength')
                orig = fits.ImageHDU(data=galaxy, name='original_spectrum')
                gas = fits.ImageHDU(data=pp_minus.gas_bestfit, name='gas_bestfit')

                hdr = fits.Header()
                hdr['HIERARCH NAME'] = name
                hdr['HIERARCH z'] = z_orig
                hdr['HIERARCH mean_age_unr'] = mean_age_minus[0]
                hdr['HIERARCH mean_metal_unr'] = mean_age_minus[1]
                hdr['HIERARCH Mg/Fe'] = alpha_minus_str  # Using string version for consistency
                hdr['HIERARCH velDisp'] = sigma
                hdr['HIERARCH SNR'] = snr

                primary_hdu = fits.PrimaryHDU(header=hdr)
                hdulist = [primary_hdu, hdu_wei, hdu_regdim, hdu_age, hdu_metal, bestfit, lam, orig, gas]
                hdulis = fits.HDUList(hdulist)

                savepath = 'outputs/ppxf_fits/'
                hdulis.writeto(savepath + name + '_ppxfout_minus.fits', overwrite=True)
                # Now compute SFH parameters for alpha-0.1
                try:
                    ppxfout_file_minus = savepath + name + '_ppxfout_minus.fits'
                    name_minus, z_minus, agesplot_minus, weiplot_minus, univ_age_minus = read_fits_summary(
                        ppxfout_file_minus)
                    df_out_minus = pd.DataFrame({'time': agesplot_minus, 'regul0': weiplot_minus})
                    y_z2_minus, x_075_minus, x_090_minus, x_100_minus, dor_90_minus, dor_100_minus = get_values_from_sfh(
                        univ_age_minus, df_out_minus, "regul0")

                    # Store the values
                    mass_fracs_minus.append(y_z2_minus)
                    times_75_minus.append(x_075_minus)
                    times_90_minus.append(x_090_minus)
                    times_100_minus.append(x_100_minus)
                    dors_100_minus.append(dor_100_minus)
                except Exception as e:
                    print(f"Error in SFH calculation for alpha-0.1 {name}: {e}")
                    mass_fracs_minus.append(None)
                    times_75_minus.append(None)
                    times_90_minus.append(None)
                    times_100_minus.append(None)
                    dors_100_minus.append(None)

            except Exception as e:
                print(f"Error in alpha-0.1 calculation for {name}: {e}")
                # Append None values if calculation fails
                logAges_minus.append(None)
                metals_minus.append(None)
                snrs_minus.append(None)
                mass_fracs_minus.append(None)
                times_75_minus.append(None)
                times_90_minus.append(None)
                times_100_minus.append(None)
                dors_100_minus.append(None)
        else:
            # If already at lower bound, just append None or duplicate values
            logAges_minus.append(None)
            metals_minus.append(None)
            snrs_minus.append(None)
            mass_fracs_minus.append(None)
            times_75_minus.append(None)
            times_90_minus.append(None)
            times_100_minus.append(None)
            dors_100_minus.append(None)

    ## Save everything - this should be outside the loop
    df_dor = pd.DataFrame()
    df_dor['logAge'] = logAges
    df_dor['[M/H]'] = metals
    df_dor['SNR'] = snrs

    df_dor['mass_frac'] = mass_fracs
    df_dor['time_75'] = times_75
    df_dor['time_90'] = times_90
    df_dor['time_100'] = times_100
    df_dor['dor_100'] = dors_100
    df_dor['univ_age'] = univ_ages

    df_dor['mass_frac_reg'] = mass_fracsr
    df_dor['time_75_reg'] = times_75r
    df_dor['time_90_reg'] = times_90r
    df_dor['time_100_reg'] = times_100r
    df_dor['dor_100_reg'] = dors_100r

    df_dor['mass_frac_unr'] = mass_fracsu
    df_dor['time_75_unr'] = times_75u
    df_dor['time_90_unr'] = times_90u
    df_dor['time_100_unr'] = times_100u
    df_dor['dor_100_unr'] = dors_100u

    # Add alpha+0.1 results
    df_dor['logAge_plus'] = logAges_plus
    df_dor['[M/H]_plus'] = metals_plus
    df_dor['SNR_plus'] = snrs_plus
    df_dor['mass_frac_plus'] = mass_fracs_plus
    df_dor['time_75_plus'] = times_75_plus
    df_dor['time_90_plus'] = times_90_plus
    df_dor['time_100_plus'] = times_100_plus
    df_dor['dor_100_plus'] = dors_100_plus

    # Add alpha-0.1 results
    df_dor['logAge_minus'] = logAges_minus
    df_dor['[M/H]_minus'] = metals_minus
    df_dor['SNR_minus'] = snrs_minus
    df_dor['mass_frac_min'] = mass_fracs_minus
    df_dor['time_75_min'] = times_75_minus
    df_dor['time_90_min'] = times_90_minus
    df_dor['time_100_min'] = times_100_minus
    df_dor['dor_100_min'] = dors_100_minus

    # Add alpha flag to indicate boundary values
    df_dor['alpha_flag'] = alpha_flags

    # Add velocity dispersion, alpha, and filenames
    df_dor['vel_disp_avg'] = sigmas
    df_dor['vel_disp_err'] = sigma_errs
    df_dor['vel_disp_max'] = sigma_maxs
    df_dor['alpha'] = alphas
    df_dor['filename'] = names

    df_dor['logM'] = masses
    df_dor['errlogM'] = mass_err

    df_dor['meanRadkpc_r'] = radii
    df_dor['meanRadErrkpc_r'] = radii_errs

    # Add the dor column needed for plotting
    df_dor['dor'] = df_dor['dor_100']

    # Create directory if it doesn't exist
    os.makedirs('outputs/stacked_catalogues', exist_ok=True)

    # Save the catalog with all the original and new measurements
    df_dor.to_csv(f'outputs/stacked_catalogues/CATALOGUE_{method}.csv', index=False)








def combine_catalogues():
    all_dfs = []
    catalogue_dir = 'outputs/stacked_catalogues'
    for file in os.listdir(catalogue_dir):
        if file.startswith('CATALOGUE_') and file.endswith('.csv'):
            df = pd.read_csv(os.path.join(catalogue_dir, file))
            method = file.replace('CATALOGUE_', '').replace('.csv', '')

            # Add a column for the method
            df['method'] = method

            # Add to our list of dataframes
            all_dfs.append(df)

    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df.to_excel('outputs/stacked_catalogues/Combined_DoR_Catalogues.xlsx', index=False)
    print(f"Combined {len(all_dfs)} catalogues into 'Combined_DoR_Catalogues.xlsx'")


def fit_spectra(nrand):
    # make output directories if they don't exist
    directories = ['outputs/ppxf_fits', 'outputs/sfh_plots', 'outputs/stacked_catalogues']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

    # Get all stacked FITS files and organize them by method
    stacked_dir = "data/stacked_fits"
    fits_files = [f for f in os.listdir(stacked_dir) if f.startswith("stacked_") and f.endswith(".fits")]

    # Group files by clustering method
    clustering_methods = {}
    for fits_file in fits_files:
        # Remove 'stacked_' prefix and '.fits' suffix
        base_name = fits_file[8:-5]  # e.g., "DoR_0" from "stacked_DoR_0.fits"

        # Extract method name (everything before the last underscore)
        method = base_name.rsplit('_', 1)[0]  # e.g., "DoR" from "DoR_0"

        # Initialize list for method if not exists
        if method not in clustering_methods:
            clustering_methods[method] = []

        # Add full base name to method's list
        clustering_methods[method].append(base_name)

    # Sort the lists within each method to ensure consistent ordering
    for method in clustering_methods:
        clustering_methods[method].sort()

    # Process each method
    for method in clustering_methods:
        print(f"\n================================")
        print(f"Fitting {method}'s clusters")
        file_names = []
        files_list = clustering_methods[method]
        for file in files_list:
            file = os.path.join(stacked_dir, f"stacked_{file}.fits")
            file_names.append(file)
        make_catalogue(file_names, method, nrand)
        print(f"\n================================")

    combine_catalogues()