from astropy.table import Table
from kcorrect.kcorrect import Kcorrect
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Polynomial

def kcorr_gkv(dataframe, zrange = [0, 2], z0 = 0, pdeg = 4, ntest = 0, responses = ['galex_FUV', 'galex_NUV', 'vst_u', 'vst_g', 'vst_r', 'vst_i', 'vista_z', 'vista_y', 'vista_j', 'vista_h', 'vista_k', 'wise_w1', 'wise_w2'], fnames = ['flux_FUVt', 'flux_NUVt', 'flux_ut', 'flux_gt', 'flux_rt', 'flux_it', 'flux_Zt', 'flux_Yt', 'flux_Jt', 'flux_Ht', 'flux_Kt', 'flux_W1t', 'flux_W2t'], ferrnames = ['flux_err_FUVt', 'flux_err_NUVt', 'flux_err_ut', 'flux_err_gt', 'flux_err_rt', 'flux_err_it', 'flux_err_Zt', 'flux_err_Yt', 'flux_err_Jt', 'flux_err_Ht', 'flux_err_Kt', 'flux_err_W1t', 'flux_err_W2t'], rband = 'flux_rt', zband = 'flux_Zt', redshift = 'Z', survey = 'GAMAIII'):
    """K-corrections for GAMA-KiDS-VIKING (GKV) catalogues."""
    
    kc = Kcorrect(responses = responses)

    sel = (dataframe['NQ'] > 2) * (dataframe[redshift] > zrange[0]) * (dataframe[redshift] < zrange[1])
    dataframe = dataframe[sel]
    dataframe.reset_index(drop=True, inplace=True)
    if ntest:
        dataframe = dataframe[:ntest]
    ngal = len(dataframe)
    redshift = dataframe[redshift]
    
    nband = len(responses)
    flux = np.zeros((ngal, nband))
    flux_err, ivar = np.zeros((ngal, nband)), np.zeros((ngal, nband))
    i = 0
    for j in range(len(fnames)):
        flux[:, i] = dataframe[fnames[j]]
        flux_err[:, i] = dataframe[ferrnames[j]]
        ivar[:, i] = flux_err[:, i]**-2
        i += 1

    # For missing bands, set flux and ivar both to zero
    if survey == 'GAMAII':
        fix = (flux > 1e10) + (flux < -9000) + (flux_err <= 0)
    else :
        fix = (flux > 1e10) + (flux < -900) + (flux_err <= 0)
    # if len(flux[fix]) > 0:
    #     pdb.set_trace()
    flux[fix] = 0
    ivar[fix] = 0
    nfix = len(flux[fix])
    print('Fixed ', len(flux[fix]), 'missing fluxes')

    # Fit SED coeffs
    coeffs = kc.fit_coeffs(redshift, flux, ivar)

    # For galaxies that couldn't be fit, use average SED of galaxies close in redshift and r-z colour
    ztol = 0.1
    rz = dataframe[rband]/dataframe[zband]
    bad = np.nonzero(coeffs.sum(axis=-1) == 0)[0]
    nbad = len(bad)
    if nbad > 0:
        print('Replacing', nbad, 'bad fits with mean')
        plt.clf()
        ax = plt.subplot(111)
        plt.xlabel('Band')
        plt.ylabel('Flux')
        for ibad in bad:
            close = np.nonzero((abs(redshift - redshift[ibad]) < ztol) *
                             (0.9 < rz[ibad]/rz) * (rz[ibad]/rz < 1.1))[0]
            # flux_mean = flux[close, :].mean(axis=-1)
            # ivar_mean = ivar[close, :].sum(axis=-1)
            flux_mean = flux[close, :].mean(axis=0)
            ivar_mean = ivar[close, :].sum(axis=0)
            coeffs[ibad, :] = kc.fit_coeffs(redshift[ibad], flux_mean, ivar_mean)
            color = next(ax._get_lines.prop_cycler)['color']
            plt.errorbar(range(len(fnames)), flux_mean, yerr=ivar_mean**-0.5, color=color)
            plt.plot(range(len(fnames)), flux_mean, color=color)
        plt.show()
    

    # Calculate and plot the k-corrections
    k = kc.kcorrect(redshift=redshift, coeffs=coeffs, band_shift=z0)
    fig, axes = plt.subplots(3, 5, sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0, wspace=0)
    for iband in range(nband):
        ax = axes.flatten()[iband]
        ax.scatter(redshift, k[:, iband], s=0.1)
        ax.text(0.5, 0.8, fnames[iband], transform=ax.transAxes)
    axes[2, 2].set_xlabel('Redshift')
    axes[1, 0].set_ylabel('K-correction')
    plt.show()

    # Polynomial fits to reconstructed r-band K-correction K_r(z)
    # We fit K + 2.5 log10(1+z0) to z-z0 with constant coefficient set at zero,
    # and then set coef[0] = -2.5 log10(1+z0), so that resulting fits pass
    # through (z0, -2.5 log10(1+z0))
    nz = 100
    redshifts = np.linspace(*zrange, nz)
    pcoeffs = np.zeros((ngal, pdeg+1))
    pcoeffs[:, 0] = -2.5*np.log10(1+z0)
    nplot = 10

    plt.clf()
    ax = plt.subplot(111)
    plt.xlabel('Redshift')
    plt.ylabel('K_r(z)')
    iband = 4
    deg = np.arange(1, pdeg+1)
    for igal in range(ngal):
        kz = kc.kcorrect(redshift=redshifts,
                         coeffs=np.broadcast_to(coeffs[igal, :], (nz, 5)),
                         band_shift=z0)
        pc = Polynomial.fit(redshifts-z0, kz[:, iband] + 2.5*np.log10(1+z0),
                            deg=deg, domain=zrange, window=zrange)
        pcoeffs[igal, 1:] = pc.coef[1:]
        if (igal < nplot):
            fit = pc(redshifts-z0) - 2.5*np.log10(1+z0)
            color = next(ax._get_lines.prop_cycler)['color']
            plt.scatter(redshifts, kz[:, iband], s=1, color=color)
            plt.plot(redshifts, fit, '-', color=color)
    index = fnames.index(rband)
    dataframe['Kcorrection'] = k.tolist()
    dataframe['r_Kcorrection'] = [x[index] for x in k]
    dataframe['pcoeffs'] = pcoeffs.tolist()
    dataframe['coeffs'] = coeffs.tolist()
    plt.show()
    return dataframe