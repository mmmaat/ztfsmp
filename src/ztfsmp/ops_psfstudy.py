#!/usr/bin/env python3

from ztfsmp.pipeline import register_op

def psfstudy_map(exposure, logger, args, op_args):
    from saunerie.plottools import binplot

    if not exposure.path.joinpath("match_catalogs.success").exists():
        return False

    import pickle

    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    from numpy.polynomial.polynomial import Polynomial
    from ztfsmp.fit_utils import RobustPolynomialFit

    matplotlib.use('Agg')

    header = exposure.exposure_header
    expid = int(header['expid'])
    mjd = float(header['obsmjd'])
    ccdid = int(header['ccdid'])
    qid = int(header['qid'])
    skylev = float(header['sexsky'])
    moonillf = float(header['moonillf'])
    seeing = float(header['seeing'])
    airmass = float(header['airmass'])

    psf_df = exposure.get_matched_catalog('psfstars')
    aper_df = exposure.get_matched_catalog('aperstars')
    gaia_df = exposure.get_matched_ext_catalog('gaia')

    aper_str = op_args['aperflux']
    eaper_str = "e{}".format(aper_str)
    aperradius = list(set(aper_df['rad{}'.format(aper_str[-1])]))[0]

    # Remove negative flux
    mask = ~np.any([psf_df['flux']<=0., aper_df[aper_str]<=0.], axis=0)
    psf_df = psf_df.loc[mask]
    aper_df = aper_df.loc[mask]
    gaia_df = gaia_df.loc[mask]

    # Compute relevant quantities, magnitudes and difference
    psf_df = psf_df.assign(mag=-2.5*np.log10(psf_df['flux']), emag=1.08*psf_df['eflux']/psf_df['flux'])
    aper_df = psf_df.assign(mag=-2.5*np.log10(aper_df[aper_str]), emag=1.08*aper_df[eaper_str]/aper_df[aper_str])
    delta_mag = aper_df['mag'] - psf_df['mag']
    delta_emag = np.sqrt(psf_df['emag']**2+aper_df['emag']**2)

    G_min, G_max = gaia_df['Gmag'].min(), gaia_df['Gmag'].max()
    G_linspace = np.linspace(G_min, G_max)
    bins = np.arange(G_min, G_max, 1.)

    plt.subplots(figsize=(10., 4.))

    # Bin everything
    G_binned, delta_mag_binned, delta_emag_binned = binplot(gaia_df['Gmag'].to_numpy(), delta_mag.to_numpy(), robust=True, data=False, scale=True, weights=1./delta_emag.to_numpy(), bins=bins, color='red', lw=2., zorder=15)

    G_binned = np.array(G_binned)
    delta_mag_binned = np.array(delta_mag_binned)
    delta_emag_binned = np.array(delta_emag_binned)

    # Remove empty bins
    m = delta_emag_binned > 0.
    G_binned = G_binned[m]
    delta_mag_binned = delta_mag_binned[m]
    delta_emag_binned = delta_emag_binned[m]

    # Fits
    poly0, poly0_chi2 = RobustPolynomialFit(G_binned, delta_mag_binned, 0, dy=delta_emag_binned, verbose=False)
    poly1, poly1_chi2 = RobustPolynomialFit(G_binned, delta_mag_binned, 1, dy=delta_emag_binned, verbose=False)
    poly2, poly2_chi2 = RobustPolynomialFit(G_binned, delta_mag_binned, 2, dy=delta_emag_binned, verbose=False)

    # Do a nice plot
    plt.title("Aperture - PSF\n{} - {} (radius={}) - skylev={:.3f}".format(exposure.name, aper_str, aperradius, skylev))
    plt.errorbar(gaia_df['Gmag'], delta_mag, yerr=delta_emag, ls='none', marker='.', markersize=5., lw=0.5)
    plt.plot([G_min, G_max], [poly0(G_min), poly0(G_max)], label="Constant - $\\chi_\\nu^2={:.2f}$".format(poly0_chi2))
    plt.plot([G_min, G_max], [poly1(G_min), poly1(G_max)], label="Linear - $\\chi_\\nu^2={:.2f}$".format(poly1_chi2))
    plt.plot(G_linspace, poly2(G_linspace), label="Quadratic - $\\chi_\\nu^2={:.2f}$".format(poly2_chi2))
    plt.legend()
    plt.ylim(-1., 1.)
    plt.grid()
    # plt.show()
    plt.savefig(exposure.path.joinpath("psfstudy_{}_{}.png".format(aper_str, exposure.name)), dpi=200.)
    plt.close()

    # Save everything on disk

    result = {}
    result['name'] = exposure.name
    result['mjd'] = mjd
    result['measure_count'] = len(gaia_df)
    result['apercat'] = aper_str
    result['aperradius'] = aperradius
    result['poly0_0'] = poly0.coef[0]
    result['poly1_0'] = poly1.coef[0]
    result['poly1_1'] = poly1.coef[1]
    result['poly2_0'] = poly2.coef[0]
    result['poly2_1'] = poly2.coef[1]
    result['poly2_2'] = poly2.coef[2]
    result['poly0_chi2'] = poly0_chi2
    result['poly1_chi2'] = poly1_chi2
    result['poly2_chi2'] = poly2_chi2
    result['expid'] = expid
    result['ccdid'] = ccdid
    result['qid'] = qid
    result['skylev'] = skylev
    result['moonillf'] = moonillf
    result['seeing'] = seeing
    result['airmass'] = airmass

    with open(exposure.path.joinpath("psfstudy_{}.pickle".format(aper_str)), 'wb') as f:
        pickle.dump(result, f)

    return True


def psfstudy_reduce(lightcurve, logger, args, op_args):
    import pickle
    import pandas as pd
    import matplotlib.pyplot as plt

    result_paths = lightcurve.path.glob("ztf_*/psfstudy_{}.pickle".format(op_args['aperflux']))
    results = []
    for result_path in result_paths:
        with open(result_path, 'rb') as f:
            results.append(pickle.load(f))

    df = pd.DataFrame(results).set_index('name', drop=True)

    df.to_parquet(lightcurve.path.joinpath("psfstudy_{}.parquet".format(op_args['aperflux'])))

    return True


register_op('psfstudy', map_op=psfstudy_map, reduce_op=psfstudy_reduce, parameters={'name': 'aperflux', 'type': str, 'default': 'apfl7', 'desc':""})
