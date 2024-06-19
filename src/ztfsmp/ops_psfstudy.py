#!/usr/bin/env python3

from ztfsmp.pipeline import register_op

def psfstudy_map(exposure, logger, args, op_args):
    import pickle

    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    from numpy.polynomial.polynomial import Polynomial

    matplotlib.use('Agg')

    psf_df = exposure.get_matched_catalog('psfstars')
    aper_df = exposure.get_matched_catalog('aperstars')
    gaia_df = exposure.get_matched_ext_catalog('gaia')

    aper_str = op_args['aperflux']
    eaper_str = "e{}".format(aper_str)

    # Remove negative flux
    mask = ~np.any([psf_df['flux']<=0., aper_df[aper_str]<=0.], axis=0)
    psf_df = psf_df.loc[mask]
    aper_df = aper_df.loc[mask]
    gaia_df = gaia_df.loc[mask]

    # Compute magnitude differences between PSF and aperture photometry
    delta_mag = -2.5*np.log10(psf_df['flux']) + 2.5*np.log10(aper_df[aper_str])
    delta_emag = np.sqrt((-1.08*psf_df['eflux']/psf_df['flux'])**2+(-1.08*aper_df[eaper_str]/aper_df[aper_str])**2)

    # Compute polynomials
    poly0, ([poly0_chi2], _, _, _) = Polynomial.fit(gaia_df['Gmag'], delta_mag, 0, w=1./delta_emag, full=True)
    poly1, ([poly1_chi2], _, _, _) = Polynomial.fit(gaia_df['Gmag'], delta_mag, 1, w=1./delta_emag, full=True)
    poly2, ([poly2_chi2], _, _, _) = Polynomial.fit(gaia_df['Gmag'], delta_mag, 2, w=1./delta_emag, full=True)

    # Compute chi2
    poly0_chi2 = poly0_chi2/(len(gaia_df)-1)
    poly1_chi2 = poly1_chi2/(len(gaia_df)-2)
    poly2_chi2 = poly2_chi2/(len(gaia_df)-3)

    # Do a nice plot
    G_min, G_max = gaia_df['Gmag'].min(), gaia_df['Gmag'].max()
    G_linspace = np.linspace(G_min, G_max)

    plt.subplots(figsize=(10., 4.))
    plt.title(exposure.name)
    plt.errorbar(gaia_df['Gmag'], delta_mag, yerr=delta_emag, ls='none', marker='.', markersize=5., lw=0.5)
    plt.plot([G_min, G_max], [poly0(G_min), poly0(G_max)], label="Constant - $\chi_\nu^2={:.2f}$".format(poly0_chi2))
    plt.plot([G_min, G_max], [poly1(G_min), poly1(G_max)], label="Linear - $\chi_\nu^2={:.2f}$".format(poly1_chi2))
    plt.plot(G_linspace, poly2(G_linspace), label="Quadratic - $\chi_\nu^2={:.2f}$".format(poly2_chi2))
    plt.legend()
    plt.ylim(-1., 1.)
    plt.grid()
    plt.savefig(exposure.path.joinpath("psfstudy_{}.png".format(exposure.name)), dpi=200.)
    plt.close()

    # Save everything on disk
    header = exposure.exposure_header
    expid = int(header['expid'])
    mjd = float(header['obsmjd'])
    ccdid = int(header['ccdid'])
    qid = int(header['qid'])
    skylev = float(header['sexsky'])
    moonillf = float(header['moonillf'])
    seeing = float(header['seeing'])
    airmass = float(header['airmass'])

    result = {}
    result['name'] = exposure.name
    result['mjd'] = mjd
    result['measure_count'] = len(gaia_df)
    result['apercat'] = aper_str
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

    with open(exposure.path.joinpath("psfstudy.pickle"), 'wb') as f:
        pickle.dump(result, f)

    return True


def psfstudy_reduce(lightcurve, logger, args, op_args):
    import pickle
    import pandas as pd
    import matplotlib.pyplot as plt

    result_paths = lightcurve.path.glob("ztf_*/psfstudy.pickle")
    results = []
    for result_path in result_paths:
        with open(result_path, 'rb') as f:
            results.append(pickle.load(f))

    df = pd.DataFrame(results).set_index('name', drop=True)

    df.to_parquet(lightcurve.path.joinpath("psfstudy.parquet"))

    return True


register_op('psfstudy', map_op=psfstudy_map, reduce_op=psfstudy_reduce, parameters={'name': 'aperflux', 'type': 'str', 'default': 'apfl7', 'desc':""})
