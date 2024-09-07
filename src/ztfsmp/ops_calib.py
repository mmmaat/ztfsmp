#!/usr/bin/env python3

from ztfsmp.pipeline import register_op


def star_averager(lightcurve, logger, args, op_args):
    import numpy as np
    import pandas as pd
    from croaks.match import NearestNeighAssoc
    from croaks import DataProxy
    from saunerie.linearmodels import LinearModel, RobustLinearSolver

    from ztfsmp.pipeline_utils import update_yaml
    from ztfsmp.listtable import ListTable

    # Load SMP star lightcurves
    logger.info("Loading stars SMP lightcurve...")
    smphot_lc_table = ListTable.from_filename(lightcurve.smphot_stars_path.joinpath("smphot_stars_cat.list"), delim_whitespace=False)
    logger.info("Found {} measurements".format(len(smphot_lc_table.df)))

    # Remove negative fluxes
    smphot_lc_df = smphot_lc_table.df.loc[smphot_lc_table.df['flux']>0.]
    logger.info("Removing negative fluxes, down to {} measurements".format(len(smphot_lc_df)))

    # Remove nans and 0's in error term
    smphot_lc_df = smphot_lc_df.dropna(subset='error')
    smphot_lc_df = smphot_lc_df.loc[smphot_lc_df['error']>0.]
    smphot_lc_df = smphot_lc_df.loc[smphot_lc_df['flux']>0.]
    smphot_lc_df = smphot_lc_df.loc[smphot_lc_df['flux']<=1e6]
    logger.info("Sanitizing measurements, down to {} measurements".format(len(smphot_lc_df)))

    # Create dataproxy for the fit
    dp = DataProxy(smphot_lc_df[['flux', 'error', 'star', 'mjd']].to_records(), flux='flux', error='error', star='star', mjd='mjd')
    dp.make_index('star')
    dp.make_index('mjd')

    w = 1./np.sqrt(dp.error**2+op_args['piedestal']**2)

    # Retrieve matching Gaia catalog to taf fitted constant stars
    gaia_df = lightcurve.get_ext_catalog('gaia', matched=False).drop_duplicates(subset='Source').set_index('Source', drop=True)
    with open(lightcurve.smphot_stars_path.joinpath("stars_gaiaid.txt"), 'r') as f:
        gaiaids = list(map(lambda x: int(x.strip()), f.readlines()))

    gaia_df = gaia_df.loc[gaiaids]

    # Fit of the constant star model
    logger.info("Building model")
    model = LinearModel(list(range(len(dp.nt))), dp.star_index, np.ones_like(dp.star, dtype=float))
    solver = RobustLinearSolver(model, dp.flux, weights=w)
    logger.info("Solving model")
    solver.model.params.free = solver.robust_solution()
    logger.info("Done")

    # Add fit imformation to the lightcurve dataframe
    smphot_lc_df = smphot_lc_df.assign(mean_flux=solver.model.params.free[dp.star_index],
                                       emean_flux=np.sqrt(solver.get_cov().diagonal())[dp.star_index],
                                       res=solver.get_res(dp.flux),
                                       bads=solver.bads)
    smphot_lc_df = smphot_lc_df.assign(mean_mag=-2.5*np.log10(smphot_lc_df['mean_flux']),
                                       emean_mag=2.5/np.log(10)*smphot_lc_df['emean_flux']/smphot_lc_df['mean_flux'])
    smphot_lc_df = smphot_lc_df.assign(mag=-2.5*np.log10(smphot_lc_df['flux']),
                                       emag=2.5/np.log(10)*smphot_lc_df['error']/smphot_lc_df['flux'])

    smphot_lc_df = smphot_lc_df.assign(wres=smphot_lc_df['res']/smphot_lc_df['error'])

    # Constant stars dataframe creation
    stars_gaiaids = gaia_df.iloc[list(dp.star_map.keys())]
    stars_df = pd.DataFrame(data={'mag': -2.5*np.log10(solver.model.params.free),
                                  'emag': 2.5/np.log(10)*np.sqrt(solver.get_cov().diagonal())/solver.model.params.free,
                                  'chi2': np.bincount(dp.star_index[~solver.bads], weights=smphot_lc_df.loc[~solver.bads]['wres']**2)/np.bincount(dp.star_index[~solver.bads]),
                                  'gaiaid': gaia_df.iloc[list(dp.star_map.keys())].index.tolist(),
                                  'star': dp.star_map.keys()})

    # Compute star lightcurve rms, FAST
    rms_flux = np.sqrt(np.bincount(dp.star_index[~solver.bads], weights=dp.flux[~solver.bads]**2)/np.bincount(dp.star_index[~solver.bads])-(np.bincount(dp.star_index[~solver.bads], weights=dp.flux[~solver.bads])/np.bincount(dp.star_index[~solver.bads]))**2)
    stars_df = stars_df.assign(rms_mag=2.5/np.log(10)*rms_flux/solver.model.params.free)

    stars_df.set_index('gaiaid', inplace=True)

    # Everything gets saved !
    logger.info("Saving to disk...")
    smphot_lc_df.to_parquet(lightcurve.smphot_stars_path.joinpath("stars_lightcurves.parquet"))
    stars_df.to_parquet(lightcurve.smphot_stars_path.joinpath("constant_stars.parquet"))
    logger.info("Done")

    # Update lightcurve yaml with fit informations
    logger.info("Updating lightcurve yaml")
    update_yaml(lightcurve.path.joinpath("lightcurve.yaml"), 'constant_stars',
                {'star_count': len(stars_df),
                 'chi2': np.sum(smphot_lc_df['wres']).item(),
                 'chi2/ndof': np.sum(smphot_lc_df['wres']).item()/len(stars_df),
                 'ndof': len(stars_df),
                 'piedestal': op_args['piedestal']})

    return True

register_op('star_averager', reduce_op=star_averager, parameters=[{'name': 'piedestal', 'type': float, 'default': 0., 'desc': "Magnitude piedestal."}])


def calib(lightcurve, logger, args, op_args):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib
    from croaks import DataProxy
    from saunerie.linearmodels import LinearModel, RobustLinearSolver, indic
    from croaks.match import match
    from saunerie.plottools import binplot
    from scipy.stats import norm
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator

    from ztfsmp.pipeline_utils import update_yaml
    from ztfsmp.ext_cat_utils import mag2extcatmag, emag2extcatemag, get_ubercal_catalog_in_cone
    from ztfsmp.misc_utils import write_ds9_reg_circles
    from ztfsmp.lightcurve import Exposure

    matplotlib.use('Agg')

    if not lightcurve.smphot_stars_path.joinpath("constant_stars.parquet").exists():
        logger.error("No constant stars catalog!")
        return False

    output_path = lightcurve.path.joinpath("calib.{}".format(op_args['photom_cat']))
    output_path.mkdir(exist_ok=True)

    # Load constant stars catalog, Gaia catalog (for star identification/matching) and external calibration catalog
    stars_df = pd.read_parquet(lightcurve.smphot_stars_path.joinpath("constant_stars.parquet"))
    gaia_df = lightcurve.get_ext_catalog('gaia').set_index('Source', drop=False).loc[stars_df.index]
    ext_cat_df = lightcurve.get_ext_catalog(op_args['photom_cat'], matched=False)

    write_ds9_reg_circles(lightcurve.path.joinpath("calib.{}/calib_stars.reg".format(op_args['photom_cat'])), gaia_df[['ra', 'dec']].to_numpy(), [10.]*len(gaia_df))

    if len(ext_cat_df) == 0:
        logger.error("Empty calibration catalog \'{}\'!".format(op_args['photom_cat']))
        return False

    i = match(gaia_df[['ra', 'dec']].to_records(), ext_cat_df[['ra', 'dec']].to_records())

    gaia_df = gaia_df.iloc[i[i>=0]].reset_index(drop=True)
    ext_cat_df = ext_cat_df.iloc[i>=0].reset_index(drop=True)


    stars_df = stars_df.loc[gaia_df['Source'].tolist()]

    # Add matched band external catalog magnitude, delta and color
    stars_df = stars_df.assign(cat_mag=ext_cat_df[mag2extcatmag[op_args['photom_cat']][lightcurve.filterid]].tolist(),
                               cat_emag=ext_cat_df[emag2extcatemag[op_args['photom_cat']][lightcurve.filterid]].tolist())
    stars_df = stars_df.assign(delta_mag=(stars_df['mag'] - stars_df['cat_mag']),
                               delta_emag=np.sqrt(stars_df['emag']**2+stars_df['cat_emag']**2+op_args['piedestal']**2))
    stars_df = stars_df.assign(cat_color=(ext_cat_df[mag2extcatmag[op_args['photom_cat']]['zg']]-ext_cat_df[mag2extcatmag[op_args['photom_cat']]['zi']]).tolist(),
                               cat_ecolor=np.sqrt(ext_cat_df[emag2extcatemag[op_args['photom_cat']]['zg']]**2+ext_cat_df[emag2extcatemag[op_args['photom_cat']]['zi']]**2).tolist())
    stars_df = stars_df.assign(gaia_color=gaia_df['BP-RP'])

    stars_df.dropna(subset=['cat_emag', 'cat_ecolor'], inplace=True)

    stars_df = stars_df.loc[stars_df['cat_color']<2.5]
    stars_df = stars_df.loc[stars_df['cat_color']>0.]

    stars_df.reset_index(inplace=True)

    # Remove nans and infs
    stars_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    stars_df.dropna(subset=['delta_mag', 'delta_emag'], inplace=True)

    w = 1./stars_df['delta_emag'].to_numpy()

    dp = DataProxy(stars_df[['delta_mag', 'delta_emag', 'cat_mag', 'cat_emag', 'gaiaid', 'cat_color', 'gaia_color']].to_records(), delta_mag='delta_mag', delta_emag='delta_emag', cat_mag='cat_mag', cat_emag='cat_emag', gaiaid='gaiaid', cat_color='cat_color', gaia_color='gaia_color')
    dp.make_index('gaiaid')

    def _build_model(dp):
        model = indic(np.zeros(len(dp.nt), dtype=int), val=dp.cat_color, name='color') + indic(np.zeros(len(dp.nt), dtype=int), name='zp')
        return RobustLinearSolver(model, dp.delta_mag, weights=w)

    def _solve_model(solver):
        solver.model.params.free = solver.robust_solution(local_param='color')

    solver = _build_model(dp)
    _solve_model(solver)

    res = solver.get_res(dp.delta_mag)[~solver.bads]
    wres = res/dp.delta_emag[~solver.bads]

    # Extract fitted parameters
    ZP = solver.model.params['zp'].full.item()
    alpha = solver.model.params['color'].full.item()

    chi2ndof = np.sum(wres**2)/(len(dp.nt)-2-sum(solver.bads)) # 2 parameters in the model

    dp.compress(~solver.bads)

    # Compute residual dispertion of bright stars
    bright_stars_threshold = 18.
    bright_stars_mask = (dp.cat_mag<=bright_stars_threshold)
    bright_stars_mu, bright_stars_std = np.mean(res[bright_stars_mask]), np.std(res[bright_stars_mask])


    # Chromaticity effect plot
    plt.subplots(ncols=1, nrows=1, figsize=(8., 5.))
    plt.title("{}-{}\nColor term of the {} catalog\n $\chi^2/\mathrm{{ndof}}={:.4f}$".format(lightcurve.name, lightcurve.filterid, op_args['photom_cat'], chi2ndof))
    plt.errorbar(dp.cat_color, dp.delta_mag-ZP, yerr=dp.delta_emag, fmt='.')
    plt.plot([np.min(dp.cat_color), np.max(dp.cat_color)], [alpha*np.min(dp.cat_color), alpha*np.max(dp.cat_color)], label="$\\alpha C_\\mathrm{{{}}}, \\alpha=${:.4f}".format(op_args['photom_cat'], alpha))
    plt.legend()
    plt.xlabel("$C_\mathrm{{{}}}$ [mag]".format(op_args['photom_cat']))
    plt.ylabel("$m_\mathrm{{ZTF}}-m_\mathrm{{{}}}-ZP$ [mag]".format(op_args['photom_cat']))
    plt.grid()
    plt.tight_layout()
    plt.savefig(output_path.joinpath("res_chromaticity.png"), dpi=200.)
    plt.close()

    # magerr = np.sqrt(dp.emag**2+(alpha*dp.cat_ecolor)**2)
    plt.subplots(ncols=1, nrows=1, figsize=(8., 5.))
    plt.title("{}-{}\nResidual plot as a function of star color\n$\chi^2/\mathrm{{ndof}}={:.4f}$".format(lightcurve.name, lightcurve.filterid, chi2ndof))
    plt.errorbar(dp.cat_color, res, yerr=dp.delta_emag, fmt='.')
    plt.grid()
    plt.xlabel("$C_\mathrm{{{}}}$ [mag]".format(op_args['photom_cat']))
    plt.ylabel("$m_\mathrm{{ZTF}}-m_\mathrm{{{cat}}}-ZP-\\alpha C_\\mathrm{{{cat}}}$ [mag]".format(cat=op_args['photom_cat']))
    plt.tight_layout()
    plt.savefig(output_path.joinpath("res_color.png"), dpi=200.)
    plt.close()

    plt.subplots(ncols=1, nrows=1, figsize=(8., 5.))
    plt.title("{}-{}\nResidual plot for the calibration fit onto {}\n$ZP$={:.4f}, $\chi^2/\mathrm{{ndof}}={:.4f}$, Star count={}".format(lightcurve.name, lightcurve.filterid, op_args['photom_cat'], ZP, chi2ndof, len(stars_df)))
    xmin, xmax = np.min(dp.cat_mag)-0.2, np.max(dp.cat_mag)+0.2
    plt.errorbar(dp.cat_mag, res, yerr=dp.delta_emag, fmt='.')
    # plt.scatter(dp.cat_mag, res, c=dp.cat_color, zorder=10., s=6.)
    # plt.colorbar()
    plt.fill_between([xmin, bright_stars_threshold], [bright_stars_mu-bright_stars_std]*2, [bright_stars_mu+bright_stars_std]*2, color='xkcd:sky blue', alpha=0.4, label='Bright stars - $\sigma_\mathrm{{res}}={:.4f}$'.format(bright_stars_std))
    plt.xlim(xmin, xmax)
    plt.ylim(-0.2, 0.2)
    plt.grid()
    plt.xlabel("$m_\mathrm{{{}}}$ [mag]".format(op_args['photom_cat']))
    plt.ylabel("$m_\mathrm{{ZTF}}-m_\mathrm{{{}}}-ZP-\\alpha C$ [mag]".format(op_args['photom_cat']))
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path.joinpath("res.png"), dpi=200.)
    plt.close()


    # Binplot of the residuals
    plt.subplots(ncols=1, nrows=2, figsize=(12., 8.), sharex=True, gridspec_kw={'hspace': 0.})
    plt.suptitle("{}-{}\nResidual plot for the calibration fit onto {}\n$ZP$={:.4f}, $\chi^2/\mathrm{{ndof}}={:.4f}$, Star count={}".format(lightcurve.name, lightcurve.filterid, op_args['photom_cat'], ZP, chi2ndof, len(stars_df)))

    ax = plt.subplot(2, 1, 1)
    ax.tick_params(which='both', direction='in')
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    xbinned_mag, yplot_res, res_dispersion = binplot(dp.cat_mag, res, nbins=10, data=False, scale=False)
    plt.plot(dp.cat_mag, res, '.', color='black', zorder=-10.)
    plt.ylabel("$m_\mathrm{{ZTF}}-m_\mathrm{{{}}}-ZP-\\alpha C$ [mag]".format(op_args['photom_cat']))
    plt.ylim(-0.4, 0.4)
    plt.grid()

    ax = plt.subplot(2, 1, 2)
    ax.tick_params(which='both', direction='in')
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    plt.plot(xbinned_mag, res_dispersion)
    plt.grid()
    plt.xlabel("$m_\mathrm{{{}}}$ [mag]".format(op_args['photom_cat']))
    plt.ylabel("$\sigma_{{m_\mathrm{{ZTF}}-m_\mathrm{{{}}}-ZP-\\alpha C}}$ [mag]".format(op_args['photom_cat']))

    plt.tight_layout()
    plt.savefig(output_path.joinpath("binned_res.png"), dpi=200.)
    plt.close()

    plt.subplots(nrows=2, ncols=2, figsize=(10., 6.), gridspec_kw={'width_ratios': [5., 1.5], 'hspace': 0., 'wspace': 0.}, sharex=False, sharey=False)
    plt.suptitle("{}-{}\nStandardized residuals for the calibration, wrt star magnitude\npiedestal={}".format(lightcurve.name, lightcurve.filterid, op_args['piedestal']))
    plt.subplot(2, 2, 1)
    xbinned_mag, yplot_stdres, stdres_dispersion = binplot(dp.cat_mag, res/dp.delta_emag, data=False, scale=False, nbins=5)
    plt.plot(dp.cat_mag, wres, '.', color='xkcd:light blue')
    plt.ylabel("$\\frac{m-m_\\mathrm{model}}{\\sigma_m}$ [mag]")
    plt.xlim([np.min(dp.cat_mag), np.max(dp.cat_mag)])
    plt.grid()
    plt.subplot(2, 2, 2)
    plt.hist(wres, bins='auto', orientation='horizontal', density=True)
    m, s = norm.fit(wres)
    x = np.linspace(np.min(wres)-0.5, np.max(wres)+0.5, 200)
    plt.plot(norm.pdf(x, loc=m, scale=s), x, label="$\sim\mathcal{{N}}(\mu={:.2f}, \sigma={:.2f})$".format(m, s))
    plt.plot(norm.pdf(x, loc=0., scale=1.), x, label="$\sim\mathcal{N}(\mu=0, \sigma=1)$")
    plt.legend()
    plt.axis('off')
    plt.subplot(2, 2, 3)
    plt.plot(xbinned_mag, stdres_dispersion)
    plt.xlim([np.min(dp.cat_mag), np.max(dp.cat_mag)])
    plt.xlabel("$m_\mathrm{{{}}}$ [mag]".format(op_args['photom_cat']))
    plt.ylabel("$\\sigma_{\\frac{m-m_\\mathrm{model}}{\\sigma_m}}$ [mag]")
    plt.axhline(1.)
    plt.grid()
    plt.subplot(2, 2, 4)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path.joinpath("pull_mag.png"), dpi=200.)
    plt.close()

    plt.subplots(nrows=2, ncols=2, figsize=(10., 6.), gridspec_kw={'width_ratios': [5., 1.5], 'hspace': 0., 'wspace': 0.}, sharex=False, sharey=False)
    plt.suptitle("{}-{}\nStandardized residuals for the calibration, wrt star color\npiedestal={}".format(lightcurve.name, lightcurve.filterid, op_args['piedestal']))
    plt.subplot(2, 2, 1)
    xbinned_mag, yplot_stdres, stdres_dispersion = binplot(dp.cat_color, wres, data=False, scale=False, nbins=5)
    plt.plot(dp.cat_color, wres, '.', color='xkcd:light blue')
    plt.ylabel("$\\frac{m-m_\\mathrm{model}}{\\sigma_m}$ [mag]")
    plt.xlim([np.min(dp.cat_color), np.max(dp.cat_color)])
    plt.grid()
    plt.subplot(2, 2, 2)
    plt.hist(wres, bins='auto', orientation='horizontal', density=True)
    m, s = norm.fit(wres)
    x = np.linspace(np.min(wres)-0.5, np.max(wres)+0.5, 200)
    plt.plot(norm.pdf(x, loc=m, scale=s), x, label="$\sim\mathcal{{N}}(\mu={:.2f}, \sigma={:.2f})$".format(m, s))
    plt.plot(norm.pdf(x, loc=0., scale=1.), x, label="$\sim\mathcal{N}(\mu=0, \sigma=1)$")
    plt.legend()
    plt.axis('off')
    plt.subplot(2, 2, 3)
    plt.plot(xbinned_mag, stdres_dispersion)
    plt.xlabel("$C_\mathrm{{{}}}$ [mag]".format(op_args['photom_cat']))
    plt.ylabel("$\\sigma_{\\frac{m-m_\\mathrm{model}}{\\sigma_m}}$ [mag]")
    plt.xlim([np.min(dp.cat_color), np.max(dp.cat_color)])
    # plt.axhline(1.)
    plt.grid()
    plt.subplot(2, 2, 4)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path.joinpath("pull_color.png"), dpi=200.)
    plt.close()

    update_yaml(lightcurve.path.joinpath("lightcurve.yaml"), 'calib',
                {'color': solver.model.params['color'].full[0].item(),
                 'zp': solver.model.params['zp'].full[0].item(),
                 'cov': solver.get_cov().todense().tolist(),
                 'chi2': np.sum(wres**2).item(),
                 'ndof': len(dp.nt),
                 'chi2/ndof': np.sum(wres**2).item()/(len(dp.nt)-2),
                 'outlier_count': np.sum(solver.bads).item(),
                 'piedestal': op_args['piedestal'],
                 'bright_stars_res_std': bright_stars_std.item(),
                 'bright_stars_res_mu': bright_stars_mu.item(),
                 'bright_stars_threshold': bright_stars_threshold,
                 'bright_stars_count': len(dp.nt[bright_stars_mask])})

    return True

register_op('calib', reduce_op=calib, parameters=[{'name': 'piedestal', 'type': float, 'default': 0., 'desc': "Piedestal"},
                                                  {'name': 'photom_cat', 'type': str, 'default': 'ps1', 'desc': "External catalog to calibrate onto."}])
