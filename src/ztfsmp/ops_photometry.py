#!/usr/bin/env python3

import time

import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix, dia_array
from sksparse import cholmod
from saunerie.fitparameters import FitParameters

from ztfsmp.pipeline import register_op

from ztfsmp.listtable import ListTable


def _dump_photoratios(model, dp, y_model, bads, reference_exposure, save_folder_path):
    zp_ref = model.params['zp'].full[dp.exposure_map[reference_exposure]]

    alphas = {}
    for exposure in dp.exposure_set:
        alphas[exposure] = 10**(-0.4*(model.params['zp'].full[dp.exposure_map[exposure]] - zp_ref))

    # TODO: compute error on alpha
    alphas_df = pd.DataFrame(data={'expccd': list(alphas.keys()), 'alpha': list(alphas.values())})
    alphas_df['ealpha'] = 0.

    ndof = len(y_model) - len(model.params.free) - sum(bads)
    chi2 = np.sum(((y_model-dp.mag)**2/dp.emag**2)[~bads])

    photom_ratios_table = ListTable({'CHI2': chi2, 'NDOF': ndof, 'RCHI2': chi2/ndof, 'REF': reference_exposure}, alphas_df)
    photom_ratios_table.write_to(save_folder_path.joinpath("photom_ratios.ntuple"))


def photometry_fit(lightcurve, logger, args, op_args):
    import pandas as pd
    import pickle
    import matplotlib
    import matplotlib.pyplot as plt
    from croaks import DataProxy
    from saunerie.linearmodels import LinearModel, RobustLinearSolver, indic

    from ztfsmp.ext_cat_utils import mag2extcatmag, mag2extcatmag
    from ztfsmp.misc_utils import make_index_from_list
    from ztfsmp.pipeline_utils import update_yaml

    matplotlib.use('Agg')

    photometry_stats = {}

    save_folder_path = lightcurve.photometry_path
    save_folder_path.mkdir(exist_ok=True)
    lightcurve.mappings_path.mkdir(exist_ok=True)

    logger.info("Building DataProxy")

    ext_cat_df = lightcurve.extract_star_catalog(['ps1'])
    if op_args['use_aper']:
        photometry_stats['cat'] = 'aper'
        matched_stars_df = pd.concat([lightcurve.extract_star_catalog(['aperstars']),
                                      ext_cat_df[['objID', mag2extcatmag['ps1'][lightcurve.filterid], 'e_{}'.format(mag2extcatmag['ps1'][lightcurve.filterid])]]], axis='columns')
        matched_stars_df = matched_stars_df.drop(columns=['cat_index', 'flux', 'eflux']).rename(columns={'objID': 'catid',
                                                                                                        mag2extcatmag['ps1'][lightcurve.filterid]: 'cat_mag',
                                                                                                        'e_{}'.format(mag2extcatmag['ps1'][lightcurve.filterid]): 'cat_emag'})
        matched_stars_df.rename(columns={'apfl4': 'flux', 'eapfl4': 'eflux'}, inplace=True)

    else:
        photometry_stats['cat'] = 'psf'
        matched_stars_df = pd.concat([lightcurve.extract_star_catalog(['psfstars']),
                                      ext_cat_df[['objID', mag2extcatmag['ps1'][lightcurve.filterid], 'e_{}'.format(mag2extcatmag['ps1'][lightcurve.filterid])]]], axis='columns')
        matched_stars_df = matched_stars_df.drop(columns='cat_index').rename(columns={'objID': 'catid',
                                                                                      mag2extcatmag['ps1'][lightcurve.filterid]: 'cat_mag',
                                                                                      'e_{}'.format(mag2extcatmag['ps1'][lightcurve.filterid]): 'cat_emag'})

    matched_stars_df['mag'] = -2.5*np.log10(matched_stars_df['flux'])
    matched_stars_df['emag'] = 1.08*matched_stars_df['eflux']/matched_stars_df['flux']
    matched_stars_df['colormag'] = ext_cat_df['imag'] - ext_cat_df['gmag']
    matched_stars_df.dropna(subset=['mag', 'emag', 'colormag', 'cat_mag', 'cat_emag'], inplace=True)

    matched_stars_df = matched_stars_df[['exposure', 'mag', 'emag', 'catid', 'cat_mag', 'cat_emag', 'colormag']]

    matched_stars_df.dropna(inplace=True) # You never know...

    photometry_stats['initial_measurements'] = len(matched_stars_df)

    catids = list(set(matched_stars_df['catid']))
    min_measurements = max([2, int(0.05*len(list(set(matched_stars_df['exposure']))))])
    to_remove = [catid for catid in catids if len(matched_stars_df.loc[matched_stars_df['catid']==catid])<min_measurements]
    star_count = len(list(set(matched_stars_df['catid'])))
    matched_stars_df = matched_stars_df.set_index('catid').drop(index=to_remove).reset_index()
    logger.info("Removing {} stars (out of {}) with less than {} measurements.".format(len(to_remove), star_count, min_measurements))
    logger.info("Total measurements={}".format(len(matched_stars_df)))

    photometry_stats['removed_stars'] = len(to_remove)
    photometry_stats['final_measurements'] = len(matched_stars_df)

    exposures_df = lightcurve.extract_exposure_catalog(files_to_check="match_catalogs.success")
    for column in ['mjd', 'airmass', 'rcid']:
        matched_stars_df[column] = exposures_df.loc[matched_stars_df['exposure'], column].to_numpy()

    #################
    # ZP + stars fit

    kwargs = dict([(keyword, keyword) for keyword in matched_stars_df.columns])
    dp_index_list = ['exposure', 'catid', 'rcid', 'mjd', 'cat_mag', 'colormag']
    dp = DataProxy(matched_stars_df.to_records(), **kwargs)
    make_index_from_list(dp, dp_index_list)

    refid = dp.exposure_map[lightcurve.get_reference_exposure()]
    piedestal = 0.

    logger.info("Piedestal={}".format(piedestal))
    photometry_stats['piedestal'] = piedestal

    def _build_model(dp):
        model = indic(dp.catid_index, name='star') + indic(dp.exposure_index, name='zp')
        model.params['zp'].fix(refid, 0.)
        return RobustLinearSolver(model, dp.mag, weights=1./np.sqrt(dp.emag**2+piedestal**2))

    def _solve_model(solver):
        solver.model.params.free = solver.robust_solution(local_param='star')

    def _filter_noisy_stars(solver, dp, threshold):
        logger.info("Removing noisy stars...")

        wres = (solver.get_res(dp.mag)/np.sqrt(dp.emag**2+piedestal**2))[~solver.bads]
        chi2 = np.bincount(dp.catid_index[~solver.bads], weights=wres**2)/np.bincount(dp.catid_index[~solver.bads])

        logger.info("Chi2 treshold={}".format(threshold))
        noisy_stars = dp.catid_set[chi2 > threshold]
        noisy_measurements = np.any([dp.catid == noisy_star for noisy_star in noisy_stars], axis=0)

        plt.subplots(figsize=(10., 5))
        ax = plt.gca()
        plt.suptitle("Star $\\chi^2$")
        plt.plot(range(len(dp.catid_map)), chi2, '.')
        plt.axhline(1., ls='--', color='black')
        plt.axhline(threshold, color='black')
        plt.text(0.2, 0.8, "$\\chi^2_\\mathrm{{threshold}}={{{}}}$".format(threshold), transform=ax.transAxes, fontsize='large')
        plt.text(0.2, 0.75, "{} stars with $\\chi^2>\\chi^2_\\mathrm{{threshold}}$ (out of {})".format(len(noisy_stars), len(dp.catid_map)), transform=ax.transAxes, fontsize='large')
        plt.xlabel("Star ID")
        plt.ylabel("$\\chi^2$")
        plt.grid()
        plt.savefig(lightcurve.photometry_path.joinpath("stars_chi2_filtering.png"), dpi=250.)
        plt.close()

        dp.compress(~noisy_measurements)
        logger.info("Filtered {} stars...".format(len(noisy_stars)))
        logger.info("New star count={}".format(len(list(set(dp.catid)))))
        logger.info("New measurement count={}".format(len(dp.nt)))

        photometry_stats['filtered_stars'] = len(noisy_stars)

        return _build_model(dp)


    solver = _build_model(dp)
    _solve_model(solver)

    res = solver.get_res(dp.mag)
    wres = res/dp.emag

    photometry_stats['unfiltered'] = {}
    photometry_stats['unfiltered']['chi2'] = np.sum(wres[~solver.bads]**2).item()
    photometry_stats['unfiltered']['ndof'] = len(dp.nt[~solver.bads])
    photometry_stats['unfiltered']['chi2/ndof'] = np.sum(wres[~solver.bads]**2).item()/len(dp.nt[~solver.bads])

    y_model = solver.model()
    with open(save_folder_path.joinpath("unfiltered_model.pickle"), 'wb') as f:
        pickle.dump([solver.model, dp, solver.bads, solver.get_cov(), solver.get_res(dp.mag), y_model, piedestal], f)

    new_solver = _filter_noisy_stars(solver, dp, op_args['max_star_chi2'])

    _solve_model(new_solver)

    res = new_solver.get_res(dp.mag)
    wres = res/dp.emag

    photometry_stats['filtered'] = {}
    photometry_stats['filtered']['chi2'] = np.sum(wres[~new_solver.bads]**2).item()
    photometry_stats['filtered']['ndof'] = len(dp.nt[~new_solver.bads])
    photometry_stats['filtered']['chi2/ndof'] = np.sum(wres[~new_solver.bads]**2).item()/len(dp.nt[~new_solver.bads])
    y_new_model = new_solver.model()
    with open(save_folder_path.joinpath("filtered_model.pickle"), 'wb') as f:
        pickle.dump([new_solver.model, dp, new_solver.bads, new_solver.get_cov(), new_solver.get_res(dp.mag), y_new_model, piedestal], f)

    logger.info("Computing photometric ratios")
    _dump_photoratios(new_solver.model, dp, y_new_model, new_solver.bads, lightcurve.get_reference_exposure(), lightcurve.mappings_path)

    update_yaml(lightcurve.path.joinpath("lightcurve.yaml"), 'photometry', photometry_stats)

    return True

register_op('photometry_fit', reduce_op=photometry_fit, parameters=[{'name': 'max_star_chi2', 'type': float, 'default': 3., 'desc': "Maximum chi2/star"},
                                                                    {'name': 'use_aper', 'type': bool, 'default': False, 'desc': "Use aperture catalogs as input"}])


def photometry_fit_plot(lightcurve, logger, args, op_args):
    import pickle
    import matplotlib
    import matplotlib.pyplot as plt
    from scipy.stats import norm
    from saunerie.plottools import binplot
    from ztfquery.fields import ccdid_qid_to_rcid

    from ztfsmp.ext_cat_utils import mag2extcatmag, extcat2colorstr
    from ztfsmp.ztf_utils import ztf_quadrant_name_explode
    from ztfsmp.misc_utils import idx2markerstyle

    matplotlib.use('Agg')

    save_folder_path = lightcurve.photometry_path

    def _do_plots(model_name):
        save_folder_path = lightcurve.photometry_path.joinpath(model_name.split("_")[0])
        save_folder_path.mkdir(exist_ok=True)

        with open(lightcurve.photometry_path.joinpath("{}.pickle".format(model_name)), 'rb') as f:
           model, dp, bads, cov, res, y_model, piedestal = pickle.load(f)

        def _show(filename, save_folder=True, plot_ext='.png'):
            plt.tight_layout()
            if save_folder:
                plt.savefig(save_folder_path.joinpath("{}{}".format(filename, plot_ext)), dpi=250.)
            else:
                plt.show()

            plt.close()

        dp.compress(~bads)
        res = res[~bads]
        wres = res/np.sqrt(dp.emag**2+piedestal**2)
        sqrtcovdiag = np.sqrt(cov.diagonal())

        stars_df = pd.DataFrame({'catid': list(dp.catid_map.keys())})
        stars_df['mag'] = model.params['star'].full
        stars_df['emag'] = sqrtcovdiag[model.params['star'].indexof()]
        stars_df['chi2'] = np.bincount(dp.catid_index, weights=wres**2)/(np.bincount(dp.catid_index)-1)
        stars_df['count'] = np.bincount(dp.catid_index)
        stars_df['sigma_mag'] = [np.std(res[dp.catid==catid]) for catid in dp.catid_set]

        stars_df.set_index('catid', drop=True, inplace=True)


        ps1_df = lightcurve.get_ext_catalog('ps1', matched=False)
        ps1_df = ps1_df.drop_duplicates('objID').set_index('objID', drop=True)

        for column in ['gmag', 'imag']:
            stars_df[column] = [ps1_df.loc[catid][column] for catid in stars_df.index.tolist()]

        stars_df['cat_mag'] = ps1_df[mag2extcatmag['ps1'][lightcurve.filterid]]

        stars_df['color'] = stars_df['imag'] - stars_df['gmag']

        stars_df.to_parquet(lightcurve.photometry_path.joinpath("psf_constant_stars.parquet"))

        zp_df = pd.DataFrame({'exposure': list(dp.exposure_map.keys())})
        zp_df['expid'] = list(dp.exposure_map.values())
        zp_df['zp'] = model.params['zp'].full
        zp_df['ezp'] = sqrtcovdiag[model.params['zp'].indexof()]
        zp_df['chi2'] = np.bincount(dp.exposure_index, weights=wres**2)/(np.bincount(dp.exposure_index)-1)
        zp_df['count'] = np.bincount(dp.exposure_index)
        zp_df['qid'] = list(map(lambda x: x[6], [ztf_quadrant_name_explode(exposure_name) for exposure_name in dp.exposure_map.keys()]))
        zp_df['ccdid'] = list(map(lambda x: x[5], [ztf_quadrant_name_explode(exposure_name) for exposure_name in dp.exposure_map.keys()]))
        zp_df['rcid'] = ccdid_qid_to_rcid(zp_df['ccdid'].to_numpy(), zp_df['qid'].to_numpy())

        exposures_df = lightcurve.extract_exposure_catalog()
        for column in ['mjd', 'airmass', 'rcid', 'skylev', 'seeing']:
            zp_df[column] = exposures_df.loc[zp_df['exposure'], column].to_numpy()

        zp_df.set_index('exposure', drop=True, inplace=True)

        rcid_df = pd.DataFrame({'rcid': list(dp.rcid_map.keys())})
        rcid_df['chi2'] = np.bincount(dp.rcid_index, weights=wres**2)/(np.bincount(dp.rcid_index)-1)
        rcid_df['count'] = np.bincount(dp.rcid_index)
        rcid_df.set_index('rcid', drop=True, inplace=True)

        # Measurements per stars
        plt.subplots(figsize=(7., 5.))
        plt.suptitle("Measurement count per stars")
        plt.plot(stars_df['cat_mag'].to_numpy(), stars_df['count'].to_numpy(), '.', color='black')
        for i, rcid_row in enumerate(rcid_df.iterrows()):
            rcid = rcid_row[0]
            plt.axhline(len(zp_df.loc[zp_df['rcid']==rcid]), label="{} - {}".format(rcid, len(zp_df.loc[zp_df['rcid']==rcid])), lw=3., color='C{}'.format(i))
        plt.legend(title="RCID - Count")
        plt.xlabel("$m$ [AB mag]")
        plt.ylabel("Measurement count")
        plt.grid()
        _show("stars_measurement_count")

        # ZP measurement count as a function of seeing
        plt.subplots(figsize=(7., 5.))
        plt.suptitle("ZP measurement count as a function of seeing")
        plt.plot(zp_df['count'].to_numpy(), zp_df['seeing'].to_numpy(), '.')
        plt.xlabel("Count")
        plt.ylabel("Seeing FWHM [px]")
        _show("zp_measurement_count_seeing")

        # Measurements per ZP
        plt.subplots(figsize=(7., 5.))
        plt.suptitle("Measurement count per ZP")
        plt.plot(zp_df['zp'].to_numpy(), zp_df['count'].to_numpy(), '.')
        plt.xlabel("$ZP$ [mag]")
        plt.ylabel("Measurement count")
        _show("zp_measurement_count")

        # Measurements per rcid
        plt.subplots(figsize=(7., 5.))
        plt.suptitle("Measurement count per rcid")
        for rcid in rcid_df.index:
            plt.bar([rcid], rcid_df.loc[rcid]['count'], label=rcid)
        plt.xlim(-0.5, 64.5)
        plt.legend(title="rcid")
        plt.xlabel("rcid")
        plt.ylabel("Measurement count")
        _show("rcid_measurement_count")

        # Binplot residuals / mag
        plt.subplots(nrows=2, ncols=1, figsize=(10., 6.), gridspec_kw={'hspace': 0.})
        plt.suptitle("Residuals as a function of magnitude")
        plt.subplot(2, 1, 1)
        xbinned_mag, yplot_res, res_dispersion = binplot(dp.cat_mag, res, data=True, rms=True, scale=False)
        plt.ylim(-0.1, 0.1)
        plt.ylabel("$m-m_\\mathrm{model}$ [mag]")
        plt.grid()

        plt.subplot(2, 1, 2)
        plt.plot(xbinned_mag, res_dispersion)
        plt.xlabel("$m_g$ [AB mag]")
        plt.ylabel("$\\sigma_{m-m_\\mathrm{model}}$ [mag]")
        plt.axhline(0.01)
        plt.ylim(0., 0.1)
        plt.grid()
        _show("res_cat_mag_binplot")

        # Pulls
        plt.subplots(nrows=2, ncols=1, figsize=(10., 6.), gridspec_kw={'hspace': 0.})
        plt.suptitle("Standardized residuals as a function of magnitude")
        plt.subplot(2, 1, 1)
        xbinned_mag, yplot_stdres, stdres_dispersion = binplot(dp.cat_mag, wres, data=True, rms=True, scale=False)
        plt.ylabel("$\\frac{m-m_\\mathrm{model}}{\\sigma_m}$ [mag]")
        plt.grid()

        plt.subplot(2, 1, 2)
        plt.plot(xbinned_mag, stdres_dispersion)
        plt.xlabel("$m_g$ [AB mag]")
        plt.ylabel("$\\sigma_{\\frac{m-m_\\mathrm{model}}{\\sigma_m}}$ [mag]")
        plt.axhline(1.)
        plt.grid()
        _show("stdres_cat_mag_binplot")

        # Star chi2 / mag
        plt.subplots(figsize=(7., 5.))
        plt.suptitle("Stars $\\chi^2$ / mag")
        plt.plot(stars_df['cat_mag'].to_numpy(), stars_df['chi2'].to_numpy(), '.', color='black')
        plt.grid()
        plt.xlabel("$m$")
        plt.ylabel("$\\chi^2$")
        _show("stars_mag_chi2")

        # Star chi2
        plt.subplots(figsize=(7., 5.))
        plt.suptitle("Stars $\\chi^2$")
        plt.plot(range(len(stars_df)), stars_df['chi2'].to_numpy(), '.')
        plt.grid()
        plt.xlabel("Star ID")
        plt.ylabel("$\\chi^2$")
        _show("stars_chi2")

        # Star magnitudes histogram
        plt.subplots(figsize=(7., 5.))
        plt.suptitle("Star magnitude $m$ distribution")
        plt.hist(stars_df['mag'].to_numpy(), bins='auto', histtype='step', color='black')
        plt.xlabel("$m$ [mag]")
        plt.ylabel("Count")
        plt.grid()
        _show("stars_mag_hist")

        # Star magnitudes vs PS1 magnitudes
        plt.subplots(figsize=(5., 5.))
        plt.suptitle("Star magnitude compared to PS1 ($g$ band)")
        plt.scatter(stars_df['mag'].to_numpy(), stars_df['cat_mag'].to_numpy(), s=0.5, c=stars_df.color.to_numpy())
        plt.colorbar()
        plt.xlabel("$m$ [mag]")
        plt.ylabel("$m_g$ [mag]")
        _show("stars_mag_ps1")

        plt.subplots(figsize=(7., 5.))
        plt.suptitle("Star magnitude residuals to PS1 compared to color")
        plt.plot(stars_df['color'].to_numpy(), (stars_df['mag'].to_numpy()-stars_df['cat_mag'].to_numpy())-(stars_df['mag'].to_numpy()-stars_df['cat_mag'].to_numpy()).mean(), '.')
        plt.xlabel("Color - $m_i-m_g$ [mag]")
        plt.ylabel("$m-m_\\mathrm{PS1}-\\left<m-m_\\mathrm{PS1}\\right>$ [mag]")
        plt.grid()
        _show("star_mag_res_ps1_color")

        plt.subplots(figsize=(7., 5.))
        plt.suptitle("Star magnitude residuals to PS1 compared to PS1 mag")
        # plt.scatter(stars_df['cat_mag'].to_numpy(), (stars_df['mag'].to_numpy()-stars_df['cat_mag'].to_numpy())-(stars_df['mag'].to_numpy()-stars_df['cat_mag'].to_numpy()).mean(), s=5, c=stars_df['color'])
        plt.scatter(stars_df['cat_mag'].to_numpy(), (stars_df['mag'].to_numpy()-stars_df['cat_mag'].to_numpy()), s=5, c=stars_df['color'])
        plt.xlabel("$m_\mathrm{PS1}$ [AB mag]")
        plt.ylabel("$m-m_\\mathrm{PS1}-\\left<m-m_\\mathrm{PS1}\\right>$ [mag]")
        plt.colorbar()
        plt.grid()
        _show("star_mag_res_ps1_mag")

        # Star magnitude vs repetability
        plt.subplots(nrows=2, ncols=1, figsize=(10., 6.))
        plt.suptitle("Star AB magnitude $m$ compared to its repetability")

        plt.subplot(2, 1, 1)
        plt.plot(stars_df['cat_mag'].to_numpy(), stars_df['sigma_mag'].to_numpy(), '.')
        plt.xlabel("$m$ [mag]")
        plt.ylabel("$\\sigma_m$ [mag]")
        plt.grid()

        plt.subplot(2, 1, 2)
        plt.plot(stars_df['cat_mag'].to_numpy(), stars_df['sigma_mag'].to_numpy(), '.')
        plt.xlabel("$m$ [mag]")
        plt.ylabel("$\\sigma_m$ [mag]")
        plt.ylim(0., 0.05)
        plt.grid()
        _show("star_repetability")

        plt.subplots(nrows=2, ncols=1, figsize=(10., 6.))
        plt.suptitle("Star AB magnitude $m$ compared to its error on the magnitude")

        plt.subplot(2, 1, 1)
        plt.scatter(stars_df['cat_mag'].to_numpy(), stars_df['emag'].to_numpy(), s=5., c=stars_df['count'].to_numpy().astype(float))
        plt.xlabel("$m$ [mag]")
        plt.ylabel("$\\sigma_\hat{m}$ [mag]")
        plt.grid()

        plt.subplot(2, 1, 2)
        plt.scatter(stars_df['cat_mag'].to_numpy(), stars_df['emag'].to_numpy(), s=5., c=stars_df['count'].to_numpy().astype(float))
        plt.xlabel("$m$ [mag]")
        plt.ylabel("$\\sigma_\hat{m}$ [mag]")
        plt.ylim(0., 0.02)
        plt.grid()
        _show("star_mag_error")

        # ZP distribution
        plt.hist(zp_df['zp'].to_numpy(), bins=40, histtype='step', color='black')
        plt.grid()
        plt.xlabel("ZP")
        plt.ylabel("Count")
        _show("zp_dist")

        # ZP chi2
        plt.subplots(figsize=(7., 5.))
        plt.suptitle("$\\chi^2$ per ZP/quadrant")
        plt.plot(zp_df['mjd'].to_numpy(), zp_df['chi2'].to_numpy(), '.')
        plt.grid()
        plt.xlabel("MJD")
        plt.ylabel("$\\chi^2$")
        _show("zp_chi2")

        # eZP / expid, color=qid
        plt.subplots(figsize=(7., 5.))
        plt.suptitle("ZP compared to mjd")
        plt.scatter(zp_df['expid'].to_numpy(), zp_df['ezp'].to_numpy(), s=5., c=zp_df['qid'])
        plt.grid()
        plt.colorbar(cmap='jet')
        plt.xlabel("MJD")
        plt.ylabel("ZP [mag]")
        _show("ezp_expid")

        # ZP / mjd
        plt.subplots(figsize=(7., 5.))
        plt.suptitle("ZP compared to mjd")
        plt.plot(zp_df['mjd'].to_numpy(), zp_df['ezp'].to_numpy(), '.')
        plt.grid()
        plt.xlabel("MJD")
        plt.ylabel("ZP [mag]")
        _show("zp_mjd")

        # Residuals distribution
        plt.hist(res, bins=100, histtype='step', color='black')
        plt.grid()
        plt.xlabel("Residuals")
        plt.ylabel("Count")
        _show("res_dist")

        # Residuals/day
        plt.plot(dp.mjd, res, ',', color='red')
        plt.grid()
        plt.xlabel("MJD")
        plt.ylabel("Residual")
        _show("res_day")

        plt.plot(dp.mjd_index, res, ',', color='red')
        plt.grid()
        plt.xlabel("MJD")
        plt.ylabel("Residual")
        _show("res_day_index")

        #Residuals/star
        plt.plot(dp.catid_index, res, ',', color='red')
        plt.grid()
        plt.xlabel("Star")
        plt.ylabel("Residual")
        _show("res_star")

        # Residuals/airmass
        plt.plot(dp.airmass, res, ',', color='red')
        plt.xlabel('Airmass')
        plt.ylabel('Residual')
        plt.grid()
        _show("res_airmass")

        # Residuals/color
        plt.subplots(figsize=(7., 4.))
        plt.plot(dp.colormag, res, ',', color='black')
        plt.xlabel("${}$ [mag]".format(extcat2colorstr['ps1']))
        plt.ylabel("Residual")
        plt.grid()
        _show("res_color")


    _do_plots("unfiltered_model")
    _do_plots("filtered_model")

register_op('photometry_fit_plot', reduce_op=photometry_fit_plot)
