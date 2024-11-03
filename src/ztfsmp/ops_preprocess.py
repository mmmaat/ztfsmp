#!/usr/bin/env python3

from ztfsmp.pipeline_utils import run_and_log
from ztfsmp.pipeline import register_op


def make_catalog(exposure, logger, args, op_args):
    logger.info("Retrieving science exposure...")
    try:
        image_path = exposure.retrieve_exposure(ztfin2p3_detrend=op_args['ztfin2p3_detrend'])
    except FileNotFoundError as e:
        print(e)
        logger.error(e)
        return False

    logger.info("Found at {}".format(image_path))

    run_and_log(["make_catalog", exposure.path, "-O", "-S"], logger)

    logger.info("Dumping header content")
    exposure.update_exposure_header()

    return exposure.path.joinpath("se.list").exists()


make_catalog_rm = ["low.fits.gz", "miniback.fits", "segmentation.cv.fits", "segmentation.fits"]
make_catalog_parameters = [{'name': 'ztfin2p3_detrend', 'type': bool, 'default': False, 'desc': ""}]

register_op('make_catalog', map_op=make_catalog, rm_list=make_catalog_rm, parameters=make_catalog_parameters)


def mkcat2(exposure, logger, args, op_args):
    from itertools import chain
    import numpy as np
    from scipy.sparse import dok_matrix
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    import matplotlib.pyplot as plt

    from ztfsmp.misc_utils import contained_in_exposure, sc_array, match_pixel_space

    run_and_log(["mkcat2", exposure.path, "-o"], logger)

    if not exposure.path.joinpath("standalone_stars.list").exists():
        return False

    if op_args['use_gaia_stars']:
        # Find standalone stars using Gaia
        logger.info("Using Gaia catalog to identify stars")
        aperse_cat = exposure.get_catalog("aperse.list")
        standalone_stars_cat = exposure.get_catalog("standalone_stars.list")
        gaia_stars_df = exposure.lightcurve.get_ext_catalog('gaia', matched=False)[['Gmag', 'RA_ICRS', 'DE_ICRS', 'pmRA', 'pmDE']].dropna()

        obsmjd = exposure.mjd
        # gaia_stars_df = gaia_stars_df.assign(ra=gaia_stars_df['ra']+(obsmjd-j2000mjd)*gaia_stars_df['pmRA'],
        #                                      dec=gaia_stars_df['dec']+(obsmjd-j2000mjd)*gaia_stars_df['pmDE'])

        logger.info("Total Gaia stars={}".format(len(gaia_stars_df)))
        # Remove Gaia stars outside of the exposure
        wcs = exposure.wcs
        ra_center, dec_center = exposure.center()
        dist = np.sqrt((ra_center-gaia_stars_df['RA_ICRS'].to_numpy())**2+(dec_center-gaia_stars_df['DE_ICRS'].to_numpy())**2)
        m = (dist <= 0.7)
        gaia_stars_df = gaia_stars_df[m]
        gaia_stars_skycoords = SkyCoord(ra=gaia_stars_df['RA_ICRS'].to_numpy(), dec=gaia_stars_df['DE_ICRS'].to_numpy(), unit='deg')
        gaia_stars_inside = wcs.footprint_contains(gaia_stars_skycoords).tolist()

        if sum(gaia_stars_inside) == 0:
            logger.error("No Gaia stars found inside the exposure!")
            return False

        gaia_stars_skycoords = gaia_stars_skycoords[gaia_stars_inside]
        gaia_stars_df = gaia_stars_df.iloc[gaia_stars_inside]
        logger.info("Total Gaia stars in the quadrant footprint={}".format(len(gaia_stars_df)))

        # Project contained Gaia stars into pixel space
        gaia_stars_x, gaia_stars_y = gaia_stars_skycoords.to_pixel(wcs)
        gaia_stars_df = gaia_stars_df.assign(x=gaia_stars_x, y=gaia_stars_y)

        if op_args['remove_flagged']:
            aperse_cat.df = aperse_cat.df.loc[aperse_cat.df['flag']==0]
            aperse_cat.df = aperse_cat.df.loc[aperse_cat.df['gflag']==0]

        # Removing measures that are too close to each other
        # Min distance should be a function of seeing idealy
        #
        n = len(aperse_cat.df)
        X = np.tile(aperse_cat.df['x'].to_numpy(), (n, 1))
        Y = np.tile(aperse_cat.df['y'].to_numpy(), (n, 1))
        dist = np.sqrt((X-X.T)**2+(Y-Y.T)**2)
        dist_mask = (dist <= op_args['isolated_star_distance'])
        sp = dok_matrix(dist_mask)
        keys = list(filter(lambda x: x[0]!=x[1], list(sp.keys())))
        too_close_idx = list(set(list(chain(*keys))))
        keep_idx = list(filter(lambda x: x not in too_close_idx, range(n)))

        logger.info("aperse catalog: {} measures".format(n))
        logger.info("Out of which, {} are too close to each other (min distance={})".format(len(too_close_idx), op_args['isolated_star_distance']))
        logger.info("{} measures are kept".format(len(keep_idx)))

        aperse_cat.df = aperse_cat.df.iloc[keep_idx]

        i = match_pixel_space(gaia_stars_df[['x', 'y']].to_records(), aperse_cat.df[['x', 'y']].to_records(), radius=1.)
        gaia_indices = i[i>=0]
        cat_indices = np.arange(len(aperse_cat.df))[i>=0]

        standalone_stars_df = aperse_cat.df.iloc[cat_indices]
        logger.info("Old star count={}".format(len(standalone_stars_cat.df)))
        logger.info("New star count={}".format(len(standalone_stars_df)))

        old_cat = exposure.get_catalog("standalone_stars.list")
        standalone_stars_cat.df = standalone_stars_df
        standalone_stars_cat.write()

        if op_args['plot_star_moment_plane']:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Ellipse
            import numpy as np

            aperse_cat = exposure.get_catalog("aperse.list")
            standalone_stars_cat = exposure.get_catalog("standalone_stars.list")
            x, y, _, sigma_x, sigma_y, corr = aperse_cat.header['starshape']

            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8., 8.))
            plt.suptitle("$N_s={}$, seeing={}\n{}".format(len(standalone_stars_cat.df), standalone_stars_cat.header['seeing'], exposure.name))
            ax.add_patch(Ellipse((x, y), width=5.*sigma_x, height=5.*sigma_y, fill=False, color='red'))
            plt.plot(np.sqrt(aperse_cat.df['gmxx'].to_numpy()), np.sqrt(aperse_cat.df['gmyy'].to_numpy()), '.', label="SE cat")
            plt.plot(np.sqrt(standalone_stars_cat.df['gmxx'].to_numpy()), np.sqrt(standalone_stars_cat.df['gmyy'].to_numpy()), '.', color='red', label="Stand. cat")
            plt.plot(np.sqrt(old_cat.df['gmxx'].to_numpy()), np.sqrt(old_cat.df['gmyy'].to_numpy()), 'x', color='green', label="Old stand. cat")
            plt.xlabel("$\\sqrt{M_g^{xx}}$")
            plt.ylabel("$\\sqrt{M_g^{yy}}$")
            plt.legend()
            plt.plot([x], [y], 'x')
            plt.xlim(0., 4.)
            plt.ylim(0., 4.)
            plt.grid()
            plt.savefig(exposure.path.joinpath("smp.png"))
            plt.close()

    return exposure.path.joinpath("standalone_stars.list").exists()

mkcat2_rm = []
mkcat2_parameters = [{'name': 'use_gaia_stars', 'type': bool, 'default': True, 'desc': "Use Gaia catalog to identify stars."},
                     {'name': 'isolated_star_distance', 'type': float, 'default': 20., 'desc': "Minimum distance between star to identify them as isolated, in arcsec."},
                     {'name': 'plot_star_moment_plane', 'type': bool, 'default': False, 'desc': "Plot isolated stars Gaussian centered second moment plane."},
                     {'name': 'remove_flagged', 'type': bool, 'default': False, 'desc': "Remove stars if flagged bad by sextractor or mkcat2."}]

register_op('mkcat2', map_op=mkcat2, rm_list=mkcat2_rm, parameters=mkcat2_parameters)


def makepsf(exposure, logger, args, op_args):
    run_and_log(["makepsf", exposure.path, "-f"], logger)

    logger.info("Dumping header content")
    exposure.update_exposure_header()

    return exposure.path.joinpath("psfstars.list").exists()

makepsf_rm = ["psf_resid_tuple.fit", "psf_res_stack.fits", "psf_resid_image.fits", "psf_resid_tuple.dat"]
makepsf_parameters = []

register_op('makepsf', map_op=makepsf, rm_list=makepsf_rm, parameters=makepsf_parameters)


def preprocess(exposure, logger, args, op_args):
    def _run_step(f, step_name):
        logger.info("Running {}".format(step_name))
        if not f(exposure, logger, args, op_args):
            exposure.path.joinpath("{}.fail".format(step_name)).touch()
            return False
        else:
            exposure.path.joinpath("{}.success".format(step_name)).touch()
            return True

    if not _run_step(make_catalog, "make_catalog"):
        return False

    if not _run_step(mkcat2, "mkcat2"):
        return False

    return _run_step(makepsf, "makepsf")

preprocess_rm = make_catalog_rm + mkcat2_rm + makepsf_rm
preprocess_parameters = make_catalog_parameters + mkcat2_parameters + makepsf_parameters

register_op('preprocess', map_op=preprocess, rm_list=preprocess_rm, parameters=preprocess_parameters)
