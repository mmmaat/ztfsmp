#!/usr/bin/env python3

import pathlib
from shutil import copyfile
import pickle
import tarfile
import json
from shutil import rmtree
import os
from itertools import chain

from ztfquery.io import get_file
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
import pandas as pd
import numpy as np
from ztfimg.utils.tools import ccdid_qid_to_rcid

from ztfsmp.listtable import ListTable
from ztfsmp.ztf_utils import ztf_quadrant_name_explode, quadrant_width_px, quadrant_height_px, ztfquadrant_center
from ztfsmp.ext_cat_utils import gaia_edr3_refmjd


class _Exposure:
    def __init__(self, lightcurve, name, path=None):
        self.__lightcurve = lightcurve
        self.__name = name
        self.__year, self.__month, self.__day, self.__field, self.__filterid, self.__ccdid, self.__qid = ztf_quadrant_name_explode(name)
        if path is None:
            self.__path = self.__lightcurve.path.joinpath(name)
        else:
            self.__path = path

    @property
    def lightcurve(self):
        return self.__lightcurve

    @property
    def name(self):
        return self.__name

    @property
    def path(self):
        return self.__path

    @property
    def raw_name(self):
        return "_".join(self.__name.split("_")[:-1])

    @property
    def year(self):
        return self.__year

    @property
    def month(self):
        return self.__month

    @property
    def day(self):
        return self.__day

    @property
    def yyyymm(self):
        return str(self.year).zfill(4) + str(self.month).zfill(2)

    @property
    def field(self):
        return self.__field

    @property
    def filterid(self):
        return self.__filterid

    @property
    def ccdid(self):
        return self.__ccdid

    @property
    def rcid(self):
        return ccdid_qid_to_rcid(self.__ccdid, self.__qid)

    @property
    def qid(self):
        return self.__qid

    @property
    def path(self):
        return self.__path

    @property
    def mjd(self):
        raise NotImplementedError()

    @property
    def wcs(self):
        raise NotImplementedError()

    @property
    def retrieve_exposure(self):
        raise NotImplementedError()

    @property
    def update_exposure_header(self):
        pass

    def get_catalog(self, cat_name, key=None):
        pass


class CompressedExposure(_Exposure):
    def __init(self, lightcurve, name, path=None):
        pass


class Exposure(_Exposure):
    def __init__(self, lightcurve, name, path=None):
        super().__init__(lightcurve, name, path=path)

    @property
    def mjd(self):
        return float(self.exposure_header['obsmjd'])

    @property
    def wcs(self):
        return WCS(self.exposure_header)

    def retrieve_exposure(self, ztfin2p3_detrend=False, force_rewrite=True, **kwargs):
        if ztfin2p3_detrend:
            from ztfin2p3.science import build_science_image
            raw_path = str(pathlib.Path(get_file(self.raw_name, downloadit=False)))

            paths = build_science_image(
                raw_path,
                store=True,
                overwrite=True,
                corr_pocket=kwargs['corr_pocket'],
                outpath=self.path)
            image_path = pathlib.Path(paths[self.qid-1])
        elif 'ztfin2p3' in self.name:
            from ztfin2p3.io import ipacfilename_to_ztfin2p3filepath
            image_path = pathlib.Path(ipacfilename_to_ztfin2p3filepath("ztf" + self.name[8:] + "_sciimg.fits"))
        else:
            image_path = pathlib.Path(get_file(self.name + "_sciimg.fits", downloadit=False))

        if not image_path.exists():
            raise FileNotFoundError("Science image at {} not found on disk!".format(image_path))

        if force_rewrite or not self.path.joinpath("calibrated.fits").exists():
            copyfile(image_path, self.path.joinpath("calibrated.fits"))

        # Poloka needs elixir.fits
        if ztfin2p3_detrend:
            copyfile(image_path, self.path.joinpath("elixir.fits"))

        return image_path

    def update_exposure_header(self):
        if self.path.joinpath("calibrated.fits").exists():
            with fits.open(self.path.joinpath("calibrated.fits")) as hdul:
                hdul[0].header.tofile(self.path.joinpath("calibrated.header"), overwrite=True)

    @property
    def exposure_header(self):
        if self.path.joinpath("calibrated.fits").exists():
           with fits.open(self.path.joinpath("calibrated.fits")) as hdul:
               return hdul[0].header
        elif self.path.joinpath("calibrated.header").exists():
            with fits.open(self.path.joinpath("calibrated.header")) as hdul:
                return hdul[0].header
        else:
            raise FileNotFoundError("Could not find calibrated.fits or calibrated.header for exposure {}!".format(self.name))

    @property
    def elixir_header(self):
        with fits.open(self.path.joinpath("elixir.fits")) as hdul:
            return hdul[0].header

    def get_catalog(self, cat_name, key=None):
        if not isinstance(cat_name, pathlib.Path):
            cat_name = pathlib.Path(cat_name)

        if cat_name.suffix == ".list" or cat_name.suffix == ".dat":
            cat = ListTable.from_filename(self.path.joinpath(cat_name.name))
        elif cat_name.suffix == ".parquet":
            cat = pd.read_parquet(self.path.joinpath(cat_name.name))
        elif cat_name.suffix == ".hd5":
            cat = pd.read_hdf(self.path.joinpath(cat_name.name), key=key)
        else:
            raise ValueError("Catalog {} extension {} not recognized!".format(cat_name.name, cat_name.suffix))

        return cat

    def get_ext_catalog(self, cat_name, pm_correction=True, project=False):
        if cat_name not in self.lightcurve.ext_star_catalogs_name:
            raise ValueError("External catalogs are {}, not {}")

        ext_cat_df = self.lightcurve.get_ext_catalog(cat_name, matched=True)

        if pm_correction:
            obsmjd = self.mjd
            if cat_name == 'gaia' or cat_name == 'ubercal_self' or cat_name == 'ubercal_ps1':
                ext_cat_df.fillna(0., inplace=True)
                ext_cat_df = ext_cat_df.assign(RA_ICRS=ext_cat_df['RA_ICRS']+(obsmjd-gaia_edr3_refmjd)*ext_cat_df['pmRA'],
                                               DE_ICRS=ext_cat_df['DE_ICRS']+(obsmjd-gaia_edr3_refmjd)*ext_cat_df['pmDE'])
            elif cat_name == 'ps1':
                # Who uses PS1 astrometry for something other than matching it with Gaia?
                pass
            else:
                raise NotImplementedError()

        if project:
            if cat_name == 'gaia':
                skycoords = SkyCoord(ra=ext_cat_df['RA_ICRS'].to_numpy(), dec=ext_cat_df['DE_ICRS'].to_numpy(), unit='deg')
                stars_x, stars_y = skycoords.to_pixel(self.wcs)
                ext_cat_df['x'] = stars_x
                ext_cat_df['y'] = stars_y
            else:
                raise ValueError("get_ext_catalog(): proper motion can only be corrected on the Gaia catalog!")

        return ext_cat_df

    def get_matched_catalog(self, cat_name):
        if cat_name != 'aperstars' and cat_name != 'psfstars':
            raise ValueError("Only matched catalog are \'aperstars\' and \'psfstars\', not {}".format(cat_name))

        with pd.HDFStore(self.path.joinpath("cat_indices.hd5"), 'r') as hdfstore:
            cat_indices = hdfstore.get('{}_indices'.format(cat_name))
            ext_cat_indices = hdfstore.get('cat_indices')

        if cat_name == 'aperstars':
            cat_name = 'standalone_stars'

        cat_df = self.get_catalog("{}.list".format(cat_name)).df
        return cat_df.iloc[cat_indices].reset_index(drop=True).iloc[ext_cat_indices].reset_index(drop=True)

    def get_matched_ext_catalog(self, cat_name, pm_correction=True, project=False):
        if cat_name not in self.lightcurve.ext_star_catalogs_name:
            raise ValueError("External catalogs are {}, not {}".format(self.lightcurve.ext_star_catalogs_name, cat_name))

        ext_cat_df = self.get_ext_catalog(cat_name, pm_correction=pm_correction, project=project)
        with pd.HDFStore(self.path.joinpath("cat_indices.hd5"), 'r') as hdfstore:
            ext_cat_inside = hdfstore.get('ext_cat_inside')
            ext_cat_indices = hdfstore.get('ext_cat_indices')['indices']

        return ext_cat_df.iloc[ext_cat_inside.tolist()].iloc[ext_cat_indices].reset_index(drop=True)

    def func_status(self, func_name):
        return self.path.joinpath("{}.success".format(func_name)).exists()

    def func_timing(self, func_name):
        timings_path = self.path.joinpath("timings_{}".format(func_name))
        if timings_path.exists():
            with open(timings_path, 'r') as f:
                return json.load(f)

    def center(self):
        return self.wcs.pixel_to_world_values(np.array([[quadrant_width_px/2., quadrant_height_px/2.]]))[0]


class _Lightcurve:
    def __init__(self, name, filterid, wd, exposure_regexp="ztf*"):
        self.__name = name
        self.__filterid = filterid
        self.__path = wd.joinpath("{}/{}".format(name, filterid))
        self.__noprocess_path = self.__path.joinpath("noprocess")
        self.__driver_path = self.__path.joinpath("smphot_driver")
        self.__ext_catalogs_path = self.__path.joinpath("catalogs")
        self.__astrometry_path = self.__path.joinpath("astrometry")
        self.__photometry_path = self.__path.joinpath("photometry")
        self.__mappings_path = self.__path.joinpath("mappings")
        self.__smphot_path = self.__path.joinpath("smphot")
        self.__smphot_stars_path = self.__path.joinpath("smphot_stars")

    @property
    def path(self):
        return self.__path

    @property
    def name(self):
        return self.__name

    @property
    def filterid(self):
        return self.__filterid

    @property
    def ext_catalogs_path(self):
        return self.__ext_catalogs_path

    @property
    def astrometry_path(self):
        return self.__astrometry_path

    @property
    def photometry_path(self):
        return self.__photometry_path

    @property
    def mappings_path(self):
        return self.__mappings_path

    @property
    def driver_path(self):
        return self.__driver_path

    @property
    def smphot_path(self):
        return self.__smphot_path

    @property
    def smphot_stars_path(self):
        return self.__smphot_stars_path

    @property
    def noprocess_path(self):
        return self.__noprocess_path

    @property
    def exposures(self):
        raise NotImplementedError()

    @property
    def star_catalogs_name(self):
        return ['aperstars', 'psfstars']

    @property
    def ext_star_catalogs_name(self):
        return ['gaia', 'ps1', 'ubercal_self', 'ubercal_ps1']

    def get_exposures(self, files_to_check=None, ignore_noprocess=False):
        raise NotImplementedError()

    def add_noprocess(self, new_noprocess_quadrants):
        raise NotImplementedError()

    def reset_noprocess(self):
        raise NotImplementedError()

    def get_ext_catalog(self, cat_name, matched=True):
        raise NotImplementedError()

    def get_catalogs(self, cat_name, files_to_check=None):
        raise NotImplementedError()

    def exposure_headers(self):
        raise NotImplementedError()

    def get_reference_exposure(self):
        raise NotImplementedError()

    def extract_exposure_catalog(self):
        raise NotImplementedError()

    def extract_star_catalog(self, catalog_names, project=False):
        raise NotImplementedError()

    def func_status(self, func_name):
        raise NotImplementedError()

    def func_timing(self, func_name):
        raise NotImplementedError()


class Lightcurve(_Lightcurve):
    def __init__(self, name, filterid, wd, is_compressed=False, exposure_regexp="ztf*"):
        super().__init__(name, filterid, wd)

        if is_compressed:
            self.uncompress()

        self.__exposures = dict([(exposure_path.name, Exposure(self, exposure_path.name)) for exposure_path in list(self.__path.glob(exposure_regexp))])

    @property
    def exposures(self):
        return self.__exposures

    def get_exposures(self, files_to_check=None, ignore_noprocess=False):
        if files_to_check is None and ignore_noprocess:
            return self.__exposures.values()

        if files_to_check is None:
            files_to_check_list = []
        elif isinstance(files_to_check, str):
            files_to_check_list = [files_to_check]
        else:
            files_to_check_list = files_to_check

        def _check_files(exposure):
            check_ok = True
            for check_file in files_to_check_list:
                if not self.__path.joinpath("{}/{}".format(exposure.name, check_file)).exists():
                    check_ok = False
                    break

            return check_ok

        return list(filter(lambda x: (x.name not in self.get_noprocess() or ignore_noprocess) and _check_files(x), list(self.__exposures.values())))

    def add_noprocess(self, new_noprocess_quadrants):
        noprocess_quadrants = self.get_noprocess()
        noprocess_written = 0
        if isinstance(new_noprocess_quadrants, str) or isinstance(new_noprocess_quadrants, pathlib.Path):
            new_noprocess_quadrants_list = [new_noprocess_quadrants]
        else:
            new_noprocess_quadrants_list = new_noprocess_quadrants

        with open(self.noprocess_path, 'a') as f:
            for new_quadrant in new_noprocess_quadrants_list:
                if new_quadrant not in noprocess_quadrants:
                    f.write("{}\n".format(new_quadrant))
                    noprocess_written += 1

        return noprocess_written

    def get_noprocess(self):
        noprocess = []
        if self.noprocess_path.exists():
            with open(self.noprocess_path, 'r') as f:
                for line in f.readlines():
                    quadrant = line.strip()
                    if quadrant[0] == "#":
                        continue
                    elif self.path.joinpath(quadrant).exists():
                        noprocess.append(quadrant)

        return noprocess

    def reset_noprocess(self):
        if self.noprocess_path().exists():
            self.noprocess_path().unlink()

    def get_ext_catalog(self, cat_name, matched=True):
        if matched:
            catalog_df = pd.read_parquet(self.ext_catalogs_path.joinpath("{}.parquet".format(cat_name)))
        else:
            catalog_df = pd.read_parquet(self.ext_catalogs_path.joinpath("{}_full.parquet".format(cat_name)))

        return catalog_df

    def get_catalogs(self, cat_name, files_to_check=None):
        if files_to_check is None:
            files_to_check = cat_name
        elif isinstance(files_to_check, str) or isinstance(files_to_check, pathlib.Path):
            files_to_check = [cat_name, files_to_check]
        else:
            files_to_check.append(cat_name)

        exposures = self.get_exposures(files_to_check=files_to_check)
        return dict([(exposure.name, exposure.get_catalog(cat_name)) for exposure in exposures])

    def exposure_headers(self):
        exposures = list(filter(lambda x: True, self.get_exposures()))
        return dict([(exposure.name, exposure.exposure_header()) for exposure in exposures])

    def get_reference_exposure(self):
        if not self.__path.joinpath("reference_exposure").exists():
            raise FileNotFoundError("{}-{}: reference exposure has not been determined!".format(self.name, self.filterid))

        with open(self.__path.joinpath("reference_exposure"), 'r') as f:
            return f.readline().strip()

    def extract_exposure_catalog(self, files_to_check=None, ignore_noprocess=False):
        def _get_key(header, keys, t):
            if isinstance(keys, str):
                keys = [keys]

            for k in keys:
                if k.upper() in header.keys():
                    return t(header[k])

            return None

        exposures = []
        for exposure in self.get_exposures(files_to_check=files_to_check, ignore_noprocess=ignore_noprocess):
            header = exposure.exposure_header
            exposure_dict = {}
            exposure_dict['name'] = exposure.name
            exposure_dict['airmass'] = _get_key(header, 'airmass', float)
            exposure_dict['mjd'] = _get_key(header, 'obsmjd', float)
            exposure_dict['seeing'] = _get_key(header, 'seeing', float)
            exposure_dict['gfseeing'] = _get_key(header, 'gfseeing', float)
            exposure_dict['ha'] = _get_key(header, 'hourangd', float) #*15
            exposure_dict['ha_15'] = _get_key(header, 'hourangd', lambda x: 15.*float(x))
            exposure_dict['lst'] = _get_key(header, 'oblst', str)
            exposure_dict['azimuth'] = _get_key(header, 'azimuth', float)
            exposure_dict['dome_azimuth'] = _get_key(header, 'dome_az', float)
            exposure_dict['elevation'] = _get_key(header, 'elvation', float)
            exposure_dict['z'] = _get_key(header, 'elvation', lambda x: 90. - float(x))
            exposure_dict['telra'] = _get_key(header, 'telrad', float)
            exposure_dict['teldec'] = _get_key(header, 'teldecd', float)

            exposure_dict['field'] = _get_key(header, ['dbfield', 'fieldid'], int)
            exposure_dict['ccdid'] = _get_key(header, ['ccd_id', 'ccdid'], int)
            exposure_dict['qid'] = _get_key(header, ['amp_id', 'qid'], int)
            exposure_dict['rcid'] = _get_key(header, ['dbrcid', 'rcid'], int)
            exposure_dict['fid'] = _get_key(header, ['dbfid', 'filterid'], int)
            # if 'dbfield'.upper() in header.keys():
            #     exposure_dict['field'] = int(header['dbfield'])
            # else:
            #     exposure_dict['field'] = _get_key(header, 'fieldid', int)

            # if 'ccd_id'.upper() in header.keys():
            #     exposure_dict['ccdid'] = int(header['ccd_id'])
            # else:
            #     exposure_dict['ccdid'] = _get_key(header, 'ccdid', int)

            # if 'amp_id'.upper() in header.keys():
            #     exposure_dict['qid'] = int(header['amp_id'])
            # else:
            #     exposure_dict['qid'] = _get_key(header, 'qid', int)

            # if 'dbrcid' in header.keys():
            #     exposure_dict['rcid'] = int(header['dbrcid'])
            # else:
            #     exposure_dict['rcid'] = _get_key(header, 'rcid', int)

            # if 'fid' in header.keys():
            #     exposure_dict['fid'] = int(header['dbfid'])
            # else:
            #     exposure_dict['fid'] = _get_key(header, 'filterid', int)

            exposure_dict['filtercode'] = exposure.filterid

            exposure_dict['temperature'] = _get_key(header, 'tempture', float)
            exposure_dict['head_temperature'] = _get_key(header, 'headtemp', float)
            exposure_dict['ccdtemp'] = _get_key(header, 'ccdtmp{}'.format(str(header['ccd_id']).zfill(2)), float)
            exposure_dict['exptime'] = _get_key(header, 'exptime', float)
            exposure_dict['expid'] = _get_key(header, 'dbexpid', int)

            exposure_dict['wind_speed'] = _get_key(header, 'windspd', float)
            exposure_dict['wind_dir'] = _get_key(header, 'winddir', float)
            exposure_dict['dewpoint'] = _get_key(header, 'dewpoint', float)
            exposure_dict['humidity'] = _get_key(header, 'humidity', float)
            exposure_dict['wetness'] = _get_key(header, 'wetness', float)
            exposure_dict['pressure'] = _get_key(header, 'pressure', float)

            exposure_dict['crpix1'] = _get_key(header, 'crpix1', float)
            exposure_dict['crpix2'] = _get_key(header, 'crpix2', float)
            exposure_dict['crval1'] = _get_key(header, 'crval1', float)
            exposure_dict['crval2'] = _get_key(header, 'crval2', float)
            exposure_dict['cd_11'] = _get_key(header, 'cd1_1', float)
            exposure_dict['cd_12'] = _get_key(header, 'cd1_2', float)
            exposure_dict['cd_21'] = _get_key(header, 'cd2_1', float)
            exposure_dict['cd_22'] = _get_key(header, 'cd2_2', float)

            exposure_dict['gain'] = _get_key(header, 'gain', float)
            exposure_dict['readnoise'] = _get_key(header, 'readnoi', float)
            exposure_dict['darkcurrent'] = _get_key(header, 'darkcur', float)

            exposure_dict['ztfmagzp'] = _get_key(header, 'magzp', float)
            exposure_dict['ztfmagzpunc'] = _get_key(header, 'magzpunc', float)
            exposure_dict['ztfmagzprms'] = _get_key(header, 'magzprms', float)

            exposure_dict['skylev'] = _get_key(header, 'sexsky', float)
            exposure_dict['sigma_skylev'] = _get_key(header, 'sexsigma', float)

            exposures.append(exposure_dict)

        return pd.DataFrame(exposures).set_index('name')

    def extract_star_catalog(self, catalog_names, project=False, pm_correction=True):
        for catalog_name in catalog_names:
            if catalog_name not in (self.star_catalogs_name+self.ext_star_catalogs_name):
                raise ValueError("Star catalog name \'{}\' does not exists!".format(catalog_name))

        catalog_list = []
        exposures = self.get_exposures(files_to_check=["cat_indices.hd5"])
        name_set = False

        for catalog_name in catalog_names:
            catalogs = []

            for exposure in exposures:
                if catalog_name in self.star_catalogs_name:
                    catalog_df = exposure.get_matched_catalog(catalog_name)
                else:
                    catalog_df = exposure.get_matched_ext_catalog(catalog_name, project=project, pm_correction=pm_correction)

                if not name_set:
                    catalog_df.insert(0, 'exposure', exposure.name)

                catalogs.append(catalog_df)

            catalogs_df = pd.concat(catalogs)

            if len(catalog_names) > 1:
                if not name_set:
                    catalogs_df.columns = ['exposure'] + ['{}_{}'.format(catalog_name, column_name) for column_name in catalogs_df.columns[1:]]
                else:
                    catalogs_df.columns = ['{}_{}'.format(catalog_name, column_name) for column_name in catalogs_df.columns]

            if not name_set:
                name_set = True

            catalog_list.append(catalogs_df)

        return pd.concat(catalog_list, axis='columns').reset_index().rename(columns={'index': 'cat_index'})

    def func_status(self, func_name):
        if self.path.joinpath("{}.success".format(func_name)).exists():
            return 1.
        elif self.path.joinpath("{}.fail".format(func_name)).exists():
            return 0.
        else:
            exposures = self.get_exposures()
            if len(exposures) == 0:
                return 0.

            success = [exposure.func_status(func_name) for exposure in exposures]
            return sum(success)/len(exposures)

    def func_timing(self, func_name):
        map_timing = {'start': float('inf'), 'end': -float('inf'), 'elapsed': 0.}
        reduce_timing = {'start': float('inf'), 'end': -float('inf'), 'elapsed': 0.}

        timing_path = self.path.joinpath("timings_{}".format(func_name))
        if timing_path.exists():
            with open(timing_path, 'r') as f:
                reduce_timing = json.load(f)

        map_timings = [exposure.func_timing(func_name) for exposure in self.get_exposures()]
        map_timings = list(filter(lambda x: x is not None, map_timings))

        if len(map_timings) > 0:
            map_timings_df = pd.DataFrame(map_timings)
            map_timing = {'start': map_timings_df['start'].min(),
                        'end': map_timings_df['end'].max()}
            map_timing['elapsed'] = map_timing['end'] - map_timing['start']

        total_timing = {'start': min([map_timing['start'], reduce_timing['start']]),
                        'end': max([map_timing['end'], map_timing['start']])}
        total_timing['elapsed'] = total_timing['end'] - total_timing['start']

        return {'map': map_timing,
                'reduce': reduce_timing,
                'total': total_timing}

    def compress_states(self):
        exposures = self.get_exposures(ignore_noprocess=True)

        files = list(chain(*[list(exposure.path.glob("*.success")) for exposure in exposures]))
        files.extend(list(chain(*[list(exposure.path.glob("*.fail")) for exposure in exposures])))
        files.extend(list(chain(*[list(exposure.path.glob("timings_*")) for exposure in exposures])))
        files.extend(list(self.path.glob("*.success")))
        files.extend(list(self.path.glob("*.fail")))
        files.extend(list(self.path.glob("timings_*")))

        if self.path.joinpath("timings_total") in files:
            files.remove(self.path.joinpath("timings_total"))

        self.path.joinpath("states.tar").unlink(missing_ok=True)
        tar = tarfile.open(self.path.joinpath("states.tar"), 'w')

        for f in files:
            tar.add(f, f.relative_to(self.path))
        tar.close()

        # Remove files
        #[f.unlink() for f in files]

    def uncompress_states(self, keep_compressed_files=False):
        if not self.path.joinpath("states.tar").exists():
            return

        tar = tarfile.open(self.path.joinpath("states.tar"), 'r')
        tar.extractall(path=self.path)
        tar.close()

        if not keep_compressed_files:
            self.path.joinpath("states.tar").unlink()

    def compress(self):
        """
        Compress lightcurve into a big hdf and tar files (more efficient storage, limiting small files)
        If funcs is provided, also fill success rates and timings for each pipeline step.
        funcs is expected to be a list of strings, each being a pipeline step.
        """

        exposures = self.get_exposures(ignore_noprocess=True)

        tar_path = self.path.joinpath("exposures.tar")
        tar_path.unlink(missing_ok=True)

        # Store catalogs into one big HDF file
        catalogs_to_store = ['standalone_stars', 'psfstars']
        with pd.HDFStore(self.path.joinpath("catalogs.hdf"), 'w') as hdfstore:
            # Store exposure catalogs
            for exposure in exposures:
                for catalog in catalogs_to_store:
                    try:
                        cat = exposure.get_catalog(catalog+".list")
                    except FileNotFoundError:
                        continue

                    for key in cat.header.keys():
                        hdfstore.put('{}/{}/header/{}'.format(exposure.name, catalog, key), pd.Series(data=cat.header[key]))

                    hdfstore.put('{}/{}/df'.format(exposure.name, catalog), cat.df)

                # Store indices
                if exposure.func_status("match_catalogs"):
                    with pd.HDFStore(exposure.path.joinpath("cat_indices.hd5")) as hdfstore_indices:
                        for cat_name in ['aperstars_indices', 'cat_indices', 'ext_cat_indices', 'psfstars_indices']:
                            hdfstore.put('{}/cat_indices/{}'.format(exposure.name, cat_name), hdfstore_indices.get(cat_name))

            # Store noprocess exposures
            noprocess_df = pd.Series(self.get_noprocess())
            hdfstore.put('noprocess', noprocess_df)

        # Store headers
        headers = dict([(exposure.name, exposure.exposure_header) for exposure in exposures])
        with open(self.path.joinpath("headers.pickle"), 'wb') as f:
            pickle.dump(headers, f)

        # First start with timings and func states
        self.compress_states()

        # self.path.joinpath("lightcurve_sn.parquet").unlink(missing_ok=True)
        # self.path.joinpath("smphot_stars_cat.parquet").unlink(missing_ok=True)

        def _tar_folder(path):
            if path.exists():
                tar_path = self.path.joinpath("{}.tar".format(path.name))
                tar_path.unlink(missing_ok=True)
                files = list(path.glob("*"))
                tar = tarfile.open(tar_path, 'w')
                for f in files:
                    tar.add(f, f.relative_to(self.path))

                rmtree(path)

                return True

            return False

        files = list(self.path.glob("*"))

        # Store SMP lightcurves if available
        if _tar_folder(self.smphot_path):
            files.remove(self.smphot_path)

        if _tar_folder(self.smphot_stars_path):
            files.remove(self.smphot_stars_path)

        # Compress all the other files
        files.remove(self.path.joinpath("catalogs.hdf"))
        files.remove(self.path.joinpath("headers.pickle"))
        files.remove(self.path.joinpath("states.tar"))

        tar = tarfile.open(tar_path, 'w')
        for f in files:
            tar.add(f, f.relative_to(self.path))
        tar.close()

        # Delete all files
        for f in files:
            if f.is_dir():
                rmtree(f)
            else:
                f.unlink()

    def uncompress(self, keep_compressed_files=False):
        self.uncompress_states(keep_compressed_files=keep_compressed_files)

        if not self.path.joinpath("exposures.tar").exists():
            return

        def _uncompress_here(path):
            if path.exists():
                tar = tarfile.open(self.path.joinpath(path), 'r')
                try:
                    tar.extractall(path=self.path)
                except tarfile.ReadError as e:
                    print(e)
                finally:
                    tar.close()

        _uncompress_here(self.path.joinpath("exposures.tar"))
        _uncompress_here(self.path.joinpath("smphot.tar"))
        _uncompress_here(self.path.joinpath("smphot_stars.tar"))

        if not keep_compressed_files:
            self.path.joinpath("catalogs.hdf").unlink(missing_ok=True)
            self.path.joinpath("headers.pickle").unlink(missing_ok=True)
            self.path.joinpath("exposures.tar").unlink(missing_ok=True)
            self.path.joinpath("smphot.tar").unlink(missing_ok=True)
            self.path.joinpath("smphot_stars.tar").unlink(missing_ok=True)


class CompressedLightcurve(_Lightcurve):
    def __init__(self, name, filterid, wd):
        super().__init(name, filterid, wd)
        pass
