#!/usr/bin/env python3

import sys
import pathlib
import shutil
import datetime
import argparse
import traceback

import logging
import pandas as pd
from ztfimg.science import ScienceQuadrant
import ztfquery.io
import joblib
import numpy as np
from astropy.io import fits
from deppol_utils import ztfnames_from_string, lc_folder_args
from utils import filtercodes


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description="Prepare directory structure for deppol.")
    argparser.add_argument('--ztfname', type=pathlib.Path, required=True)
    argparser.add_argument('--output', type=pathlib.Path, required=True)
    argparser.add_argument('-j', '--n_jobs', type=int, default=1)
    argparser.add_argument('--lc-folder', type=pathlib.Path)
    argparser.add_argument('--no-deads', action='store_true')

    args = argparser.parse_args()
    args.output = args.output.expanduser().resolve()
    args.lc_folder = lc_folder_args(args)

    prepare_logger = logging.getLogger("prepare")
    prepare_logger.addHandler(logging.FileHandler(args.output.joinpath("prepare.log")))
    prepare_logger.addHandler(logging.StreamHandler())
    prepare_logger.setLevel(logging.INFO)
    prepare_logger.info(datetime.datetime.today())
    prepare_logger.info("Preparing deppol folder in {}".format(args.output))

    prepare_logger.info("Loading ztfnames...")
    ztfnames = ztfnames_from_string(args.ztfname)
    if not ztfnames:
        prepare_logger.error("Found no SN 1a in ztfname file!")
        exit()

    prepare_logger.info("Found {} SNe 1a".format(len(ztfnames)))

    prepare_logger.info("Checking for their LC file counterpart")
    lc_files = [lc_file.stem for lc_file in list(args.lc_folder.glob("*.hd5"))]
    new_ztfnames = []
    for ztfname in ztfnames:
        if ztfname not in lc_files:
            prepare_logger.warning("Could not find LC file for {}".format(ztfname))
        else:
            prepare_logger.info("Found LC for {}".format(ztfname))
            new_ztfnames.append(ztfname)

    ztfnames = new_ztfnames

    prepare_logger.info("Preparing folder for {} SNe".format(len(ztfnames)))

    for ztfname in ztfnames:
        prepare_logger.info("In SN {}".format(ztfname))
        args.output.joinpath("{}".format(ztfname)).mkdir(exist_ok=True)
        logger = logging.getLogger(ztfname)
        logger.addHandler(logging.FileHandler(args.output.joinpath("{}/prepare_{}.log".format(ztfname, ztfname))))
        logger.addHandler(logging.StreamHandler())
        logger.setLevel(logging.INFO)
        logger.info(datetime.datetime.today())
        logger.info("Supernova: {}".format(ztfname))
        logger.info("In folder {}".format(args.output.joinpath(ztfname)))

        def _create_subfolders(filtercode, hdfstore):
            logger.info("In filter {} - creating directory".format(filtercode))
            args.output.joinpath("{}/{}".format(ztfname, filtercode)).mkdir(exist_ok=True)
            lc_df = pd.read_hdf(hdfstore, key='lc_{}'.format(filtercode))

            def _create_subfolder(sciimg_filename):
                quadrant_logger = logging.getLogger(sciimg_filename)
                quadrant_logger.addHandler(logging.FileHandler(args.output.joinpath("{}/prepare_{}.log".format(ztfname, ztfname))))
                quadrant_logger.addHandler(logging.StreamHandler())
                quadrant_logger.setLevel(logging.INFO)


                try:
                    # Check files exist
                    sciimg_path = pathlib.Path(ztfquery.io.get_file(sciimg_filename, downloadit=False, suffix='sciimg.fits'))
                    mskimg_path = pathlib.Path(ztfquery.io.get_file(sciimg_filename, downloadit=False, suffix='mskimg.fits'))
                    mskimg_path_gz = pathlib.Path(ztfquery.io.get_file(sciimg_filename, downloadit=False, suffix='mskimg.fits.gz'))

                    path = args.output.joinpath("{}/{}/{}".format(ztfname, filtercode, sciimg_filename[:37]))
                    if path.exists():
                        if path.joinpath("elixir.fits").exists() and path.joinpath(".dbstuff").exists():
                            quadrant_logger.info("Success (already here): {}".format(sciimg_path))
                            return

                    # Much faster than relying on ScienceQuadrant.from_filename() throwing error
                    if not sciimg_path.exists() or not (mskimg_path.exists() or mskimg_path_gz.exists()):
                        raise FileNotFoundError(sciimg_path)


                    # First create filter path
                    args.output.joinpath("{}/{}".format(ztfname, filtercode)).mkdir(exist_ok=True)
                    folder_name = sciimg_filename[:37]
                    folder_path = args.output.joinpath("{}/{}/{}".format(ztfname, filtercode, folder_name))

                    folder_path.mkdir(exist_ok=True)
                    folder_path.joinpath(".dbstuff").touch()

                    def _create_symlink(path, symlink_to):
                        if path.exists():
                            path.unlink()

                        path.symlink_to(symlink_to)

                    _create_symlink(folder_path.joinpath("elixir.fits"), sciimg_path)

                    # Dead mask
                    if not args.no_deads:
                        z = ScienceQuadrant.from_filename(sciimg_path, use_dask=False)

                        deads = np.array(z.get_mask(tracks=False, ghosts=False, spillage=False, spikes=False, dead=True,
                                                    nan=False, saturated=False, brightstarhalo=False, lowresponsivity=False,
                                                    highresponsivity=False, noisy=False, verbose=False), dtype=np.uint8)

                        mskhdu = fits.PrimaryHDU([deads])
                        mskhdu.writeto(folder_path.joinpath("dead.fits.gz"), overwrite=True)

                except FileNotFoundError as e:
                    quadrant_logger.error("Fail: {}".format(sciimg_path))
                except Exception as e:
                    quadrant_logger.exception("Exception error for: {}".format(sciimg_path))
                else:
                    if args.no_deads:
                        quadrant_logger.info("Success: {}, dead pixel count=-1".format(sciimg_filename))
                    else:
                        quadrant_logger.info("Success: {}, dead pixel count={}".format(sciimg_filename, np.sum(deads)))
                finally:
                    for handler in quadrant_logger.handlers:
                        handler.close()

            joblib.Parallel(n_jobs=args.n_jobs)(joblib.delayed(_create_subfolder)(sciimg_filename) for sciimg_filename in lc_df['ipac_file'])


        if args.lc_folder.joinpath("{}.hd5".format(ztfname)).exists():
            with pd.HDFStore(args.lc_folder.joinpath("{}.hd5".format(ztfname)), mode='r') as hdfstore:
                for filtercode in filtercodes:
                    if '/lc_{}'.format(filtercode) in hdfstore.keys():
                        _create_subfolders(filtercode, hdfstore)

        for handler in logger.handlers:
            handler.close()
