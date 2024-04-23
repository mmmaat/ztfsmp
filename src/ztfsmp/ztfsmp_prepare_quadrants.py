#!/usr/bin/env python3

import argparse
import logging
import pathlib
import datetime

import numpy as np
import ztfquery
import joblib
from ztfimg.science import ScienceQuadrant
from astropy.io import fits

from utils import filtercodes


def _create_symlink(path, symlink_to):
    if path.exists():
        path.unlink()

    path.symlink_to(symlink_to)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description="Prepare directory structure for quadrants.")
    argparser.add_argument('--quadrant-list', type=pathlib.Path, required=True)
    argparser.add_argument('--output', type=pathlib.Path, required=True)
    argparser.add_argument('-j', '--n_jobs', type=int, default=1)
    argparser.add_argument('--use-raw', action='store_true')
    argparser.add_argument('--ztfin2p3-path', type=pathlib.Path)
    argparser.add_argument('--filtercode', choices=filtercodes+['all'], default='all')

    args = argparser.parse_args()
    args.output = args.output.expanduser().resolve()
    args.quadrant_list = args.quadrant_list.expanduser().resolve()

    if not args.quadrant_list.exists():
        print("{} does not exist!".format(args.quadrant_list))

    args.output.mkdir(exist_ok=True)
    prepare_logger = logging.getLogger("prepare")
    prepare_logger.addHandler(logging.FileHandler(args.output.joinpath("prepare.log")))
    prepare_logger.addHandler(logging.StreamHandler())
    prepare_logger.setLevel(logging.INFO)
    prepare_logger.info(datetime.datetime.today())
    prepare_logger.info("Preparing deppol folder in {}".format(args.output))

    prepare_logger.info("Quadrant list={}".format(args.quadrant_list))
    prepare_logger.info("Looking for list in all filtercodes")

    if args.filtercode == 'all':
        filtercode_choice = filtercodes
    else:
        filtercode_choice = [args.filtercode]

    quadrant_lists={}
    for filtercode in filtercode_choice:
        quadrant_list = args.quadrant_list.with_name("{}_{}.txt".format(args.quadrant_list.name, filtercode))
        prepare_logger.info("Looking for {}".format(quadrant_list))
        if quadrant_list.exists():

            with open(quadrant_list, 'r') as f:
                image_filenames = list(map(lambda x: x.strip(), f.readlines()))

            prepare_logger.info("Found. {} images.".format(len(image_filenames)))
            quadrant_lists[filtercode] = image_filenames
        else:
            prepare_logger.info("Not found.")

    for filtercode in quadrant_lists.keys():
        def _create_folder(filtercode, quadrant_list):
            prepare_logger.info("In filter {} - creating directory {}".format(filtercode, args.output.joinpath(filtercode)))
            args.output.joinpath(filtercode).mkdir(exist_ok=True)

            def _create_subfolder(image_filename):
                try:
                    # Check files exist
                    quadrant_logger = logging.getLogger(image_filename)
                    quadrant_logger.addHandler(logging.FileHandler(args.output.joinpath("{}/prepare.log".format(filtercode))))
                    quadrant_logger.addHandler(logging.StreamHandler())
                    quadrant_logger.setLevel(logging.INFO)

                    if not args.use_raw:
                        if args.ztfin2p3_path:
                            sciimg_path = args.ztfin2p3_path.joinpath("sci/{}/{}/{}/{}".format(image_filename[9:13], image_filename[13:17], image_filename[17:23], image_filename))
                            mskimg_path = None
                        else:
                            sciimg_path = pathlib.Path(ztfquery.io.get_file(image_filename, downloadit=False, suffix='sciimg.fits'))
                            mskimg_path = pathlib.Path(ztfquery.io.get_file(image_filename, downloadit=False, suffix='mskimg.fits'))
                        #mskimg_path_gz = pathlib.Path(ztfquery.io.get_file(image_filename, downloadit=False, suffix='mskimg.fits.gz'))

                        # Much faster than relying on ScienceQuadrant.from_filename() throwing error
                        # if not sciimg_path.exists() or not (mskimg_path.exists() or mskimg_path_gz.exists()):
                        #     raise FileNotFoundError(sciimg_path)
                        if not sciimg_path.exists():
                            raise FileNotFoundError(sciimg_path)

                        if args.ztfin2p3_path:
                            folder_name = "_".join(image_filename.split("_")[:-1])
                        else:
                            folder_name = image_filename[:37]

                        folder_path = args.output.joinpath("{}/{}".format(filtercode, folder_name))

                        folder_path.mkdir(exist_ok=True)
                        folder_path.joinpath(".dbstuff").touch()


                        _create_symlink(folder_path.joinpath("elixir.fits"), sciimg_path)

                        if mskimg_path and mskimg_path.exists():
                            z = ScienceQuadrant.from_filename(sciimg_path)
                            deads = np.array(z.get_mask(tracks=False, ghosts=False, spillage=False, spikes=False, dead=True,
                                                        nan=False, saturated=False, brightstarhalo=False, lowresponsivity=False,
                                                        highresponsivity=False, noisy=False, verbose=False), dtype=np.uint8)

                            mskhdu = fits.PrimaryHDU([deads])
                            mskhdu.writeto(folder_path.joinpath("dead.fits.gz"), overwrite=True)
                    else:
                        raw_path = pathlib.Path(ztfquery.io.get_file(image_filename))

                        if not raw_path.exists():
                            raise FileNotFoundError(raw_path)

                        folder_name = image_filename[:37]
                        folder_path = args.output.joinpath("{}/{}".format(filtercode, folder_name))

                        folder_path.mkdir(exist_ok=True)
                        folder_path.joinpath(".dbstuff").touch()

                        _create_symlink(folder_path.joinpath("elixir.fits"), raw_path)


                except FileNotFoundError as e:
                    quadrant_logger.error("Fail: {}".format(sciimg_path))
                except Exception as e:
                    quadrant_logger.exception("Exception error for: {}".format(sciimg_path))
                else:
                    if args.use_raw:
                        quadrant_logger.info("Success: {}".format(image_filename))
                    else:
                        if mskimg_path and mskimg_path.exists():
                            quadrant_logger.info("Success: {}, dead pixel count={}".format(image_filename, np.sum(deads)))
                        else:
                            quadrant_logger.info("Success: {}, no dead image found.".format(image_filename))
                finally:
                    for handler in quadrant_logger.handlers:
                        handler.close()

            joblib.Parallel(n_jobs=args.n_jobs)(joblib.delayed(_create_subfolder)(image_filename) for image_filename in quadrant_list)

        _create_folder(filtercode, quadrant_lists[filtercode])
