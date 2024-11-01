#!/usr/bin/env python3

import os
import argparse
import pathlib
import logging
import datetime
import time
import shutil
import sys
import socket
import copy
import traceback
import yaml
import importlib

from dask import delayed, compute, visualize
from dask.distributed import Client, LocalCluster, wait, get_worker
from dask_jobqueue import SLURMCluster
from dask.graph_manipulation import bind, checkpoint
import pandas as pd
from dask import config as cfg

from ztfsmp.pipeline import pipeline
from ztfsmp.lightcurve import Exposure, Lightcurve
from ztfsmp.run_arguments import RunArguments
from ztfsmp.pipeline_utils import run_and_log, dump_timings, dump_timings_reduce, load_timings, quadrants_from_band_path, noprocess_quadrants


cfg.set({'distributed.scheduler.worker-ttl': None})

ztf_filtercodes = ['zg', 'zr', 'zi', 'all']

import ztfsmp.ztfsmp_pipeline as ztfsmp_pipeline
search_paths = [pathlib.Path(ztfsmp_pipeline.__file__).parent]
for search_path in search_paths:
    for ops_module in list(search_path.glob("ops_*.py")):
        globals()[ops_module] = importlib.import_module("ztfsmp." + str(ops_module.stem))

def dump_hw_infos(scratch, output_path):
    logger = logging.getLogger("dump")
    logger.addHandler(logging.FileHandler(output_path))
    logger.setLevel(logging.INFO)

    run_and_log(["lscpu"], logger)

    run_and_log(["df", "-h"], logger)

    if args.scratch:
        if args.scratch.exists():
            scratch_usage = shutil.disk_usage(args.scratch)
            logger.info("Scratch disk usage")
            logger.info("  Total: {:.2f} GB".format(scratch_usage.total/1e9))
            logger.info("  Used : {:.2f} GB".format(scratch_usage.used/1e9))
            logger.info("  Free : {:.2f} GB".format(scratch_usage.free/1e9))

        if 'TMPDIR' in os.environ:
            run_and_log(["ls", "-lah", os.environ['TMPDIR']], logger)


def map_op(exposure_name, wd, name, filtercode, func, args, op_args):
    try:
        exposure = Exposure(Lightcurve(name, filtercode, wd), exposure_name)

        start_time = time.perf_counter()
        exposure_path = wd.joinpath("{}/{}/{}".format(name, filtercode, exposure_name))
        lightcurve_path = wd.joinpath("{}/{}".format(name, filtercode))

        logger = None
        if func != 'clean':
            if exposure_name in exposure.lightcurve.get_noprocess():
                return

            logger = logging.getLogger(exposure_name)
            logger.setLevel(logging.INFO)

            if args.exposure_workspace:
                # If exposure working directory is specified (such as using /dev/shm), rebuild the exposure object accordingly
                exposure_workspace = args.exposure_workspace.joinpath(exposure_name)
                shutil.copytree(exposure_path, exposure_workspace, symlinks=False, dirs_exist_ok=True)
                exposure_path = exposure_workspace
                exposure = Exposure(Lightcurve(name, filtercode, wd), exposure_name, path=exposure_path)

            logger.addHandler(logging.FileHandler(str(exposure_path.joinpath("output.log")), mode='a'))
            logger.info("")
            logger.info("="*80)
            logger.info("{}-{}".format(name, filtercode))
            logger.info(datetime.datetime.today())
            logger.info("Running map operation \"{}\" on exposure {}.".format(func, exposure_name))
            logger.info("Exposure directory: {}".format(exposure_path))
            logger.info("Working directory: {}".format(wd))

        result = False
        try:
            start_time = time.perf_counter()
            result = pipeline.ops[func]['map_op'](exposure, logger, args, op_args)
        except Exception as e:
            if func != 'clean':
                logger.error(traceback.format_exc())

            print("{}-{}: {}".format(name, filtercode, func))
            print("Exposure: {}".format(exposure_name))
            print(e)
            print(traceback.format_exc())
            result = False
        finally:
            end_time = time.perf_counter()
            if func != 'clean':
                logger.info("End of func \"{}\".".format(func))

                if result:
                    exposure_path.joinpath("{}.success".format(func)).touch()
                else:
                    exposure_path.joinpath("{}.fail".format(func)).touch()

                # Remove intermediate files if any
                if args.rm_intermediates and (pipeline.ops[func]['rm_list'] is not None):
                    logger.info("Removing intermediate files...")
                    for f in pipeline.ops[func]['rm_list']:
                        to_remove = exposure_path.joinpath(f)
                        if to_remove.exists():
                            to_remove.unlink()
                        else:
                            logger.warning("Tried to remove {} but it does not exist!".format(f))
                            logger.warning(" Full path: {}".format(to_remove))

                if args.dump_timings:
                    dump_timings(start_time, end_time, exposure_path.joinpath("timings_{}".format(func)))

                if args.exposure_workspace:
                    logger.info("Copying exposure data from temporary working directory back into original.")
                    exposure_path.joinpath("elixir.fits").unlink()
                    shutil.copytree(exposure_path, wd.joinpath("{}/{}/{}".format(name, filtercode, exposure_name)), dirs_exist_ok=True)
                    logger.info("Erasing exposure data from temporary working directory.")
                    shutil.rmtree(exposure_path)

                [handler.close() for handler in logger.handlers] # Needed to flush last msg

    except Exception as e:
        print("Exception raised for exposure {} in {}-{}!".format(exposure_name, name, filtercode))
        traceback.print_exc()
        print("{} content:".format(exposure_path))
        print(list(exposure_path.glob("*")))
        return False
    else:
        return result


def reduce_op(wd, name, filtercode, func, save_stats, args, op_args):
    lightcurve_path = wd.joinpath("{}/{}".format(name, filtercode))
    lightcurve = Lightcurve(name, filtercode, wd)
    # If we want to agregate run statistics on the previous map operation
    # if save_stats and results is not None and any(results) and func != 'clean':
    #     results_df = pd.DataFrame([result for result in results if result is not None], columns=['result', 'time_end', 'time_start', 'worker_id'])
    #     results_df.to_csv(folder.joinpath("results_{}.csv".format(func)), index=False)

    # if func != 'clean':
    #     pass

    start_time = time.perf_counter()
    if func != 'clean' and (pipeline.ops[func]['reduce_op'] is not None):
        logger = logging.getLogger("{}-{}".format(name, filtercode))
        if not logger.handlers:
            logger.addHandler(logging.FileHandler(lightcurve_path.joinpath("output.log"), mode='a'))
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter("%(asctime)s;%(levelname)s;%(message)s"))
            logger.addHandler(ch)
            logger.setLevel(logging.INFO)

        logger.info("")
        logger.info("="*80)
        logger.info("{}-{}".format(name, filtercode))
        logger.info(datetime.datetime.today())
        logger.info("Running reduction {}".format(func))
    else:
        logger = None

    if pipeline.ops[func]['reduce_op'] is not None:
        try:
            result = pipeline.ops[func]['reduce_op'](lightcurve, logger, args, op_args)
        except Exception as e:
            result = False
            if func != 'clean':
                logger.error("{}-{}: {}".format(name, filtercode, func))
                logger.error(traceback.format_exc())
            else:
                print("{}-{}: {}".format(name, filtercode, func))
                traceback.print_exc()
        finally:
            end_time = time.perf_counter()
            if func != 'clean':
                logger.info("="*80)

                if result:
                    lightcurve_path.joinpath("{}.success".format(func)).touch()
                else:
                    lightcurve_path.joinpath("{}.fail".format(func)).touch()
            else:
                return 0
    else:
        end_time = time.perf_counter()

    if args.dump_timings and func != 'clean':
        map_start_time = None
        map_end_time = None
        if pipeline.ops[func]['map_op'] is not None:
            exposures = lightcurve.get_exposures(files_to_check="{}.success".format(func))
            timings = [load_timings(exposure.path.joinpath("timings_{}".format(func))) for exposure in exposures if exposure.path.joinpath("{}.success".format(func)).exists()]
            timings_df = pd.DataFrame.from_records(timings)
            timings_df['exposure'] = list(map(lambda x: x.name, exposures))
            timings_df.set_index('exposure', inplace=True)

            map_start_time = timings_df['start'].min()
            map_end_time = timings_df['end'].max()

        dump_timings_reduce({'map': map_start_time, 'reduce': start_time},
                            {'map': map_end_time, 'reduce': end_time},
                            lightcurve_path.joinpath("timings_{}".format(func)))

    return 0


def main():
    print(len(pipeline.ops))
    argparser = RunArguments(description="")
    argparser.add_argument('--run-arguments', type=pathlib.Path, help="")
    argparser.add_argument('--ztfname', type=str, help="If provided, perform computation on one SN1a. If it points to a valid text file, will perform computation on all keys. If not provided, process the whole working directory.")
    argparser.add_argument('-j', '--n_proc', dest='n_jobs', type=int, default=1)
    argparser.add_argument('--wd', type=pathlib.Path, help="Working directory", required=True)
    argparser.add_argument('--filtercode', choices=ztf_filtercodes, default='all', help="Only perform computations on one or all filters.")
    argparser.add_argument('--func', type=str, help="Pipeline function file to run. {} registered functions.".format(len(pipeline.ops)), required=True)
    argparser.add_argument('--no-map', dest='no_map', action='store_true', help="Skip map operations.")
    argparser.add_argument('--no-reduce', dest='no_reduce', action='store_true', help="Skip reduce operations.")
    argparser.add_argument('--cluster-worker', type=int, default=0)
    argparser.add_argument('--scratch', type=pathlib.Path, help="")
    argparser.add_argument('--from-scratch', action='store_true', help="When using scratch, does not transfer from distant directory first.")
    argparser.add_argument('--exposure-workspace', type=pathlib.Path, help="Exposure workspace directory to use instead of the one given by --wd. Useful to acceleratre IOs by moving onto a SSD disk or in memory mapped filesystem.")
    argparser.add_argument('--lc-folder', dest='lc_folder', type=pathlib.Path)
    argparser.add_argument('--dump-timings', action='store_true')
    argparser.add_argument('--rm-intermediates', action='store_true', help="Remove intermediate files.")
    argparser.add_argument('--synchronous-compute', action='store_true', help="Run computation synchronously on the main thread. Usefull for debugging and plotting on the fly.")
    argparser.add_argument('--log-std', action='store_true', help="If set, output log to standard output.")
    argparser.add_argument('--use-raw', action='store_true', help="If set, uses raw images instead of science images.")
    argparser.add_argument('--discard-calibrated', action='store_true')
    argparser.add_argument('--dump-hw-infos', action='store_true', help="If set, dump hardware informations.")
    argparser.add_argument('--log-overwrite', action='store_true', help="If set, all logs will be overwritten.")
    argparser.add_argument('--parallel-reduce', action='store_true', help="If set, parallelize reduce operations (if op has a parallel codepath).")
    argparser.add_argument('--compress', action='store_true', help="If set, work with compressed working directory")
    argparser.add_argument('--ztfin2p3-path', type=pathlib.Path)
    argparser.add_argument('--ext-catalog-cache', type=pathlib.Path)
    argparser.add_argument('--footprints', type=pathlib.Path)
    argparser.add_argument('--starflats', action='store_true', help="Indicates we are processing starflats. This changes some behaviours in the pipeline (for example no PS1 catalog gets retrieved).")
    argparser.add_argument('--photom-use-aper', action='store_true', help="Use aperture catalog for relative photometry.")
    argparser.add_argument('--ubercal-config-path', type=pathlib.Path, help="Path of the Ubercal configuration file.")
    argparser.add_argument('--slurm', action='store_true', help="Assumes computations are run in a SLURM environnement")
    argparser.add_argument('--list-ops', action='store_true', help="List available pipeline operations.")

    logger = logging.getLogger("main")

    args = argparser.parse_args()

    filtercodes = ztf_filtercodes[:3]
    if args.filtercode != 'all':
        filtercodes = [args.filtercode]

    # Read ztfnames
    ztfnames = None
    if args.ztfname is not None:
        if pathlib.Path(args.ztfname).stem == str(args.ztfname):
            ztfnames = [str(args.ztfname)]
        else:
            args.ztfname = pathlib.Path(args.ztfname).expanduser().resolve()
            if args.ztfname.exists():
                with open(args.ztfname, 'r') as f:
                    ztfnames = [ztfname[:-1] for ztfname in f.readlines()]
            else:
                pass

    print("Found {} exposure stacks.".format(len(ztfnames)))

    pipeline.read_pipeline_from_file(args.func)
    print("Running pipeline:")
    print(" -> ".join([d['op'] for d in pipeline.pipeline_desc]))

    # Temporary folder creation
    # if args.quadrant_workspace or args.scratch:
    #     import signal
    #     import atexit
    #     def delete_tree_at_exit(tree_path):
    #         shutil.rmtree(tree_path, ignore_errors=True)

    #     if args.quadrant_workspace:
    #         args.quadrant_workspace.mkdir(exist_ok=True, parents=True)
    #         atexit.register(delete_tree_at_exit, tree_path=args.quadrant_workspace)

    #     if args.scratch:
    #         args.scratch.mkdir(exist_ok=True, parents=True)
    #         atexit.register(delete_tree_at_exit, tree_path=args.scratch)

    # Allocate cluster
    if args.cluster_worker > 0:
        cluster = SLURMCluster(cores=args.n_jobs,
                               processes=args.n_jobs,
                               memory="{}GB".format(6*args.n_jobs),
                               account="ztf",
                               walltime="6-0",
                               queue="htc",
                               job_extra_directives=["-L sps"],
                               local_directory=os.getenv('TMPDIR', default="."))

        cluster.scale(jobs=args.cluster_worker)
        client = Client(cluster)
        print(client.dashboard_link, flush=True)
        print(socket.gethostname(), flush=True)
        print("Running {} workers with {} processes each ({} total).".format(args.cluster_worker, args.n_jobs, args.cluster_worker*args.n_jobs))
        client.wait_for_workers(1)
    elif not args.synchronous_compute:
        localCluster = LocalCluster(n_workers=args.n_jobs, memory_limit=None, processes=True, threads_per_worker=1, local_directory="{}/dask-workers".format(os.getenv('TMPDIR', default=".")))
        client = Client(localCluster)

        print("Running a local cluster with {} processes.".format(args.n_jobs))
        print("Dask dashboard at: {}".format(client.dashboard_link))
    else:
        print("Running computations synchronously.")

    jobs = []
    map_count = 0
    reduction_count = 0
    map_count = 0

    # Rename compute functions to get better reporting on the dask dashboard
    def _rename_op(op, op_name):
        _op = op
        _op.__name__ = op_name
        return _op

    # If requested, move relevant data into a temporary folder, e.g. stratch
    print("", flush=True)
    if args.scratch:
        print("Moving data into scratch folder.", flush=True)
        for ztfname in ztfnames:
            for filtercode in filtercodes:
                band_path = args.wd.joinpath("{}/{}".format(ztfname, filtercode))
                if band_path.exists():
                    scratch_band_path = args.scratch.joinpath("{}/{}".format(ztfname, filtercode))
                    print("Moving {}-{} into {}".format(ztfname, filtercode, scratch_band_path), flush=True)
                    shutil.rmtree(scratch_band_path, ignore_errors=True)
                    quadrant_folder = args.wd.joinpath("{}/{}".format(ztfname, filtercode))
                    if args.from_scratch:
                        def _from_scratch_ignore(current_folder, files):
                            to_copy = [".dbstuff", "elixir.fits", "dead.fits.gz"]
                            to_ignore = [str(f) for f in map(lambda x: pathlib.Path(x), files) if (str(f.name) not in to_copy) and not (pathlib.Path(current_folder).joinpath(f).is_dir() and str(f)[:4] == "ztf_")]
                            return to_ignore
                        shutil.copytree(band_path, scratch_band_path, symlinks=True, dirs_exist_ok=True, ignore=_from_scratch_ignore)
                    else:
                        shutil.copytree(args.wd.joinpath("{}/{}".format(ztfname, filtercode)), scratch_band_path, symlinks=True)

    # if args.compress:
    #     for ztfname in ztfnames:
    #         for filtercode in filtercodes:
    #             if args.scratch:
    #                 cd = args.scratch
    #             else:
    #                 cd = args.wd

    #             if not cd.joinpath("{}/{}/exposures.tar".format(ztfname, filtercode)).exists():
    #                 continue

    #             print("Uncompressing {}-{} into {}... ".format(ztfname, filtercode, cd), end="", flush=True)
    #             lightcurve = Lightcurve(ztfname, filtercode, cd)
    #             lightcurve.uncompress()
    #             print("Done")

    # Untar folders if needed
    for ztfname in ztfnames:
        for filtercode in filtercodes:
            if args.scratch:
                cd = args.scratch
            else:
                cd = args.wd

            if not cd.joinpath("{}/{}/exposures.tar".format(ztfname, filtercode)).exists():
                continue

            print("Uncompressing {}-{} into {}... ".format(ztfname, filtercode, cd), end="", flush=True)
            lightcurve = Lightcurve(ztfname, filtercode, cd)
            lightcurve.uncompress()
            print("Done")

    # Build compute tree
    for ztfname in ztfnames:
        for filtercode in filtercodes:
            print("Building compute tree for {}-{}... ".format(ztfname, filtercode), flush=True, end="")

            if args.scratch:
                band_path = args.scratch.joinpath("{}/{}".format(ztfname, filtercode))
            else:
                band_path = args.wd.joinpath("{}/{}".format(ztfname, filtercode))

            if not band_path.exists():
                print("No quadrant found.")
                continue

            # Save pipeline arguments and description
            # Since runs can be launched sequencialy without intermediate cleaning, an unique filename is created per run
            # If a clean operation is scheduled, overwrite the first ones (they will be ignored)
            if 'clean' in list([op['op'] for op in pipeline.pipeline_desc]):
                run_count = 0
            else:
                run_count = len(list(band_path.glob("run_arguments_full_*.yaml")))

            shutil.copy(args.func, band_path.joinpath("run_pipeline_{}.txt".format(run_count)))
            if args.run_arguments:
                shutil.copy(args.run_arguments, band_path.joinpath("run_arguments_{}.txt".format(run_count)))

            with open(band_path.joinpath("run_arguments_full_{}.yaml".format(run_count)), 'w') as f:
                d = dict(vars(args))
                for key in d.keys():
                    if isinstance(d[key], pathlib.Path):
                        d[key] = str(d[key])

                yaml.dump(d, f)

            # Save SLURM node configurations
            if args.slurm:
                with open(band_path.joinpath("run_slurm_{}.txt".format(run_count)), 'w') as f:
                    f.write("SLURM_JOB_NODELIST:\t {}\n".format(os.environ['SLURM_JOB_NODELIST']))
                    f.write("SLURM_CPUS_ON_NODE: \t {}\n".format(os.environ['SLURM_CPUS_ON_NODE']))
                    f.write("SLURM_MEM_PER_NODE: \t {}\n".format(os.environ['SLURM_MEM_PER_NODE']))
                    f.write("SLURM_JOB_ID: \t\t {}\n".format(os.environ['SLURM_JOB_ID']) )
                    f.write("SLURM_JOB_NAME: \t {}\n".format(os.environ['SLURM_JOB_NAME']))

            # Dump hardware informations
            if args.dump_hw_infos:
                dump_hw_infos(args.scratch, band_path.joinpath("run_hw_infos_{}.txt".format(run_count)))

            quadrants = quadrants_from_band_path(band_path, logger, paths=False, ignore_noprocess=True)
            print("{} quadrants found.".format(len(quadrants)))

            sn_jobs = []
            map_jobs = []
            reduce_job = None
            sn_job = None
            last_job = None
            for op_desc in pipeline.pipeline_desc:
                op_name = op_desc['op']
                op_parameters = op_desc['parameters']
                wd = args.wd
                if args.scratch and op_name != 'clean':
                    wd = args.scratch

                print("Pipeline operation \"{}\". ".format(op_name), flush=True, end="")
                map_run = False
                if (pipeline.ops[op_name]['map_op'] is not None) and (not args.no_map):
                    map_run = True
                    map_count += len(quadrants)
                    print("(building map jobs)", flush=True, end="")
                    #map_jobs = [delayed(_rename_op(map_op, op_name + "_map"))(quadrant, wd, ztfname, filtercode, pipeline.ops[op_name]['map_op'], args, op_parameters) for quadrant in quadrants]
                    map_jobs = [delayed(_rename_op(map_op, op_name + "_map"))(quadrant, wd, ztfname, filtercode, op_name, args, op_parameters) for quadrant in quadrants]

                    if last_job is not None:
                        print("(checkpoint)", flush=True, end="")
                        map_job = checkpoint(map_jobs)
                        print("(binding)", flush=True, end="")
                        map_job = bind(map_job, last_job)
                    else:
                        map_job = map_jobs

                    print("{} map operations. ".format(len(quadrants)), end="", flush=True)
                else:
                    map_job = None

                if ((pipeline.ops[op_name]['reduce_op'] is not None) or args.dump_timings) and not args.no_reduce:
                    print("(building reduce job)", flush=True, end="")
                    #reduce_job = delayed(_rename_op(reduce_op, op_name + "_reduce"))(wd, ztfname, filtercode, pipeline.ops[op_name]['reduce_op'], True, args, op_parameters)
                    reduce_job = delayed(_rename_op(reduce_op, op_name + "_reduce"))(wd, ztfname, filtercode, op_name, True, args, op_parameters)
                    print("(binding)", flush=True, end="")
                    if map_job is not None:
                        last_job = bind(reduce_job, map_job)
                    else:
                        last_job = bind(reduce_job, last_job)

                    reduction_count += 1
                    print("Reduction operation.", end="", flush=True)
                elif map_run:
                    last_job = checkpoint(map_job)

                print("")

            jobs.append(last_job)

    print("")
    print("Running. ", end="", flush=True)

    if map_count > 0:
        print("Processing {} mappings. ".format(map_count), end="", flush=True)

    if reduction_count > 0:
        print("Processing {} reductions.".format(reduction_count), end="", flush=True)

    print("", flush=True)

    start_time = time.perf_counter()
    if args.synchronous_compute:
        compute(jobs, scheduler="sync")
    else:
        fjobs = client.compute(jobs)
        wait(fjobs)

    end_time = time.perf_counter()

    print("Done. Elapsed time={}".format(end_time - start_time))
    if len(ztfnames) == 1 and len(filtercodes) == 1:
        dump_timings(start_time, end_time, args.wd.joinpath("{}/{}/timings_total".format(ztfnames[0], filtercodes[0])))

    if not args.synchronous_compute:
        client.close(30)
        client.shutdown()

    if args.scratch:
        print("Moving data back from scratch folder into working directory.")
        print("Scratch folder: {}".format(args.scratch))
        print("Working directory: {}".format(args.wd))

        to_ignore = ["elixir.fits", "dead.fits.gz"]
        if args.discard_calibrated:
            to_ignore.extend(["calibrated.fits", "weight.fz"])

        print("Ignoring files {}".format(to_ignore))
        shutil.copytree(args.scratch, args.wd, dirs_exist_ok=True, ignore=shutil.ignore_patterns(*to_ignore))
        shutil.rmtree(args.scratch)
        print("Done")

    if args.compress:
        print("Compressing...")
        for ztfname in ztfnames:
            for filtercode in filtercodes:
                band_path = args.wd.joinpath("{}/{}".format(ztfname, filtercode))
                if band_path.exists():
                    print("{}-{}... ".format(ztfname, filtercode), end="", flush=True)
                    lightcurve = Lightcurve(ztfname, filtercode, args.wd)
                    lightcurve.compress()
                    print("Done")

if __name__ == '__main__':
    sys.exit(main())
