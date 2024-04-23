#!/usr/bin/env python3

import argparse
import pathlib
import sys
import logging
import subprocess
import shutil
import itertools

import pandas as pd
from joblib import Parallel, delayed


from deppol_utils import run_and_log, load_timings, quadrants_from_band_path
import utils


def get_current_running_sne():
    out = subprocess.run(["squeue", "-o", "%j,%t", "-p", "htc", "-h"], capture_output=True)
    scheduled_jobs_raw = out.stdout.decode('utf-8').split("\n")
    return dict([(scheduled_job.split(",")[0][4:], scheduled_job.split(",")[1]) for scheduled_job in scheduled_jobs_raw if scheduled_job[:4] == "smp_"])

def generate_jobs(wd, run_folder, func, run_name, lightcurves):
    print("Working directory: {}".format(wd))
    print("Run folder: {}".format(run_folder))
    print("Func list: {}".format(func))

    print("Saving jobs under {}".format(run_folder))
    batch_folder = run_folder.joinpath("{}/batches".format(run_name))
    log_folder = run_folder.joinpath("{}/logs".format(run_name))
    status_folder = run_folder.joinpath("{}/status".format(run_name))

    batch_folder.mkdir(exist_ok=True)
    log_folder.mkdir(exist_ok=True)
    status_folder.mkdir(exist_ok=True)

#OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 deppol --ztfname={} --filtercode={} -j {j} --wd={} --func={} --lc-folder=/sps/ztf/data/storage/scenemodeling/lc --exposure-workspace=/dev/shm/llacroix --rm-intermediates --scratch=${{TMPDIR}}/llacroix --astro-degree=5 --max-seeing=4.5 --discard-calibrated --from-scratch --dump-timings --parallel-reduce --use-gaia-stars --ext-catalog-cache=/sps/ztf/data/storage/scenemodeling/cat_cache --footprints=~/data/ztf/starflat_footprints.csv --discard-calibrated
# deppol --ztfname={} --filtercode={} -j {j} --wd={} --func={} --lc-folder=/sps/ztf/data/storage/scenemodeling/lc --rm-intermediates --dump-timings --parallel-reduce --use-gaia-stars --ext-catalog-cache=/sps/ztf/data/storage/scenemodeling/cat_cache --photom-max-star-chi2=4. --photom-cat=ps1 --astro-max-chi2=3. --astro-degree=5 --scratch=$TMPDIR/llacroix --from-scratch
#OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 deppol --ztfname={ztfname} --filtercode={filtercode} -j {j} --wd={wd} --func={func} --lc-folder={lc_folder} --exposure-workspace=/dev/shm/llacroix --rm-intermediates --scratch=${{TMPDIR}}/llacroix --astro-degree=5 --discard-calibrated --from-scratch --dump-timings --parallel-reduce --use-gaia-stars --ext-catalog-cache=/sps/ztf/data/storage/scenemodeling/cat_cache --discard-calibrated
#
    for lightcurve_folder in lightcurves.keys():
        for filtercode in lightcurves[lightcurve_folder]:
            print("{}-{}".format(lightcurve_folder, filtercode))
            job = """#!/bin/sh
echo "running" > {status_path}
conda activate pol
deppol_dask_env.sh
ulimit -n 4096
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Command line for SNe lightcurves
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 deppol --ztfname={ztfname} --filtercode={filtercode} -j $SLURM_NTASKS --wd={wd} --func={func} --lc-folder={lc_folder} --exposure-workspace=/dev/shm/llacroix --rm-intermediates --scratch=${{TMPDIR}}/llacroix --astro-degree=5 --discard-calibrated --from-scratch --dump-timings --parallel-reduce --use-gaia-stars --ext-catalog-cache=/sps/ztf/data/storage/scenemodeling/cat_cache

# Command line for starflats
#OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 deppol --ztfname={ztfname} --filtercode={filtercode} -j $SLURM_NTASKS --wd={wd} --func={func} --rm-intermediates --dump-timings --use-gaia-stars
echo "done" > {status_path}
""".format(ztfname=lightcurve_folder, filtercode=filtercode, wd=wd, func=",".join(func), status_path=run_folder.joinpath("{}/status/{}-{}".format(run_name, lightcurve_folder, filtercode)), j=args.ntasks, lc_folder="/sps/ztf/data/storage/scenemodeling/lc")
            with open(batch_folder.joinpath("{}-{}.sh".format(lightcurve_folder, filtercode)), 'w') as f:
                f.write(job)


def schedule_jobs(run_folder, run_name, lightcurves):
    print("Run folder: {}".format(run_folder))
    batch_folder = run_folder.joinpath("{}/batches".format(run_name))
    log_folder = run_folder.joinpath("{}/logs".format(run_name))
    status_folder = run_folder.joinpath("{}/status".format(run_name))
    status_folder.mkdir(exist_ok=True)

    logger = logging.getLogger("schedule_jobs")
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)

    batches = list(itertools.chain.from_iterable([["{}-{}.sh".format(lightcurve_folder, band) for band in lightcurves[lightcurve_folder]] for lightcurve_folder in lightcurves.keys()]))
    batches = list(map(lambda x: run_folder.joinpath("{}/{}/batches/{}".format(run_folder, run_name, x)), batches))

    # Write batch list in run folder
    with open(run_folder.joinpath("{}/lightcurves.txt".format(run_name)), 'w') as f:
        for batch in batches:
            f.write("{}\n".format(batch))

    # First get currently running lightcurves
    scheduled_jobs = get_current_running_sne()

    # Submit lightcurve jobs
    for batch in batches:
        batch_name = batch.name.split(".")[0]

        # If lightcurve is already running, do not submit it again
        if batch_name in scheduled_jobs.keys():
            continue

        # Check if lightcurve already ran
        batch_status_path = status_folder.joinpath(batch_name)
        if batch_status_path.exists() and not args.ztfname:
            with open(batch_status_path, 'r') as f:
                status = f.readline().strip()
                if status == "done":
                    continue

        # If not, submit it through the SLURM batch system
        # Configured to run @ CCIN2P3
        cmd = ["sbatch", "--ntasks={}".format(args.ntasks),
               "-D", "{}".format(run_folder.joinpath(run_name)),
               "-J", "smp_{}".format(batch_name),
               "-o", log_folder.joinpath("log_{}".format(batch_name)),
               "-A", "ztf",
               "-L", "sps",
               "--mem={}G".format(4*args.ntasks),
               "-t", "5-0",
               batch]

        returncode = run_and_log(cmd, logger)

        with open(batch_status_path, 'w') as f:
            if returncode == 0:
                f.write("scheduled")
            else:
                f.write("failedtoschedule")
                break
        print("{}: {}".format(batch_name, returncode))


def lightcurves_from_ztfname(wd, ztfname):
    lightcurves = {}

    # In that case, no ztfname is specified. We simply build the lightcurve list from the working directory
    if ztfname is None:
        for lightcurve_folder in list(wd.glob("*")):
            bands = []
            for filtercode in utils.filtercodes:
                if lightcurve_folder.joinpath(filtercode).exists():
                    bands.append(filtercode)

            if len(bands) == 0:
                continue

            lightcurves[lightcurve_folder.name] = bands

    # ztfname corresponds to a valid file. Read lightcurves from it
    elif ztfname.exists():
        with open(ztfname, 'r') as f:
            lines = list(map(lambda x: x.strip(), f.readlines()))

        for line in lines:
            # This line specify a lightcurve_folder and a band
            if "-" in line:
                lightcurve_folder, band = line.split("-")
                if lightcurve_folder not in lightcurves.keys():
                    lightcurves[lightcurve_folder] = [band]
                else:
                    lightcurves[lightcurve_folder] = list(set(lightcurves[lightcurve_folder] + [band]))

            # This line only specify a lightcurve_folder
            else:
                lightcurve_folder = line
                bands = list(map(lambda x: x.name, list(filter(lambda x: x not in utils.filtercodes, list(wd.joinpath(lightcurve_folder).glob("z*"))))))
                if len(bands) == 0:
                    continue

                if lightcurve_folder not in lightcurves.keys():
                    lightcurves[lightcurve_folder] = bands
                else:
                    lightcurves[lightcurve_folder] = list(set(lightcurves[lightcurve_folder] + bands))

    # ztfname does not correspond to a filename, might be the name of a lightcurve_folder or lightcurve
    else:
        ztfname = ztfname.name

        # SN and band combination
        if "-" in ztfname:
            lightcurve_folder, band = ztfname.split("-")
            if band in utils.filtercodes and wd.joinpath("{}/{}".format(lightcurve_folder, band)).exists():
                lightcurves[lightcurve_folder] = [band]
            else:
                print("For SN/lightcurve_folder {}, {} does not correspond to a valid filtercode! (or there might be no corresponding folder)".format(lightcurve_folder, band, utils.filtercodes)) # Cryptic error message
                sys.exit()

        # Only a SN, get all bands from working directory
        elif wd.joinpath(ztfname).exists():
            bands = list(filter(lambda x: x in utils.filtercodes, list(wd.joinpath(ztfname).glob("z*"))))
            lightcurves[ztfname] = bands

        else:
            print("{} does not correspond to a valid ztfname! (read --help menu to see what constitute a valid one)".format(ztfname))
            sys.exit()

    return lightcurves


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description="")
    argparser.add_argument('--generate-jobs', action='store_true', help="If set, generate list of jobs")
    argparser.add_argument('--schedule-jobs', action='store_true', help="If set, schedule jobs onto SLURM")
    argparser.add_argument('--wd', type=pathlib.Path, required=False, help="Working directory.")
    argparser.add_argument('--run-folder', type=pathlib.Path, required=True)
    argparser.add_argument('--func', type=pathlib.Path, help="Comma separated list of pipeline operations. If a valid file, read from it instead, where each line correspond to a pipeline operation.")
    argparser.add_argument('--run-name', type=str, required=True)
    argparser.add_argument('--ntasks', default=1, type=int, help="Number of worker to use when submitting jobs")
    argparser.add_argument('--purge-status', action='store_true')
    argparser.add_argument('--purge', action='store_true')
    argparser.add_argument('--ztfname', type=pathlib.Path, help="If left empty, process all lightcurves in the working directory. If it corresponds to one lightcurve folder (such as a SN folder), process this one in each available bands. If it corresponds to a lightcurve folder name and a filtercode, separated by \"-\" (e.g. ZTF19aaripqw-zg), only process the specified lightcurve. If set to a valid filename, interpret each line of it in the same way as previously described")

    args = argparser.parse_args()

    if args.ztfname:
        args.ztfname = args.ztfname.expanduser().resolve()

    if args.wd:
        args.wd = args.wd.expanduser().resolve()
        if not args.wd.exists():
            sys.exit("Working folder does not exist!")

    if args.purge_status:
        status_folder = args.run_folder.joinpath("{}/status".format(args.run_name))
        if status_folder.exists():
            shutil.rmtree(status_folder)

    if not args.run_folder.exists():
        sys.exit("Run folder does not exist!")

    if args.func:
        funcs = []
        # If points to a correct file, read from it
        if args.func.exists():
            with open(args.func, 'r') as f:
                funcs = list(map(lambda x: x.strip(), f.readlines()))

        # Comma sepparated operations
        else:
            funcs = str(args.func).split(",")

        # Check operations are valid
        import imp
        deppol = imp.load_source('deppol', "deppol") # This is deprecated but there are more important things at hand...
        for func in funcs:
            if func not in deppol.poloka_func.keys():
                print("{} pipeline operation does not exists! Exiting...".format(func))
                sys.exit()

    args.run_folder.joinpath(args.run_name).mkdir(exist_ok=True)

    print("Building lightcurve list...")
    lightcurves = lightcurves_from_ztfname(args.wd, args.ztfname)
    print("Found {} lightcurve folders, totalling {} lightcurves!".format(len(lightcurves), len(list(itertools.chain.from_iterable(list(lightcurves.values()))))))

    if args.generate_jobs:
        generate_jobs(args.wd, args.run_folder, funcs, args.run_name, lightcurves)

    if args.schedule_jobs:
        schedule_jobs(args.run_folder, args.run_name, lightcurves)
