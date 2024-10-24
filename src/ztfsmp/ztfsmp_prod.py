#!/usr/bin/env python3

import importlib
import argparse
import pathlib
import sys
import logging
import subprocess
import shutil
import itertools
import os 

from joblib import Parallel, delayed

from ztfsmp.pipeline import pipeline
from ztfsmp.pipeline_utils import run_and_log
from ztfsmp.ztf_utils import filtercodes

def create_script_submit_under_apptainer(batch):
    """
    batch : path/name of script SLURM batch
    """
    dir_batch = os.path.dirname(batch)
    name_batch =  os.path.basename(batch)
    print(dir_batch, name_batch)    
    if os.getenv('ZTF_EXT_ENV') is not None:
        script_app = f"""#!/bin/sh
echo "==================="
echo "Job under apptainer"   
echo "==================="    

apptainer exec --bind /pbs,/sps/ztf,/scratch $ZTF_APPTAINER /usr/local/bin/_entrypoint.sh $ZTF_BOOT_APP $ZTF_EXT_ENV {batch}
"""
    else:
        app_name = os.getenv('APPTAINER_CONTAINER')
        script_app = f"""#!/bin/sh
echo "==================="
echo "Job under apptainer"   
echo "==================="    

apptainer exec --bind /pbs,/sps/ztf,/scratch {app_name} /usr/local/bin/_entrypoint.sh {batch}
"""        
    pn_script = f"{dir_batch}/apptainer/app_{name_batch}" 
    with open(pn_script, 'w') as f:
        f.write(script_app)
    os.chmod(pn_script, 0o775)
    return pn_script

def get_current_running_sne():
    out = subprocess.run(["squeue", "-o", "%j,%t", "-p", "htc", "-h"], capture_output=True)
    scheduled_jobs_raw = out.stdout.decode('utf-8').split("\n")
    return dict([(scheduled_job.split(",")[0].split("_")[-1], scheduled_job.split(",")[1]) for scheduled_job in scheduled_jobs_raw if scheduled_job[:7] == "ztfsmp_"])

def generate_jobs(wd, run_folder, ops, run_name, lightcurves, run_arguments):
    script_name = 'ztfsmp-pipeline'
    f_apptainer = os.getenv('APPTAINER_NAME') is not None
    
    
    print("Working directory: {}".format(wd))
    print("Run folder: {}".format(run_folder))
    print("Pipeline description: {}".format(ops))

    print("Saving jobs under {}".format(run_folder))
    batch_folder = run_folder.joinpath("{}/batches".format(run_name))
    log_folder = run_folder.joinpath("{}/logs".format(run_name))
    status_folder = run_folder.joinpath("{}/status".format(run_name))
    batch_folder.mkdir(exist_ok=True)
    
    if f_apptainer:
        # We are in apptainer env, need to create specific slurm batches for apptainer
        batch_folder.joinpath("apptainer").mkdir(exist_ok=True)
        # Manage magic rename of ztfsmp_pipeline.py by pip install
        if shutil.which('ztfsmp_pipeline.py') is not None:
            # a ztfsmp version is defined as an external package of apptainer env
            script_name = 'ztfsmp_pipeline.py'
    log_folder.mkdir(exist_ok=True)
    status_folder.mkdir(exist_ok=True)

    for lightcurve_folder in lightcurves.keys():
        for filtercode in lightcurves[lightcurve_folder]:
            print("{}-{}".format(lightcurve_folder, filtercode))
            job = """#!/bin/sh
echo "running" > {status_path}
ulimit -n 4096

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 {pipeline} --ztfname={ztfname} --filtercode={filtercode} -j $SLURM_CPUS_ON_NODE --wd={wd} --func={ops} --run-arguments={run_arguments}

echo "done" > {status_path}
""".format(pipeline=script_name, ztfname=lightcurve_folder, filtercode=filtercode, wd=wd, ops=ops, status_path=run_folder.joinpath("{}/status/{}-{}".format(run_name, lightcurve_folder, filtercode)), run_arguments=run_arguments)            
            pn_script = batch_folder.joinpath("{}-{}.sh".format(lightcurve_folder, filtercode)) 
            with open(pn_script, 'w') as f:
                f.write(job)
            os.chmod(pn_script, 0o775)
            #
            if f_apptainer:
                create_script_submit_under_apptainer(pn_script)


def schedule_jobs(run_folder, run_name, lightcurves, ntasks, gb_per_task, force_reprocessing):
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
        print(batch)
        batch_name = batch.name.split(".")[0]

        # If lightcurve is already running, do not submit it again
        if batch_name in scheduled_jobs.keys():
            continue

        # Check if lightcurve already ran. If provided in lightcurves, force reprocessing
        batch_status_path = status_folder.joinpath(batch_name)
        if batch_status_path.exists() and not force_reprocessing:
                with open(batch_status_path, 'r') as f:
                    status = f.readline().strip()
                    if status == "done":
                        continue

        # If not, submit it through the SLURM batch system
        # Configured to run @ CCIN2P3
        cmd = ["sbatch", "--ntasks={}".format(ntasks),
               "-D", "{}".format(run_folder.joinpath(run_name)),
               "-J", "ztfsmp_{}_{}".format(run_name, batch_name),
               "-o", log_folder.joinpath("log_{}".format(batch_name)),
               "-A", "ztf",
               "-L", "sps",
               "--mem={}G".format(gb_per_task*ntasks),
               "-t", "7-0",
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
            for filtercode in filtercodes:
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
                bands = list(map(lambda x: x.name, list(filter(lambda x: x not in filtercodes, filter(lambda x: x.name in filtercodes, list(wd.joinpath(lightcurve_folder).glob("z*")))))))
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
            if band in filtercodes and wd.joinpath("{}/{}".format(lightcurve_folder, band)).exists():
                lightcurves[lightcurve_folder] = [band]
            else:
                print("For SN/lightcurve_folder {}, {} does not correspond to a valid filtercode! (or there might be no corresponding folder)".format(lightcurve_folder, band, filtercodes)) # Cryptic error message
                sys.exit()

        # Only a SN, get all bands from working directory
        elif wd.joinpath(ztfname).exists():
            bands = list(filter(lambda x: x in filtercodes, list(wd.joinpath(ztfname).glob("z*"))))
            lightcurves[ztfname] = bands

        else:
            print("{} does not correspond to a valid ztfname! (read --help menu to see what constitute a valid one)".format(ztfname))
            sys.exit()

    return lightcurves


def main():
    argparser = argparse.ArgumentParser(description="Deploy ztfsmp on a SLURM cluster. For now only support CC IN2P3 cluster.")
    argparser.add_argument('--generate-jobs', action='store_true', help="If set, generate list of jobs")
    argparser.add_argument('--schedule-jobs', action='store_true', help="If set, schedule jobs onto SLURM")
    argparser.add_argument('--wd', type=pathlib.Path, required=False, help="Working directory.")
    argparser.add_argument('--run-folder', type=pathlib.Path, required=True)
    argparser.add_argument('--ops', type=pathlib.Path, help="Pipeline description to run. See ztfsmp-pipeline for valid operations.")
    argparser.add_argument('--run-name', type=str, required=True)
    argparser.add_argument('--ntasks', default=1, type=int, help="Number of worker to use when submitting jobs")
    argparser.add_argument('--purge-status', action='store_true')
    argparser.add_argument('--ztfname', type=pathlib.Path, help="If left empty, process all lightcurves in the working directory. If it corresponds to one lightcurve folder (such as a SN folder), process this one in each available bands. If it corresponds to a lightcurve folder name and a filtercode, separated by \"-\" (e.g. ZTF19aaripqw-zg), only process the specified lightcurve. If set to a valid filename, interpret each line of it in the same way as previously described")
    argparser.add_argument('--run-arguments', type=pathlib.Path)
    argparser.add_argument('--gb-per-task', type=int, default=4, help="Number of GB to query per tasks.")
    argparser.add_argument('--force-reprocessing', action='store_true', help="If set, reschedule lightcurves even if they already ran.")

    args = argparser.parse_args()
    args.run_folder = args.run_folder.expanduser().resolve()

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

    args.run_folder.joinpath(args.run_name).mkdir(exist_ok=True)

    print("Building lightcurve list...")
    lightcurves = lightcurves_from_ztfname(args.wd, args.ztfname)
    print("Found {} lightcurve folders, totalling {} lightcurves!".format(len(lightcurves), len(list(itertools.chain.from_iterable(list(lightcurves.values()))))))

    if args.generate_jobs:
        args.run_arguments = args.run_arguments.expanduser().resolve()
        args.ops = args.ops.expanduser().resolve()

        # Check the pipeline description file is valid
        import ztfsmp.ztfsmp_pipeline as ztfsmp_pipeline
        search_paths = [pathlib.Path(ztfsmp_pipeline.__file__).parent]
        for search_path in search_paths:
            for ops_module in list(search_path.glob("ops_*.py")):
                globals()[ops_module] = importlib.import_module("ztfsmp." + str(ops_module.stem))

        pipeline.read_pipeline_from_file(args.ops)

        generate_jobs(args.wd, args.run_folder, args.ops, args.run_name, lightcurves, args.run_arguments)

    if args.schedule_jobs:
        schedule_jobs(args.run_folder, args.run_name, lightcurves, args.ntasks, args.gb_per_task, args.force_reprocessing)


if __name__ == '__main__':
    sys.exit(main())
