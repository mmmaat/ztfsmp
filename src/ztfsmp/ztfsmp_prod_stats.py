#!/usr/bin/env python3

import argparse
import sys
import pathlib
import importlib

import yaml
import pandas as pd

from ztfsmp.pipeline import pipeline


def flatten_dict(d, subkey=None):
    dd = {}

    for key in d.keys():
        if isinstance(d[key], dict):
            nd = flatten_dict(d[key], key)
        else:
            nd = {key: d[key]}

        if subkey is not None:
            for key in nd.keys():
                dd['{}_{}'.format(subkey, key)] = nd[key]
        else:
            for key in nd.keys():
                dd[key] = nd[key]

    return dd

def retrieve_job_core_count(path):
    with open(path, 'r') as f:
        for line in f.readlines():
            key, val = list(map(lambda x: x.strip(), line.strip().split(":")))
            if key == 'SLURM_CPUS_ON_NODE':
                return int(val)

def main():
    argparser = argparse.ArgumentParser(description="Retrieve stats and info for a given run.")
    argparser.add_argument('--wd', type=pathlib.Path, required=True, help="Working directory.")
    # argparser.add_argument('--run-folder', type=pathlib.Path, required=True)
    argparser.add_argument('--ops', type=pathlib.Path, help="Pipeline description to run. See ztfsmp-pipeline for valid operations.", required=True)
    # argparser.add_argument('--run-name', type=str, required=True)
    argparser.add_argument('--output', type=pathlib.Path, required=False)
    argparser.add_argument('--ztfnames', type=pathlib.Path, default=None, required=False)

    args = argparser.parse_args()

    args.wd = args.wd.expanduser().resolve()
    args.ops = args.ops.expanduser().resolve()
    # args.run_folder = args.run_folder.expanduser().resolve()
    args.output = args.output.expanduser().resolve()

    # Retrieve pipeline description
    print("Reading pipeline description file.")
    import ztfsmp.ztfsmp_pipeline as ztfsmp_pipeline
    search_paths = [pathlib.Path(ztfsmp_pipeline.__file__).parent]
    for search_path in search_paths:
        for ops_module in list(search_path.glob("ops_*.py")):
            globals()[ops_module] = importlib.import_module("ztfsmp." + str(ops_module.stem))

    pipeline.read_pipeline_from_file(args.ops)

    print("Retrieving all light curves.")
    # Retrieve all light curve paths in the given working directory
    if args.ztfnames is not None:
        if args.ztfnames.exists():
            with open(args.ztfnames, 'r') as f:
                lightcurve_paths = [args.wd.joinpath("{}/{}".format(*ztfname.strip().split("-"))) for ztfname in list(f.readlines())]
                print(len(lightcurve_paths))
    else:
        lightcurve_paths = list(args.wd.glob("*/z*"))
    # lightcurve_paths = list(args.wd.glob("{}/z*".format("ZTF18aahqavd")))

    print("Collecting pipeline status for each light curve.")
    d = {}
    timings = {}
    lightcurve_yamls = []
    for lightcurve_path in lightcurve_paths:
        ztfname = lightcurve_path.parts[-2]
        filtercode = lightcurve_path.parts[-1]

        quadrant_paths = list(lightcurve_path.glob("ztf_*"))

        if lightcurve_path.joinpath("lightcurve.yaml").exists():
            with open(lightcurve_path.joinpath("lightcurve.yaml"), 'r') as f:
                lightcurve_yaml = yaml.safe_load(f)

            lightcurve_yaml['ztfname'] = ztfname
            lightcurve_yaml['filtercode'] = filtercode
            lightcurve_yamls.append(flatten_dict(lightcurve_yaml))

        if ztfname not in d.keys():
            d[ztfname] = {filtercode: {}}
            timings[ztfname] = {filtercode: {}}
        else:
            d[ztfname][filtercode] = {}
            timings[ztfname][filtercode] = {}

        d[ztfname][filtercode]['quadrant_count'] = len(quadrant_paths)
        timings[ztfname][filtercode]['quadrant_count'] = len(quadrant_paths)
        if lightcurve_path.joinpath("run_slurm_0.txt").exists():
            timings[ztfname][filtercode]['core_count'] = retrieve_job_core_count(lightcurve_path.joinpath("run_slurm_0.txt"))

        for op in pipeline.pipeline_desc:
            op_name = op['op']
            is_map = (pipeline.ops[op_name]['map_op'] is not None)
            is_reduce = (pipeline.ops[op_name]['reduce_op'] is not None)

            assert not (is_map and is_reduce), "Does not support (yet) pipeline operations having both a mapping and reduction function. op_name={}".format(op_name)


            if is_map:
                d[ztfname][filtercode][op_name] = sum([quadrant_path.joinpath("{}.success".format(op_name)).exists() for quadrant_path in quadrant_paths])

            if is_reduce:
                d[ztfname][filtercode][op_name] = lightcurve_path.joinpath("{}.success".format(op_name)).exists()

            if lightcurve_path.joinpath("timings_{}".format(op_name)).exists():
                with open(lightcurve_path.joinpath("timings_{}".format(op_name)), 'r') as f:
                    timings_op = yaml.safe_load(f)

                timings[ztfname][filtercode][op_name] = timings_op['total']['elapsed']

        print(".", end="", flush=True)

    print("")
    print("Done. Saving.")

    dd = {(ztfname, filtercode): d[ztfname][filtercode] for ztfname in d.keys() for filtercode in d[ztfname].keys()}
    df = pd.DataFrame.from_dict(dd, orient='index')
    df.to_csv(args.output.joinpath("pipeline_status.csv"))
    df.to_parquet(args.output.joinpath("pipeline_status.parquet"))

    dd = {(ztfname, filtercode): timings[ztfname][filtercode] for ztfname in timings.keys() for filtercode in timings[ztfname].keys()}
    timings_df = pd.DataFrame.from_dict(dd, orient='index')
    timings_df.to_csv(args.output.joinpath("timings.csv"))
    timings_df.to_parquet(args.output.joinpath("timings.parquet"))

    yamls_df = pd.DataFrame(lightcurve_yamls).set_index(['ztfname', 'filtercode'])
    yamls_df.to_csv(args.output.joinpath("lightcurve_yamls.csv"))
    yamls_df.to_parquet(args.output.joinpath("lightcurve_yamls.parquet"))


sys.exit(main())
