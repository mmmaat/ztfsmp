#!/usr/bin/env python3

import os
import pathlib
import subprocess
import time
from collections.abc import Iterable
import json
import yaml
import tarfile
import shutil
import yaml


def run_and_log(cmd, logger=None, return_log=False):
    if logger:
        logger.info("Running command: \"{}\"".format(" ".join([str(s) for s in cmd])))
        start_time = time.perf_counter()

    out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, universal_newlines=True)

    if logger:
        logger.info("Done running command. Elapsed time={}".format(time.perf_counter() - start_time))
        logger.info("Command stdout/stderr output:")
        logger.info(out.stdout)
        logger.info("=========================== output end ===========================")

    if return_log:
        return out.returncode, out.stdout

    return out.returncode


def dump_timings(start_time, end_time, output_file):
    with open(output_file, 'w') as f:
        f.write(json.dumps({'start': start_time, 'end': end_time, 'elapsed': end_time-start_time}))


def dump_timings_reduce(start_times, end_times, output_file):
    with open(output_file, 'w') as f:
        if start_times['map'] is None and end_times['map'] is None:
            f.write(json.dumps({'map': {'start': 0., 'end': 0., 'elapsed': 0.},
                                'reduce': {'start': start_times['reduce'], 'end': end_times['reduce'], 'elapsed': end_times['reduce']-start_times['reduce']},
                                'total': {'start': start_times['reduce'], 'end': end_times['reduce'], 'elapsed': end_times['reduce']-start_times['reduce']}}))
        else:
            f.write(json.dumps({'map': {'start': start_times['map'], 'end': end_times['map'], 'elapsed': end_times['map']-start_times['map']},
                                'reduce': {'start': start_times['reduce'], 'end': end_times['reduce'], 'elapsed': end_times['reduce']-start_times['reduce']},
                                'total': {'start': start_times['map'], 'end': end_times['reduce'], 'elapsed': end_times['reduce']-start_times['map']}}))

def load_timings(filename):
    with open(filename, 'r') as f:
        return json.load(f)


def ztfnames_from_string(ztfname):
    if ztfname is not None:
        if ztfname.stem == str(ztfname):
            return [str(ztfname)]
        else:
            ztfname = ztfname.expanduser().resolve()
            if ztfname.exists():
                with open(ztfname, 'r') as f:
                    ztfnames = [ztfname[:-1] for ztfname in f.readlines()]
                return ztfnames
            else:
                pass


def quadrants_from_band_path(band_path, logger, check_files=None, paths=True, ignore_noprocess=False):
    if not ignore_noprocess:
        noprocess = noprocess_quadrants(band_path)
    else:
        noprocess = []

    quadrant_paths = [quadrant_path for quadrant_path in list(band_path.glob("ztf*")) if quadrant_path.name not in noprocess]

    if check_files:
        if not isinstance(check_files, Iterable) or isinstance(check_files, str) or isinstance(check_files, pathlib.Path):
            check_files = [check_files]

        def _check_files(quadrant_path):
            check_ok = True
            for check_file in check_files:
                if not quadrant_path.joinpath(check_file).exists():
                    check_ok = False
                    break

            return check_ok

        quadrant_paths = list(filter(_check_files, quadrant_paths))

    if paths:
        return quadrant_paths
    else:
        return [quadrant_path.name for quadrant_path in quadrant_paths]


def noprocess_quadrants(band_path):
    noprocess = []
    if band_path.joinpath("noprocess").exists():
        with open(band_path.joinpath("noprocess"), 'r') as f:
            for line in f.readlines():
                quadrant = line.strip()
                if quadrant[0] == "#":
                    continue
                elif band_path.joinpath(quadrant).exists():
                    noprocess.append(quadrant)

    return noprocess


def update_yaml(path, key, value):
    _yaml = {}
    if path.exists():
        with open(path, 'r') as f:
            _yaml = yaml.load(f, Loader=yaml.Loader)

    if _yaml is None:
        _yaml = {}

    with open(path, 'w') as f:
        _yaml[key] = value
        yaml.dump(_yaml, f)

