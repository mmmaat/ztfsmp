#!/usr/bin/env python3

import os
import sys
import argparse
import pathlib

class RunArguments(argparse.ArgumentParser):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def parse_args(self, args=None, namespace=None):
        if args is None:
            args = sys.argv[1:]

        parsed = super().parse_args(args, namespace)

        # If run arguments are provided in a file, first load those
        # if 'run_arguments' in vars(parsed).keys():
        if parsed.run_arguments is not None:
            run_arguments = []
            run_arguments_path = vars(parsed)['run_arguments'].expanduser().resolve()

            with open(vars(parsed)['run_arguments'], 'r') as f:
                lines = f.readlines()
                for line in lines:
                    # ignore comments (all after '#')
                    line = ''.join(line.split('#')[:1]).strip()
                    if line:
                        run_arguments.extend(line.split())

            # Resolve environement variables
            for i in range(len(run_arguments)):
                run_argument = run_arguments[i]
                if "$" in run_argument:
                    run_arguments[i] = os.path.expandvars(run_argument)

            # Arguments are overriden by command line
            parsed = super().parse_args(run_arguments+args, namespace)

        # Resolve all paths
        for key in vars(parsed).keys():
            if isinstance(vars(parsed)[key], pathlib.Path):
                vars(parsed)[key] = vars(parsed)[key].expanduser().resolve()

        return parsed
