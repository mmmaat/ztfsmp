#!/usr/bin/env python3

import os
from collections.abc import Iterable
import pathlib

from ztfsmp.op_parameters import OpParameters, OpParameter, OpParameterDesc


def strtobool (val):
    """Convert a string representation of truth to true (1) or false (0).
    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.

    Copied and adapted from distutils.util.strtobool() (which is deprecated for Python >= 3.12)
    """
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return True
    elif val in ('n', 'no', 'f', 'false', 'off', '0'):
        return False
    else:
        raise ValueError("invalid truth value %r" % (val,))


class Pipeline:
    def __init__(self):
        super().__init__()
        self.__ops = {}
        self.__pipeline_desc = []

    @property
    def ops(self):
        return self.__ops

    @property
    def pipeline_desc(self):
        return self.__pipeline_desc

    def register_op(self, name, map_op=None, reduce_op=None, plot_op=None, rm_list=None, op_parameters_desc=None):
        self.__ops[name] = {'map_op': map_op,
                            'reduce_op': reduce_op,
                            'plot_op': plot_op,
                            'rm_list': rm_list,
                            'parameters': op_parameters_desc}

    def _parse_parameters_from_str(self, op_name, s):
        parameters_str = s.strip().split(",")
        parameters = {}
        if parameters_str != "":
            for parameter_str in parameters_str:
                key, value = parameter_str.strip().split("=")
                key = key.strip()

                if key not in self.__ops[op_name]['parameters'].keys():
                    raise KeyError("\'{}\' not in parameter list! Available parameters: {}".format(key, list(self.__ops[op_name]['parameters'].keys())))

                if self.__ops[op_name]['parameters'][key].type is bool:
                    parameters[key] = strtobool(value.strip())
                else:
                    parameters[key] = self.__ops[op_name]['parameters'][key].type(value.strip())

        return parameters

    def _parse_op_parameters_from_str(self, s):
        if ("(") in s and (")" in s):
            op_name, parameter_str = s.strip().split("(")
            op_name = op_name.strip()
            op_parameter_str = parameter_str.strip().split(")")[0].strip()
        else:
            op_name = s.strip()
            op_parameter_str = None

            if op_name not in self.__ops.keys():
                raise KeyError("{} is not a recognised pipeline op! Please use --help to get full pipeline op list.".format(op_name))

            op_parameters = self._parse_parameters_from_str(op_name, op_parameter_str)
            for op_parameter in op_parameters:
                pass

            self.__pipeline_desc.append({'op': op_name, 'parameters': self.__ops[op_name]['parameters']})

    def read_pipeline_from_file(self, path):
        with open(path, 'r') as f:
            lines = f.readlines()

        lines = list(map(lambda x: x.strip(), lines)) # Remove spaces and line jumps
        lines = list(filter(lambda x: x[0]!="#", lines)) # Removed commented lines

        for line in lines:
            if ("(" in line) and (")" in line):
                op_name, parameter_str = line.split("(")
                op_name = op_name.strip()
                op_parameter_str = parameter_str.strip().split(")")[0].strip()
            else:
                op_name = line.strip()
                op_parameter_str = None

            if op_name not in self.__ops.keys():
                raise KeyError("{} is not a recognised pipeline op! Please use --help to get full pipeline op list.".format(op_name))

            op = {'op': op_name, 'parameters': OpParameters(self.__ops[op_name]['parameters'])}

            if (op_parameter_str is not None) and (op_parameter_str != ""):
                parameter_values = self._parse_parameters_from_str(op_name, op_parameter_str)
                assert len(parameter_values) > 0
                for parameter_key in parameter_values.keys():
                    if self.__ops[op_name]['parameters'][parameter_key].type == pathlib.Path:
                        op['parameters'][parameter_key] = pathlib.Path(os.path.expandvars(parameter_values[parameter_key]))
                    else:
                        op['parameters'][parameter_key] = parameter_values[parameter_key]

            self.__pipeline_desc.append(op)

    def read_pipeline_from_str(self, s):
        ops = s.strip().split(";")
        for op in ops:
            pass


# In my coding journey, I've sinned—I used a global variable to be used as a singleton.
# As deadlines loomed, I muttered, "Only God can judge me."
# Amidst all coding doctrines, here I stand, judged perhaps by many, but truly, only by the heavens.
# May they find humor in my mortal choices.


pipeline = Pipeline()


def register_op(name, map_op=None, reduce_op=None, plot_op=None, rm_list=None, parameters=None):
    assert name not in list(pipeline.ops.keys()), "{} pipeline operation already registed!".format(name)
    if isinstance(parameters, dict):
        parameters = [parameters]

    if parameters is None:
        op_parameters_desc = {}
    else:
        op_parameters_desc = dict([(parameter['name'], OpParameterDesc(parameter['name'], parameter['type'], parameter.get('default', None), parameter.get('desc', None))) for parameter in parameters])

    pipeline.register_op(name, map_op=map_op, reduce_op=reduce_op, plot_op=plot_op, rm_list=rm_list, op_parameters_desc=op_parameters_desc)
