#!/usr/bin/env python3

import pathlib


class OpParameterDesc:
    """
    Parameter description helper class. Store parameter name, type, default value and small text description.
    """
    def __init__(self, name, param_type, default=None, desc=""):
        self.__name = name
        self.__type = param_type

        # If no default parameter is set, get the default for the given type. If type is a path, keep None.
        if default is None and param_type != pathlib.Path:
            self.__default = self.__type()
        else:
            self.__default = default

        if desc == "":
            self.__desc = "/"
        else:
            self.__desc = desc

    @property
    def name(self):
        return self.__name

    @property
    def type(self):
        return self.__type

    @property
    def default(self):
        return self.__default

    @property
    def desc(self):
        return self.__desc

    def __repr__(self):
        return "OpParameterDesc(name=\'{}\',type={},default={},desc=\"{}\")".format(self.__name, self.__type.__name__, self.__default, self.__desc)

    def __str__(self):
        return "Name: {}\nType={}\nDefault={}\nDescription={}".format(self.__name, self.__type.__name__, self.__default, self.__desc)


class OpParameter:
    """
    Store a parameter following some parameter description.
    """
    def __init__(self, op_parameter_desc, value=None):
        self.__parameter_desc = op_parameter_desc

        if value is None:
            value = self.__parameter_desc.default

        self.__value = value

    @property
    def value(self):
        return self.__value

    @value.setter
    def value(self, value):
        if not isinstance(value, self.__parameter_desc.type):
            raise ValueError("\'{}\' is of type {}! Must be {}".format(self.__parameter_desc.name, type(value), self.__parameter_desc.type.__name__))

        self.__value = value

    def __repr__(self):
        return "OpParameter(OpParameterDesc={}, value={})".format(self.__parameter_desc.__repr__(), self.__value)

    def __str__(self):
        return "{}\nValue={}".format(self.__parameter_desc, self.__value)


class OpParameters:
    """
    Store a collection of parameters.
    """
    def __init__(self, op_parameters_desc, op_parameters=None):
        self.__op_parameters_desc = op_parameters_desc
        self.__op_parameters = dict([(op_parameter_name, OpParameter(self.__op_parameters_desc[op_parameter_name])) for op_parameter_name in self.__op_parameters_desc.keys()])
        if op_parameters is not None:
            for op_parameter_key in op_parameters.keys():
                self.__op_parameters[op_parameter_key].value = op_parameters[op_parameter_key]

    @property
    def op_parameters_desc(self):
        return self.__op_parameters_desc

    def __getitem__(self, key):
        return self.__op_parameters[key].value

    def __setitem__(self, key, value):
        self.__op_parameters[key].value = value

    def __repr__(self):
        return "OpParameters({})".format(self.__op_parameters)

    def __str__(self):
        s = ""
        for key in self.__op_parameters_desc.keys():
            s += str(self.__op_parameters[key]) + "\n\n"

        return s

if __name__ == '__main__':
    desc = [OpParameterDesc('cat', str, 'gaia', "Catalog to use"),
            OpParameterDesc('threshold', float),
            OpParameterDesc('use_cat', bool)]
    params = OpParameters(desc)
    print(repr(params))
