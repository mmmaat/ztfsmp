#!/usr/bin/env python3

import pathlib
from collections.abc import Iterable

import pandas as pd


def read_list(f):
    if isinstance(f, str) or isinstance(f, pathlib.Path):
        with open(f, 'r') as fi:
            header, df, _, _ = read_list_ext(fi)

    else:
        header, df, _, _ = read_list_ext(f)

    return header, df


def read_list_ext(f, delim_whitespace=True):
    # Extract global @ parameters
    header = {}
    line = f.readline().strip()
    curline = 1
    while line[0] == "@":
        # line = line[:-1]
        splitted = line.split()
        key = splitted[0][1:]
        splitted = splitted[1:]

        first = splitted[0]
        if first[0] == '-':
            first = first[1:]

        if first.isdigit():
            t = int
        else:
            try:
                float(first)
            except ValueError:
                t = str
            else:
                t = float

        values = list(map(t, splitted))

        if len(values) == 1:
            values = values[0]

        header[key.lower()] = values

        line = f.readline()
        curline += 1

    # Extract dataframe
    # First, column names
    columns = []
    df_format = None
    df_desc = {}
    while line[0] == "#":
        curline += 1
        line = line[1:-1].strip()

        splitted = line.split(" ")
        if splitted[0].strip() == "end":
            break

        if splitted[0].strip() == "format":
            df_format = str(" ".join(line.split(" ")[1:])).strip()
            line = f.readline()
            continue

        splitted = line.split(":")
        column = splitted[0].strip()
        columns.append(column)
        if len(splitted) > 1:
            df_desc[column] = splitted[1].strip()

        line = f.readline()

    if delim_whitespace:
        df = pd.read_csv(f, delim_whitespace=True, names=columns, index_col=False, skipinitialspace=True)
    else:
        df = pd.read_csv(f, sep=' ', names=columns, index_col=False, skipinitialspace=False)

    return header, df, df_desc, df_format


def write_list(filename, header, df, df_desc, df_format):
    with open(filename, 'w') as f:
        if header:
            for key in header.keys():
                if isinstance(header[key], Iterable) and not isinstance(header[key], str):
                    f.write("@{} {}\n".format(key.upper(), " ".join(map(str, header[key]))))
                else:
                    f.write("@{} {}\n".format(key.upper(), str(header[key])))

        for column in df.columns:
            if column in df_desc.keys():
                f.write("#{} : {}\n".format(column, df_desc[column]))
            else:
                f.write("#{} :\n".format(column))

        if df_format is not None:
            f.write("# format {}\n".format(df_format))

        f.write("# end\n")

        df.to_csv(f, sep=" ", index=False, header=False)


class ListTable:
    def __init__(self, header, df, df_desc=None, df_format=None, filename=None):
        if not df_desc:
            df_desc = {}

        self.header = header
        self.df = df
        self.df_desc = df_desc
        self.df_format = df_format
        self.filename = filename

    @classmethod
    def from_filename(cls, filename, delim_whitespace=True):
        with open(filename, 'r') as f:
            header, df, df_desc, df_format = read_list_ext(f, delim_whitespace=delim_whitespace)

        return cls(header, df, df_desc, df_format, filename=filename)

    def write_to(self, filename):
        write_list(filename, self.header, self.df, self.df_desc, self.df_format)

    def write_to_csv(self, filename):
        self.df.to_csv(filename)

    def write(self):
        self.write_to(self.filename)

    def write_csv(self):
        self.write_to_csv(self.filename.with_suffix(".csv"))
