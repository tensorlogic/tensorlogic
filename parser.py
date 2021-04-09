import re

from structs import TensorData


def read_program_from_file(program_path):
    with open(program_path, 'r') as f:
        for line in f.readlines():
            yield line


def get_program_lines(program_line):

    for line in program_line:

        line = line.strip()
        line = re.sub(' +', ' ', line)
        line = re.sub(r" ?, ?", ",", line)
        line = re.sub(r" ?= ?", "=", line)
        line = re.sub(r":- ?([^[]+\[)", r":-\1", line)
        line = re.sub(r"] ?:-", r"]:-", line)

        if len(line):
            line = line.split(" ")
            yield list(filter(lambda x: "=" not in x, line)), list(filter(lambda x: "=" in x, line))


def parse_program_line(program, info, flags):

    id_string = info[0]
    flags = _parse_flags(flags)

    if id_string in RangeProgramLine.id_strings:
        RangeProgramLine.add_to_program(program, info, flags)
    elif id_string in TensorProgramLine.id_strings:
        TensorProgramLine.add_to_program(program, info, flags)
    elif id_string in InputProgramLine.id_strings:
        InputProgramLine.add_to_program(program, info, flags)
    elif id_string in OpProgramLine.id_strings:
        OpProgramLine.add_to_program(program, info, flags)
    else:
        raise ValueError("Unknown id_string", id_string)


def _parse_tuple(s, t=None):
    if t is None:
        return tuple(v for v in s[1:-1].split(","))
    elif t == int:
        return tuple(int(v) for v in s[1:-1].split(","))
    elif t == float:
        return tuple(float(v) for v in s[1:-1].split(","))


def _parse_coord(coord):
    name, idx = coord.split(".")
    return name, int(idx)


def _parse_flags(flags): return dict(tuple(f.split("=")) for f in flags)


def _parse_tensor_string(tensor_string):

    ranges = re.findall(r"\[([^\]]+)\]", tensor_string)
    assert len(ranges) in [0, 1], "Rule format not valid"

    ranges = ranges[0].split(",") if len(ranges) == 1 else ranges
    ranges = (_parse_coord(r) if "." in r else (r, 0) for r in ranges)

    name = re.findall(r"(.+)\[[^]]*\]", tensor_string)
    assert len(name) == 1, "Rule format not valid"
    name = name[0]

    return TensorData(name=name, ranges=tuple(ranges))


def _parse_equation_side(side_string):
    return tuple((_parse_tensor_string(tensor_string) for tensor_string in re.findall(r"[^]]+\[[^]]*\]", side_string)))


class ProgramLine(object):

    id_strings = []

    # each program line subclass creates a dictionary with the kwargs which will be passed to the function which
    # initializes the objects for each line
    @staticmethod
    def _get_parse_dicts(info, flags): raise NotImplemented

    @staticmethod
    def _add_to_program(program, p_dict): raise NotImplemented

    # the ParseDict object is passed to the handler class which will create the objects corresponding to each line
    @classmethod
    def add_to_program(cls, program, info, flags):
        for p_dict in cls._get_parse_dicts(info, flags):
            cls._add_to_program(program, p_dict)


class RangeProgramLine(ProgramLine):

    id_strings = ("range", )

    @staticmethod
    def _get_parse_dicts(info, flags):

        id_string = info[1]
        range_names = info[2:]

        if id_string == "coords":
            for i, name in enumerate(range_names):
                coords = _parse_tuple(name)
                name = coords[0].split(".")[0]
                for c in coords[1:]:
                    assert name == c.split(".")[0]
                flags["ndim"] = len(coords)
                range_names[i] = name

        for name in range_names:
            yield dict(id_string=id_string, name=name, **flags)

    @staticmethod
    def _add_to_program(program, p_dict):
        program.add_range(p_dict.pop("name"), p_dict)


class TensorProgramLine(ProgramLine):

    id_strings = ("tensor", )

    @staticmethod
    def _get_parse_dicts(info, flags):

        tensor_names = info[1:]

        shape = _parse_tuple(flags["shape"], int)

        for name in tensor_names:
            yield dict(name=name, shape=shape)

    @staticmethod
    def _add_to_program(program, p_dict):
        program.add_tensor(p_dict["name"], p_dict)


class OpProgramLine(ProgramLine):

    id_strings = ("loss", "einsum")

    @staticmethod
    def _get_parse_dicts(info, flags):

        id_string = info[0]

        lhs, rhs = info[1].split(":-")

        lhs_tensor_data = _parse_equation_side(lhs)
        rhs_tensor_data = _parse_equation_side(rhs)

        if "mult" in flags:
            flags["mult"] = float(flags["mult"])

        yield dict(id_string=id_string,
                   lhs_tensor_data=lhs_tensor_data,
                   rhs_tensor_data=rhs_tensor_data,
                   **flags)

    @staticmethod
    def _add_to_program(program, p_dict):
        program.add_op(p_dict["id_string"], p_dict)


class InputProgramLine(ProgramLine):

    id_strings = ("input", "constant", "weight")

    @staticmethod
    def _get_parse_dicts(info, flags):

        id_string = info[0]

        lhs = info[1].split(":-")[0]

        lhs_tensor_data = _parse_equation_side(lhs)

        if "std" in flags:
            flags["std"] = float(flags["std"])
        if "from" in flags:
            flags["from"] = float(flags["from"])
        if "to" in flags:
            flags["to"] = float(flags["to"])

        yield dict(id_string=id_string,
                   lhs_tensor_data=lhs_tensor_data,
                   **flags)

    @staticmethod
    def _add_to_program(program, p_dict):
        program.add_input(p_dict["id_string"], p_dict)