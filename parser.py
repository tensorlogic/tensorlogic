import re

from structs import TensorData


def read_program_from_file(program_path):
    """
    Iterator for a txt file
    :param program_path: str
        Path to the file
    :return: generator[str]
        File lines
    """
    with open(program_path, 'r') as f:
        for line in f.readlines():
            yield line


def get_program_lines(program_line):
    """
    Perform some basic parsing of a program line by removing unneeded empty space. Split the line on the remaining empty
    space. Then split the items thus obtained into ones that have an equal sign(flags) and ones that don't(info)
    :param program_line: str
        Raw program line
    :return: tuple[list[str], list[str]]
        Parsed line info and parsed flags
    """
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

    """
    The main parsing function called by the program. The type of program line and the way to further parse it is
    established by the first item in info which is an unique identifier for each type of line to parse. The info and
    flags are then passed to this more specific parser which will further parse the info and flags and which will
    call the appropriate program function to add the created object to the program
    :param program: Program
        The program parsing the file
    :param info: list[str]
        Information parsed which is not a flag
    :param flags: list[str]
        Parsed flags
    :return:
    """
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
    s = s[1:-1]
    if len(s):
        if t is None:
            return tuple(v for v in s.split(","))
        elif t == int:
            return tuple(int(v) for v in s.split(","))
        elif t == float:
            return tuple(float(v) for v in s.split(","))
    else:
        return ()


def _parse_coord(coord):
    name, idx = coord.split(".")
    return name, int(idx)


def _parse_flags(flags): return dict(tuple(f.split("=")) for f in flags)


def _parse_tensor_string(tensor_string):
    """
    Parse a tensor string (e.g "T[x.0, x.1, y]") into a tensor data structure.
    :param tensor_string: str
    :return: TensorData
    """
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

    """
        A ProgramLine object is designed to parse an info list and a list of flags, create the objects specific to
        the line being parsed, and add them to the program via the _add_to_program function. Each subclass has a
        set of unique_id strings which will get matched to the first item in info to determine the appropriate
        subclass of ProgramLine to be used to parse the data.
    """
    id_strings = []

    @staticmethod
    def _get_parse_dicts(info, flags):
        """
        Parse the info and flags and obtain a dictionary with the data required to initialize the Program Objects.
        :param info: list[str]
            Information parsed which is not a flag
        :param flags: list[str]
            Parsed flags
        :return: generator[dict]
            A list of dictionary containing all the parsed information with keys being the arguments expected by the
            Program object that needs to be created for the current line. One dictionary per object that has to be
            created.
        """
        raise NotImplemented

    @staticmethod
    def _add_to_program(program, p_dict):
        """
        Use the parsed info to create the Program objects and to add them to the Program
        :param program: Program
            The Program to add to
        :param p_dict: dict
            The object initialization dictionary
        :return:
        """
        raise NotImplemented

    @classmethod
    def add_to_program(cls, program, info, flags):
        for p_dict in cls._get_parse_dicts(info, flags):
            cls._add_to_program(program, p_dict)


class RangeProgramLine(ProgramLine):

    """
    Parser for Range program lines
    """

    id_strings = ("range", )

    @staticmethod
    def _get_parse_dicts(info, flags):

        id_string = info[1]   # id string identifies the type of range
        range_names = info[2:]   # the remaining info represents all the names of the ranges being created on this line

        # return one dictionary for each range being created
        for name in range_names:
            yield dict(id_string=id_string, name=name, **flags)

    @staticmethod
    def _add_to_program(program, p_dict):
        program.add_range(p_dict.pop("name"), p_dict)


class TensorProgramLine(ProgramLine):

    """
    Parser for Tensor program lines
    """

    id_strings = ("tensor", )

    @staticmethod
    def _get_parse_dicts(info, flags):

        tensor_names = info[1:]  # the remaining info represents all the names of the tensors being created on this line
        shape = _parse_tuple(flags["shape"], int)

        # return one dictionary for each range being created
        for name in tensor_names:
            yield dict(name=name, shape=shape)

    @staticmethod
    def _add_to_program(program, p_dict):
        program.add_tensor(p_dict["name"], p_dict)


class OpProgramLine(ProgramLine):

    """
    Parser for ProgramOp program lines
    """

    id_strings = ("loss", "einsum")

    @staticmethod
    def _get_parse_dicts(info, flags):

        id_string = info[0]  # id string identifies the type of operation

        lhs, rhs = info[1].split(":-")  # split the equation the lhs and rhs

        # parse the sides of the equation into tuples of TensorData objects(one TensorData object for each tensor)
        lhs_tensor_data = _parse_equation_side(lhs)
        rhs_tensor_data = _parse_equation_side(rhs)

        if "mult" in flags:
            flags["mult"] = float(flags["mult"])

        # yield the initialization dictionary
        yield dict(id_string=id_string,
                   lhs_tensor_data=lhs_tensor_data,
                   rhs_tensor_data=rhs_tensor_data,
                   **flags)

    @staticmethod
    def _add_to_program(program, p_dict):
        program.add_op(p_dict["id_string"], p_dict)


class InputProgramLine(ProgramLine):

    """
    Parser for ProgramInput program lines
    """

    id_strings = ("input", "constant", "weight")

    @staticmethod
    def _get_parse_dicts(info, flags):

        id_string = info[0]  # id string identifies the type of input

        # inputs only have a lhs tensor. We parse that tensor into a TensorData object
        lhs = info[1].split(":-")[0]
        lhs_tensor_data = _parse_equation_side(lhs)

        if "mean" in flags:
            flags["mean"] = float(flags["mean"])
        if "std" in flags:
            flags["std"] = float(flags["std"])
        if "from" in flags:
            flags["from"] = float(flags["from"])
        if "to" in flags:
            flags["to"] = float(flags["to"])
        if "gain" in flags:
            flags["gain"] = float(flags["gain"])
        if "batch_dim" in flags:
            flags["batch_dim"] = int(flags["batch_dim"])

        # yield the initialization dictionary
        yield dict(id_string=id_string,
                   lhs_tensor_data=lhs_tensor_data,
                   **flags)

    @staticmethod
    def _add_to_program(program, p_dict):
        program.add_input(p_dict["id_string"], p_dict)