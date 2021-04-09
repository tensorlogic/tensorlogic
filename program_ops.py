from itertools import chain
import torch.nn
import torch.nn.functional

from subtensor import SubTensor
from stack_ops import LossOp, EinsumOp
from domain_op import Intersection
from domain import Domain


class ProgramNode(object):

    id_string = None

    def __init__(self, program, lhs_tensor_data, rhs_tensor_data):

        self.program = program

        self.lhs_tensor_data = lhs_tensor_data[0] if lhs_tensor_data else None
        self.rhs_tensor_data = rhs_tensor_data if rhs_tensor_data else None

        self.lhs_tensor = self.program.get_tensor(lhs_tensor_data[0]) if lhs_tensor_data else None
        self.rhs_tensors = tuple(self.program.get_tensor(tensor_data) for tensor_data in rhs_tensor_data) if rhs_tensor_data else None


# ############################## OPS ##############################


class ProgramOp(ProgramNode):

    id_string = None

    def __init__(self, program, lhs_tensor_data, rhs_tensor_data):
        super(ProgramOp, self).__init__(program, lhs_tensor_data, rhs_tensor_data)

    @staticmethod
    def init_op(program, id_string, lhs_tensor_data, rhs_tensor_data, **kwargs):

        if id_string == Loss.id_string:
            return Loss(program, kwargs.pop("name"), rhs_tensor_data, kwargs.pop("loss"), kwargs)
        elif id_string == Einsum.id_string:
            return Einsum(program, lhs_tensor_data, rhs_tensor_data, **kwargs)
        else:
            raise ValueError("Unknown id_string", id_string)

    def initialize(self): raise NotImplemented


class Einsum(ProgramOp):

    id_string = "einsum"

    def __init__(self, program, lhs_tensor_data, rhs_tensor_data, activation=None):

        super(Einsum, self).__init__(program, lhs_tensor_data, rhs_tensor_data)

        self.activation_func = self._activation(activation)
        self.einsum_string = self._einsum_string()

        self.lhs_domain = None

        self.rhs_domains = None
        self.rhs_domain_dicts = None

    @staticmethod
    def _activation(activation):

        if activation is None:
            return lambda x: x
        elif activation == 'relu':
            return torch.nn.ReLU()
        elif activation == "sigmoid":
            return torch.nn.ReLU()
        elif activation == "tanh":
            return torch.nn.Tanh()
        elif activation == "softmax":
            return torch.nn.functional.softmax
        else:
            raise ValueError

    def _einsum_string(self):

        range_map = {}

        ranges_out = (self.lhs_tensor_data.ranges, )
        ranges_in = tuple(tensor_data.ranges for tensor_data in self.rhs_tensor_data)
        max_letter = "a"

        for r in chain(*ranges_in, *ranges_out):
            if r not in range_map:
                range_map[r] = max_letter
                max_letter = chr(ord(max_letter) + 1)

        eq_parts = tuple(map(lambda ranges: "".join(range_map[r] for r in ranges), chain(ranges_in, ranges_out)))
        return ",".join(eq_parts[:-1]) + "->" + eq_parts[-1]

    def initialize(self):

        self.lhs_domain = self.program.initialize_domain(self.lhs_tensor_data)

        self.rhs_domains = []
        self.rhs_domain_dicts = []

        for rhs_tensor_data in self.rhs_tensor_data:
            rhs_domain = self.program.initialize_domain(rhs_tensor_data, assert_constant=True)
            rhs_domain_dict = dict(zip(rhs_tensor_data.ranges, rhs_domain.indexed_ranges))
            self.rhs_domains.append(rhs_domain)
            self.rhs_domain_dicts.append(rhs_domain_dict)

        self.program.rules[self.lhs_tensor_data.name].append(self)

    def get_stack_op(self, query_domain):

        lhs_domain = Intersection(self.lhs_domain, query_domain)
        if lhs_domain is None:
            return None

        lhs_domain_dict = dict(zip(self.lhs_tensor_data.ranges, lhs_domain.indexed_ranges))
        rhs_domains = []
        for rhs_domain_dict, rhs_tensor_data, rhs_tensor in \
                zip(self.rhs_domain_dicts, self.rhs_tensor_data, self.rhs_tensors):
            ranges = (lhs_domain_dict[r] if r in lhs_domain_dict else rhs_domain_dict[r] for r in rhs_tensor_data.ranges)
            rhs_domains.append(Domain.domain_from_ranges(ranges, rhs_tensor.shape))

        return EinsumOp(self.program, lhs_domain, rhs_domains, self.rhs_tensors, self.einsum_string, self.activation_func)


class Loss(ProgramOp):

    id_string = "loss"

    def __init__(self, program, name, rhs_tensor_data, loss, loss_kwargs):

        super(Loss, self).__init__(program, lhs_tensor_data=(), rhs_tensor_data=rhs_tensor_data)

        assert len(rhs_tensor_data) == 2

        self.loss_func = self._loss(loss, loss_kwargs)
        self.name = name

        self.op = None
        self.subtensor = None

    @staticmethod
    def _loss(loss, loss_kwargs):
        if loss == "mse":
            return torch.nn.MSELoss(**loss_kwargs)
        else:
            raise ValueError

    def initialize(self):

        self.program.losses[self.name] = self

        rhs_domain_1 = self.program.initialize_domain(self.rhs_tensor_data[0])
        rhs_domain_2 = self.program.initialize_domain(self.rhs_tensor_data[1])

        self.op = LossOp(self.program, self.rhs_tensors[0], rhs_domain_1, self.rhs_tensors[1], rhs_domain_2)
        self.subtensor = self.op.subtensor


# ############################## INPUTS ##############################


class ProgramInput(ProgramNode):

    id_string = None

    def __init__(self, program, lhs_tensor_data, rhs_tensor_data):

        super(ProgramInput, self).__init__(program, lhs_tensor_data, rhs_tensor_data)
        self.subtensor = None

    @staticmethod
    def init_input(program, id_string, lhs_tensor_data, **kwargs):

        if id_string == Input.id_string:
            return Input(program, lhs_tensor_data)
        elif id_string == Weight.id_string:
            return Weight(program, lhs_tensor_data, **kwargs)
        elif id_string == Constant.id_string:
            return Constant(program, lhs_tensor_data)
        else:
            raise ValueError

    def initialize(self, *args, **kwargs): raise NotImplemented


class Constant(ProgramInput):

    id_string = "constant"

    def __init__(self, program, lhs_tensor_data):
        super(Constant, self).__init__(program, lhs_tensor_data=lhs_tensor_data, rhs_tensor_data=())

    def initialize(self, constant_tensor):
        self.subtensor = SubTensor(self.program.initialize_domain(self.lhs_tensor_data, assert_constant=True))
        self.subtensor.set_data_from_input(constant_tensor)
        self.lhs_tensor.add_initial_input(self)


class Weight(ProgramInput):

    id_string = "weight"

    def __init__(self, program, lhs_tensor_data, initializer, initializer_kwargs):
        super(Weight, self).__init__(program, lhs_tensor_data=lhs_tensor_data, rhs_tensor_data=())
        self.initializer = initializer
        self.initializer_kwargs = initializer_kwargs

    def initialize(self):
        self.subtensor = SubTensor(self.program.initialize_domain(self.lhs_tensor_data, assert_constant=True))
        self.subtensor.set_data_from_weight_init(self.initializer, **self.initializer_kwargs)
        self.lhs_tensor.add_initial_input(self)


class Input(ProgramInput):

    id_string = "input"

    def __init__(self, program, lhs_tensor_data):
        super(Input, self).__init__(program, lhs_tensor_data=lhs_tensor_data, rhs_tensor_data=())

    def initialize(self):
        self.subtensor = SubTensor(self.program.initialize_domain(self.lhs_tensor_data))
        self.lhs_tensor.add_input(self)
