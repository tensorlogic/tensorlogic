from itertools import chain
import torch.nn
import torch.nn.functional

from program_node import ProgramNode
from stack_ops import LossOp, EinsumOp
from domain_op import Intersection
from domain import Domain


class ProgramOp(ProgramNode):
    """
    A ProgramOp object will be used to represent the program rules and other program operations such as losses.
    The ProgramOps generally correspond to intentional data. The user declares the ops when specifying the program.
    The Program will store the ops and match them to any queries performed by the user or by other ops.
    The ProgramOp will store persistent data which is consistent through all runs, but for each time the op is
    matched by the program during the run stage, the ProgramOp will create a StackOp which will be pushed onto the
    Program stack and which will perform the actual computation.
    """
    id_string = None

    def __init__(self, program, lhs_tensor_data, rhs_tensor_data):
        """
        Creates a ProgramOp.
        :param program: Program
            The program to which the node will be added
        :param lhs_tensor_data: tuple[TensorData]
            A tuple of TensorData objects specifying the information about the lhs tensor(should always have length 1).
        :param rhs_tensor_data: tuple[TensorData]
            A tuple of TensorData objects specifying the information about the rhs tensors.
        """
        super(ProgramOp, self).__init__(program, lhs_tensor_data, rhs_tensor_data)

    @staticmethod
    def init_op(program, id_string, lhs_tensor_data, rhs_tensor_data, **kwargs):
        """
        Creates a ProgramOp object from the parsed dictionary. The correct type of ProgramOp to create is
        determined by the id_string.
        :param program: Program
            The program to which the op will be added
        :param id_string: str
            Unique string identifier for each ProgramOp type
        :param lhs_tensor_data: tuple[TensorData]
            A tuple of TensorData objects specifying the information about the lhs tensor(should always have length 1).
        :param rhs_tensor_data: tuple[TensorData]
            A tuple of TensorData objects specifying the information about the rhs tensors.
        :param kwargs: dict
            Other parsed parameters which will be used by the ProgramInput. These are specific to each type of input.
        :return:
        """
        if id_string == Loss.id_string:
            return Loss(program, kwargs.pop("name"), rhs_tensor_data, kwargs.pop("loss"), kwargs)
        elif id_string == Einsum.id_string:
            return Einsum(program, lhs_tensor_data, rhs_tensor_data, **kwargs)
        else:
            raise ValueError("Unknown id_string", id_string)

    def initialize(self):
        """
        Creates data structures and precomputes information which persists through all runs of the program. Adds the
        op to the program. This is run during compilation.
        :return:
        """
        raise NotImplemented

    def stack_op(self, *args, **kwargs):
        """
        The ProgramOp will create a StackOp which will be pushed onto the Program stack and which will perform the
        actual operation. The StackOp will be created using a combination of the persistent information stored in the
        ProgramOp and information provided by the program during each run(for example, the query information). If no
        operation is need(for example, a rule does not match a query), the function returns None.
        :param args:
        :param kwargs:
        :return: StackOp, None
            The StackOp to be pushed on the Program's stack or None if the operation is not needed.
        """
        raise NotImplemented


class Einsum(ProgramOp):

    """

    Implements the Einsum operation. Each Einsum operation corresponds to an intensional program rule.
    The operation is stored by the program and gets matched to any query asking about the lhs tensor of the rule.
    In addition to matching the query to the tensor on the lhs, we also match on the Domain of the query and the
    Domain of the lhs of the Einsum rule. Substitution is performed by intersecting the two domains and
    by pushing queries with the rhs Ranges replaced by those in the intersection where necessary onto the stack.

    For now, the Einsum operation only implements dense einsum. Note however that only the substituted parts of
    the rhs tensors are used in the einsum, so the operation only pulls parts of the rhs tensors relevant to computing
    the queried domain on the lhs.

    Each Einsum is declared during the program specification.

    """
    id_string = "einsum"

    def __init__(self, program, lhs_tensor_data, rhs_tensor_data, activation=None):

        """
        Creates an Einsum operation.
        :param program: Program
            The program to which the node will be added.
        :param lhs_tensor_data: tuple[TensorData]
            A tuple of TensorData objects specifying the information about the lhs tensor(should always have length 1).
        :param rhs_tensor_data: tuple[TensorData]
            A tuple of TensorData objects specifying the information about the rhs tensors.
        :param activation: str
            String identifier specifying the activation to be applied after the Einsum is performed.
        """
        super(Einsum, self).__init__(program, lhs_tensor_data, rhs_tensor_data)

        # get the activation function from the string id
        self.activation_func = self._activation_function(activation)
        # get the einsum string from the TensorData information. This einsum string could be modified during the
        # computation.
        self.einsum_string = self._einsum_string()

        self.lhs_domain = None

        self.rhs_domains = None
        self.rhs_domain_dicts = None

    @staticmethod
    def _activation_function(activation):

        if activation is None:
            return lambda x: x
        elif activation == 'relu':
            return torch.nn.ReLU()
        elif activation == "sigmoid":
            return torch.nn.Sigmoid()
        elif activation == "tanh":
            return torch.nn.Tanh()
        elif activation == "elu":
            return torch.nn.ELU()
        elif activation == "softmax":
            return torch.nn.Softmax(dim=-1)
        elif activation == "log_softmax":
            return torch.nn.LogSoftmax(dim=-1)
        else:
            raise ValueError("Unknown activation type", activation,
                             ". Available values: relu, sigmoid, tanh, elu, softmax, log_softmax")

    def _einsum_string(self):
        """
        Create the einsum string given the rule's rhs and lhs data. This string can be modified during the actual
        computation due to some SubTensors having integer Ranges causing dimensions to disappear or to data being
        permuted inside the SubTensors
        :return: str
        """
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
        """
        Creates data structures and precomputes information which persists through all runs of the program. Adds the
        op to the program. It creates Domains for both the rhs and lhs, Domains whose Ranges have to be constant.
        Furthermore, it creates maps that map the TensorData range names to Ranges of the rhs Domains. These maps
        will be used when performing substitution.
        :return:
        """
        self.lhs_domain = self.program.initialize_domain(self.lhs_tensor_data, assert_constant=True)

        self.rhs_domains = []
        self.rhs_domain_dicts = []

        for rhs_tensor_data in self.rhs_tensor_data:

            rhs_domain = self.program.initialize_domain(rhs_tensor_data, assert_constant=True)
            rhs_domain_dict = dict(zip(rhs_tensor_data.ranges, rhs_domain.indexed_ranges))

            self.rhs_domains.append(rhs_domain)
            self.rhs_domain_dicts.append(rhs_domain_dict)

        self.program.rules[self.lhs_tensor_data.name].append(self)

    def stack_op(self, query_domain):
        """
        Creates the Einsum StackOp. After the rule gets matched by the Program using the lhs tensor name, we check
        if the Domain over which the rule is applicable(the lhs_domain) matches the Domain being queried. We check this
        via intersection of the Domains. If the intersection is empty, we return None. If that is not the case, we
        take the result of the intersection and substitute in the rhs Domains. We the return the Einsum StackOp.
        :param query_domain: Domain
            The Domain of the query
        :return: StackOp
            The einsum StackOp created for this rule and a given query domain.
        """

        # check intersection and return None if the intersection is empty.
        lhs_domain = Intersection(self.lhs_domain, query_domain)
        if lhs_domain is None:
            return None

        # map the tensor range names to the Range objects in the lhs_domain.
        lhs_domain_dict = dict(zip(self.lhs_tensor_data.ranges, lhs_domain.indexed_ranges))

        # perform substitution
        rhs_domains = []
        for rhs_domain_dict, rhs_tensor_data, rhs_tensor in zip(self.rhs_domain_dicts, self.rhs_tensor_data, self.rhs_tensors):
            # replace all ranges from the initial rhs_domains with ranges from lhs_domain if the range is present
            # on the lhs.
            ranges = (lhs_domain_dict[r] if r in lhs_domain_dict else rhs_domain_dict[r] for r in rhs_tensor_data.ranges)
            rhs_domains.append(Domain.domain_from_ranges(ranges, rhs_tensor.shape))

        return EinsumOp(self.program, lhs_domain, rhs_domains, self.rhs_tensors, self.einsum_string, self.activation_func)


class Loss(ProgramOp):

    """
    Implements the Loss operation. The Loss operation is performed by selecting two SubTensors which must have
    the data dimensions required by the torch loss declared by the user during program creation. The torch loss
    will be computed on the two queried SubTensors at each run. That is, at each run we push onto the stack queries
    for both terms of the loss(if they are not empty) and compute the loss.
    """

    id_string = "loss"

    def __init__(self, program, name, rhs_tensor_data, loss, loss_kwargs):

        """
        Creates an Einsum operation.
        :param program: Program
            The program to which the node will be added.
        :param name: str
            Unique string name to identify the loss by.
        :param rhs_tensor_data: tuple[TensorData]
            A tuple of TensorData objects specifying the information about the rhs tensors.
        :param loss: str
            Unique string identifier for the torch loss to perform.
        :param loss_kwargs: dict
            Other parsed loss parameters.
        """
        super(Loss, self).__init__(program, lhs_tensor_data=(), rhs_tensor_data=rhs_tensor_data)

        assert len(rhs_tensor_data) == 2, "Loss can only have two rhs tensors"

        self.loss_func = self._loss_func(loss, loss_kwargs)
        self.name = name

        self.rhs_domain_1 = None
        self.rhs_domain_2 = None

    @staticmethod
    def _loss_func(loss, loss_kwargs):
        if loss == "mse":
            return torch.nn.MSELoss(**loss_kwargs)
        elif loss == "l1":
            return torch.nn.L1Loss(**loss_kwargs)
        elif loss == "nll":
            return torch.nn.NLLLoss(**loss_kwargs)
        else:
            raise ValueError("Unknown loss type", loss, ". Available values: mse, l1, nll")

    def initialize(self):
        """
        Creates data structures and precomputes information which persists through all runs of the program. Adds the
        loss to the program. It creates Domains for the rhs SubTensors which will be used to query the data
        needed to perform the loss.
        :return:
        """
        self.program.losses[self.name] = self
        self.rhs_domain_1 = self.program.initialize_domain(self.rhs_tensor_data[0])
        self.rhs_domain_2 = self.program.initialize_domain(self.rhs_tensor_data[1])

    def stack_op(self):
        """
        Creates a new Loss StackOp.
        :return: StackOp
            The loss StackOp.
        """
        if self.rhs_domain_1.empty or self.rhs_domain_2.empty:
            return None
        return LossOp(self.program, self.rhs_tensors[0], self.rhs_tensors[1], self.rhs_domain_1, self.rhs_domain_2,
                      self.loss_func)
