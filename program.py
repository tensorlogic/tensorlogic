import torch
from collections import defaultdict

from parser import read_program_from_file, get_program_lines, parse_program_line
from range import Range
from domain import Domain
from program_tensor import ProgramTensor
from program_ops import ProgramOp
from program_inputs import ProgramInput
from stack_ops import QueryOp, ResetLeafOp


class Program(object):
    """
    The Program class is responsible for parsing, compiling, and executing a TensorLogic program.

    The Program is initialized from a text file which contains the specification of the TensorLogic program.

    The Program parses the file, then it enters the compilation stage. After the program is compiled(program ops and
    tensors are initialized, constant tensors and ranges are inputted by the user, weights are initialized, an optimizer
    is potentially created), the program enters the run stage. Here the program will perform queries, compute losses,
    perform backprop and update the weights. The run stage is repeated during training and one backpropagation step is
    performed per each run.

    The Program computes the operations using a stack. Before each run stage, the stack gets initialized with the
    operations need to update the leaf tensors using the facts(inputs) given by the user during the run stage or
    given during compilation via constants or weights. Then loss ops are pushed onto the stack, followed by
    the queries ran by the user. The Program pops from the stack until empty.

    Each operation on the stack has two stages: expand and apply. In the expand stage, the StackOp pushes unto the stack
    subgoals that it needs evaluated before the apply stage(for example, an einsum StackOp will create Query ops for all
    tensors on the rhs before computing the einsum). In the apply stage, the StackOp uses the information in the
    subgoals to compute the operation and stores the result such that other StackOps can use it.

    When the stack is empty, backpropagation is applied(if backpropagtion is required by the user), and the queries and
    losses are returned.
    """
    def __init__(self, program_path=None):
        """
        Creates a TensorLogic Program from the program file.
        :param program_path: str
            Path to the file specifying the Program.
        """
        self.stack = []
        self.stack_init_ops = []  # ops that initialize the leaf tensors.

        # store the ranges by their names.
        # constant ranges have their values set during compilation, while variable ranges expect values during the run
        # stage(if no value is given, they are considered empty).
        self.ranges = {}
        self.constant_ranges, self.variable_ranges = {}, {}

        # store tensor information by their names. Leaf tensors are tensors which receive some form of input(facts).
        self.tensors = {}
        self.leaf_tensors = {}

        self.program_ops = defaultdict(list)
        self.program_inputs = defaultdict(list)

        # the key of the rules dictionary is the name of the tensor on the LHS of the rule.
        self.rules = defaultdict(list)

        self.losses = {}
        self.weights = []

        self.optimizer = None

        self._parse_program(program_path)

    def push_stack(self, op): self.stack.append(op)
    def pop_stack(self): return self.stack.pop()

    def get_tensor(self, name_or_tensor_data):
        return self.tensors[name_or_tensor_data] if isinstance(name_or_tensor_data, str) else self.tensors[name_or_tensor_data.name]

    def _get_range(self, name, assert_constant):
        """
        Get the range by its name. Ranges which have not been declared in the program file are considered to be unbound
        and will be treated as DummyRanges. When this is the case, we return None.
        :param name: str
            Name of the range to get
        :param assert_constant: bool
            Flag indicating whether we should check if the range returned is constant. For example, rules should always
            have constant ranges.
        :return: Range, None
        """
        r = self.ranges.get(name, None)
        if assert_constant and r is not None:
            assert name in self.constant_ranges, "Range " + str(name) + " has to be constant"
        return r

    def initialize_domain(self, tensor_data, assert_constant=False):
        """
        Initialize the domain. Use the TensorData object to retrieve the ranges declared in the program file(or
        DummyRanges if the range was not declared) used to index the tensor. Join them into a domain and return it
        :param tensor_data: TensorData
            The TensorData object which specifies the names of the ranges indexing a Tensor.
        :param assert_constant: bool
            Flag indicating whether we should check if the ranges in the domain are constant. For example, rules should
            always have constant domains.
        :return: Domain
        """
        ranges = ((self._get_range(name, assert_constant), c_idx) for name, c_idx in tensor_data.ranges)
        return Domain.domain_from_ranges(ranges, self.get_tensor(tensor_data).shape)

    # ############################## PARSE PROGRAM ##############################

    def _parse_program(self, program_path):
        # read each program line and then pass it to the parser which will create the required object and add it to
        # the program.
        program_lines = read_program_from_file(program_path)
        for line in get_program_lines(program_lines):
            parse_program_line(self, *line)

    # for each parsed line, add the parsed object to the appropriate dictionary.
    def add_range(self, name, init_dict): self.ranges[name] = Range.init_range(**init_dict)
    def add_tensor(self, name, init_dict): self.tensors[name] = ProgramTensor(self, **init_dict)
    def add_op(self, name, init_dict): self.program_ops[name].append(ProgramOp.init_op(self, **init_dict))
    def add_input(self, name, init_dict): self.program_inputs[name].append(ProgramInput.init_input(self, **init_dict))

    # ############################## COMPILE PROGRAM ##############################

    def _compile_optimizer(self, optimizer, optimizer_kwargs):
        """
        Create the Program's optimizer using the arguments specified in optimizer_kwargs
        :param optimizer: str
            String identifier of the optimizer type.
        :param optimizer_kwargs: dict
            Optimizer arguments. Check torch documentation for details.
        :return:
        """
        optimizer_kwargs = optimizer_kwargs or dict()
        if optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.weights, **optimizer_kwargs)
        elif optimizer == "sgd":
            self.optimizer = torch.optim.SGD(self.weights, **optimizer_kwargs)
        else:
            raise ValueError("Unknown optimizer type", self.optimizer, ". Available values: adam, sgd.")

    def compile(self,
                constant_ranges=None,
                constant_tensors=None,
                optimizer=None,
                optimizer_kwargs=None):

        """
        Compile the TensorLogic Program.
        :param constant_ranges: dict
            A dictionary with the range names as keys and the values of the ranges as values.
        :param constant_tensors: dict
            A dictionary with the tensor names as keys and the values of the tensors as values. The shape of the values
            must match the shape of the index created by the ranges indexing the tensor. See numpy/torch fancy indexing
            rules for more details.
        :param optimizer: str
            String identifier of the optimizer type.
        :param optimizer_kwargs: dict
            Optimizer arguments. Check torch documentation for details.
        :return:
        """
        constant_ranges = constant_ranges or dict()
        constant_tensors = constant_tensors or dict()

        # for each range value given during compilation set the value of the range and store the range as a constant
        for k, v in constant_ranges.items():
            self.ranges[k].value = v
            self.constant_ranges[k] = self.ranges[k]

        # for all the other ranges declared in the program which did not have the values set during compilation, add
        # them to the variable ranges dict.
        for k, v in self.ranges.items():
            if k not in self.constant_ranges:
                self.variable_ranges[k] = self.ranges[k]

        # initialize the program's inputs.
        for program_input in self.program_inputs["constant"]:
            name = program_input.lhs_tensor.name
            try:
                program_input.initialize(constant_tensors.get(name, None))
            except KeyError:
                raise KeyError("Constant input", name, "needs to be set during compilation")
        for program_input in self.program_inputs["weight"]:
            program_input.initialize()
        for program_input in self.program_inputs["input"]:
            program_input.initialize()

        # initialize the tensors
        for tensor in self.tensors.values():
            tensor.initialize()
            # if the tensor is a leaf, add an initialization operation to the stack that will update the leaf tensor
            # at the beginning of each run stage.
            if tensor.leaf:
                self.leaf_tensors[tensor.name] = tensor
                self.stack_init_ops.append(ResetLeafOp(self, tensor))

        # initialize the program ops
        for program_op in self.program_ops.values():
            for op in program_op:
                op.initialize()

        # create the optimizer
        if optimizer is not None:
            self._compile_optimizer(optimizer, optimizer_kwargs)

    # ############################## RUN PROGRAM ##############################

    def run(self, queries=None, losses=None, input_tensors=None, input_ranges=None, backprop=False):

        """
        Run the TensorLogic Program.
        :param queries: tuple[QueryData], list[QueryData]
            User defined queries to perform. See the QueryData structure to see how to create a query.
        :param losses: tuple[str], list[str]
            The list of losses to compute during this run stage
        :param input_ranges: dict
            A dictionary with the range names as keys and the values of the ranges as values.
        :param input_tensors: dict
            A dictionary with the tensor names as keys and the values of the tensors as values. The shape of the values
            must match the shape of the index created by the ranges indexing the tensor. See numpy/torch fancy indexing
            rules for more details.
        :param backprop: bool
            Whether backprop should be ran.
        :return: tuple[torch.Tensor], tuple[torch.Tensor]
            A tuple with the computed losses(if the desired loss could not be computed then a None will be present)
            A tuple with the values computed by each query.
        """

        queries = queries or ()
        losses = losses or ()

        input_tensors = input_tensors or dict()
        input_ranges = input_ranges or dict()

        # if we are performing backprop, reset the optimizer.
        if backprop:
            self.optimizer.zero_grad()

        # reset the variable ranges. This way if a variable range does not receive an input, then it is considered
        # empty.
        for v in self.variable_ranges.values():
            v.reset()

        # set the values of the variable ranges from the input ranges
        for k, v in input_ranges.items():
            try:
                self.variable_ranges[k].value = v
            except KeyError:
                raise KeyError("Range", k, "is undeclared or constant")

        # set the values of the leaf tensors from the input tensors
        for v in self.program_inputs['input']:
            v.subtensor.set_data_from_input(input_tensors.get(v.lhs_tensor.name), v.cast_to_float)

        # create the query and the loss ops
        query_ops = [QueryOp.from_query_data(self, query_data) for query_data in queries]
        loss_ops = [stack_op for stack_op in (self.losses[loss].stack_op() for loss in losses) if stack_op is not None]

        # initialize the stack
        self.stack = query_ops + loss_ops + self.stack_init_ops

        # while the stack is not empty
        while self.stack:
            # pop
            stack_op = self.pop_stack()

            # try to apply the op, in which case the op is permanently removed from the stack.
            if stack_op.can_apply:
                stack_op.apply()
            else:
                # otherwise push the op back unto the stack and try to expand the op.
                self.push_stack(stack_op)
                stack_op.can_apply = True
                # if the expand fails, pop again.
                if not stack_op.expand():
                    self.pop_stack()

        # if we are performing backprop, then update the weights.
        if backprop:
            for loss in (loss.data for loss in loss_ops):
                if loss is not None:
                    loss.backward()
                    self.optimizer.step()

        return tuple(q.subtensor.data for q in query_ops), tuple(loss.data for loss in loss_ops)

