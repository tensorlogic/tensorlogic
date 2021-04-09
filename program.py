from collections import defaultdict

from parser import read_program_from_file, get_program_lines, parse_program_line
from range import Range
from domain import Domain
from program_tensor import ProgramTensor
from program_ops import ProgramOp, ProgramInput
from stack_ops import QueryOp, UpdateLeafOp


class Program(object):

    def __init__(self, program_path=None):

        self.stack = []
        self.stack_init_ops = []

        self.ranges = {}
        self.constant_ranges, self.variable_ranges = {}, {}
        self.tensors = {}
        self.leaf_tensors = {}

        self.program_ops = defaultdict(list)
        self.program_inputs = defaultdict(list)

        self.rules = defaultdict(list)
        self.losses = {}
        self.inputs = {}

        self._parse_program(program_path)

    def push_stack(self, op): self.stack.append(op)
    def pop_stack(self): return self.stack.pop()

    def get_range(self, name, assert_constant):
        r = self.ranges.get(name, None)
        if assert_constant and r is not None:
            assert name in self.constant_ranges, "Range " + str(name) + " has to be constant"
        return r

    def get_tensor(self, name_or_tensor_data):
        return self.tensors[name_or_tensor_data] if isinstance(name_or_tensor_data, str) else self.tensors[name_or_tensor_data.name]

    def get_tensor_shape(self, name_or_tensor_data):
        return self.get_tensor(name_or_tensor_data).shape

    def initialize_domain(self, tensor_data, assert_constant=False):
        ranges = ((self.get_range(name, assert_constant), c_idx) for name, c_idx in tensor_data.ranges)
        return Domain.domain_from_ranges(ranges, self.get_tensor_shape(tensor_data))

    # ############################## PARSE PROGRAM ##############################

    def _parse_program(self, program_path):
        program_lines = read_program_from_file(program_path)
        for line in get_program_lines(program_lines):
            parse_program_line(self, *line)

    def add_range(self, name, init_dict): self.ranges[name] = Range.init_range(**init_dict)
    def add_tensor(self, name, init_dict): self.tensors[name] = ProgramTensor(self, **init_dict)
    def add_op(self, name, init_dict): self.program_ops[name].append(ProgramOp.init_op(self, **init_dict))
    def add_input(self, name, init_dict): self.program_inputs[name].append(ProgramInput.init_input(self, **init_dict))

    # ############################## COMPILE PROGRAM ##############################

    def compile(self,
                constant_ranges=None,
                constant_tensors=None):

        constant_ranges = constant_ranges or dict()
        constant_tensors = constant_tensors or dict()

        for k, v in constant_ranges.items():
            self.ranges[k].value = v
            self.constant_ranges[k] = self.ranges[k]
        for k, v in self.ranges.items():
            if k not in self.constant_ranges:
                self.variable_ranges[k] = self.ranges[k]

        for op_type, program_op in self.program_ops.items():
            for op in program_op:
                op.initialize()

        for program_input in self.program_inputs["constant"]:
            name = program_input.lhs_tensor.name
            try:
                program_input.initialize(constant_tensors.get(name, None))
            except KeyError:
                raise KeyError("Constant input", name, "needs to be set during compilation")
            self.leaf_tensors[name] = self.tensors[name]

        for program_input in self.program_inputs["weight"]:
            program_input.initialize()
            name = program_input.lhs_tensor.name
            self.leaf_tensors[name] = self.tensors[name]

        for program_input in self.program_inputs["input"]:
            program_input.initialize()
            name = program_input.lhs_tensor.name
            self.leaf_tensors[name] = self.tensors[name]

        for tensor in self.tensors.values():
            tensor.initialize()
            if tensor.name in self.leaf_tensors:
                self.stack_init_ops.append(UpdateLeafOp(self, tensor))

        for loss_op in self.losses.values():
            self.stack_init_ops.append(loss_op.op)

    def run(self, queries=None, input_tensors=None, input_ranges=None):

        queries = queries or ()
        input_tensors = input_tensors or dict()
        input_ranges = input_ranges or dict()

        for v in self.variable_ranges.values():
            v.reset()

        for k, v in input_ranges.items():
            try:
                self.variable_ranges[k].value = v
            except KeyError:
                raise KeyError("Range", k, "is undeclared or constant")

        for v in self.program_inputs['input']:
            v.subtensor.set_data_from_input(input_tensors.get(v.lhs_tensor.name, None))

        query_ops = [QueryOp.from_query_data(self, query_data, True) for query_data in queries]
        self.stack = query_ops + self.stack_init_ops

        while self.stack:
            stack_op = self.pop_stack()
            if stack_op.can_apply:
                stack_op.apply()
            else:
                self.push_stack(stack_op)
                stack_op.can_apply = True
                if not stack_op.expand():
                    self.pop_stack()

        return [q.subtensor.data for q in query_ops]


if __name__ == "__main__":

    import sparse as sp
    import numpy as np
    from structs import QueryData

    p = Program("friends.txt")

    friends = (sp.random(density=0.15, shape=(50, 50)) > 0).astype(np.float32)
    stressed = (sp.random(density=0.45, shape=(50,)) > 0).astype(np.float32)
    smokes = (sp.random(density=0.2, shape=(50,)) > 0).astype(np.float32)
    drinks = (sp.random(density=0.25, shape=(50,)) > 0).astype(np.float32)

    friends_init = friends.coords
    stressed_init = stressed.coords.flatten()
    smokes_init = smokes.coords.flatten()
    drinks_init = drinks.coords.flatten()

    friends_2 = friends @ friends
    stressed_2 = sp.dot(friends_2, stressed) + stressed
    smokes_2 = sp.dot(friends_2, smokes) + smokes + stressed_2
    drinks_2 = sp.dot(friends_2, drinks) + drinks + stressed_2
    cancer = smokes_2 + drinks_2 + stressed_2

    cancer_friends = friends_2 * cancer.reshape((50, 1)) * cancer.reshape((1, 50))
    drinks_2_friends = friends_2 * drinks_2.reshape((50, 1)) * drinks_2.reshape((1, 50))
    smokes_2_friends = friends_2 * smokes_2.reshape((50, 1)) * smokes_2.reshape((1, 50))
    stressed_2_friends = friends_2 * stressed_2.reshape((50, 1)) * stressed_2.reshape((1, 50))

    p.compile(constant_ranges=dict(init_friends=friends_init, init_stressed=stressed_init, init_smokers=smokes_init,
                                   init_drinkers=drinks_init),
              constant_tensors=dict(Drinks_init=True, Smokes_init=True, Friends_init=True, Stressed_init=True))

    # for i in range(0, 50):
    #
    #     result = p.run(queries=[QueryData(tensor="Friends_init", domain_tuple=("i", "x"), domain_vals=dict(i=i, x=None)),
    #                             QueryData(tensor="Stressed_init", domain_tuple=("i", ), domain_vals=dict(i=i)),
    #                             QueryData(tensor="Smokes_init", domain_tuple=("i", ), domain_vals=dict(i=i)),
    #                             QueryData(tensor="Drinks_init", domain_tuple=("i", ), domain_vals=dict(i=i)),
    #                             QueryData(tensor="Stressed", domain_tuple=("i", ), domain_vals=dict(i=i)),
    #                             QueryData(tensor="Smokes", domain_tuple=("i", ), domain_vals=dict(i=i)),
    #                             QueryData(tensor="Drinks", domain_tuple=("i", ), domain_vals=dict(i=i)),
    #                             QueryData(tensor="Cancer", domain_tuple=("i", ), domain_vals=dict(i=i))])
    #
    #     assert np.allclose(result[0], friends[i].todense())
    #     assert np.allclose(result[1], stressed[i])
    #     assert np.allclose(result[2], smokes[i])
    #     assert np.allclose(result[3], drinks[i])
    #     assert np.allclose(result[4], stressed_2[i])
    #     assert np.allclose(result[5], smokes_2[i])
    #     assert np.allclose(result[6], drinks_2[i])
    #     assert np.allclose(result[7], cancer[i])
    #
    # for i, j in zip(np.random.randint(0, 50, (200, )), np.random.randint(0, 50, (200, ))):
    #
    #     result = p.run(queries=[QueryData(tensor="Friends", domain_tuple=("i", "j"), domain_vals=dict(i=i, j=j)),
    #                             QueryData(tensor="Friends", domain_tuple=("i", "x"), domain_vals=dict(i=i, x=None)),
    #                             QueryData(tensor="Friends", domain_tuple=("x", "j"), domain_vals=dict(j=j, x=None)),
    #                             QueryData(tensor="CancerFriends", domain_tuple=("i", "j"), domain_vals=dict(i=i, j=j)),
    #                             QueryData(tensor="CancerFriends", domain_tuple=("i", "x"), domain_vals=dict(i=i, x=None)),
    #                             QueryData(tensor="CancerFriends", domain_tuple=("x", "j"), domain_vals=dict(j=j, x=None)),
    #                             QueryData(tensor="DrinksFriends", domain_tuple=("i", "j"), domain_vals=dict(i=i, j=j)),
    #                             QueryData(tensor="DrinksFriends", domain_tuple=("i", "x"), domain_vals=dict(i=i, x=None)),
    #                             QueryData(tensor="DrinksFriends", domain_tuple=("x", "j"), domain_vals=dict(j=j, x=None)),
    #                             QueryData(tensor="SmokesFriends", domain_tuple=("i", "j"), domain_vals=dict(i=i, j=j)),
    #                             QueryData(tensor="SmokesFriends", domain_tuple=("i", "x"), domain_vals=dict(i=i, x=None)),
    #                             QueryData(tensor="SmokesFriends", domain_tuple=("x", "j"), domain_vals=dict(j=j, x=None)),
    #                             QueryData(tensor="StressedFriends", domain_tuple=("i", "j"), domain_vals=dict(i=i, j=j)),
    #                             QueryData(tensor="StressedFriends", domain_tuple=("i", "x"), domain_vals=dict(i=i, x=None)),
    #                             QueryData(tensor="StressedFriends", domain_tuple=("x", "j"), domain_vals=dict(j=j, x=None))])
    #
    #     assert np.allclose(result[0], friends_2[i, j])
    #     assert np.allclose(result[1], friends_2[i, :].todense())
    #     assert np.allclose(result[2], friends_2[:, j].todense())
    #     assert np.allclose(result[3], cancer_friends[i, j])
    #     assert np.allclose(result[4], cancer_friends[i, :].todense())
    #     assert np.allclose(result[5], cancer_friends[:, j].todense())
    #     assert np.allclose(result[6], drinks_2_friends[i, j])
    #     assert np.allclose(result[7], drinks_2_friends[i, :].todense())
    #     assert np.allclose(result[8], drinks_2_friends[:, j].todense())
    #     assert np.allclose(result[9], smokes_2_friends[i, j])
    #     assert np.allclose(result[10], smokes_2_friends[i, :].todense())
    #     assert np.allclose(result[11], smokes_2_friends[:, j].todense())
    #     assert np.allclose(result[12], stressed_2_friends[i, j])
    #     assert np.allclose(result[13], stressed_2_friends[i, :].todense())
    #     assert np.allclose(result[14], stressed_2_friends[:, j].todense())

    coords = np.stack([np.random.randint(0, 50, (200,)), np.random.randint(0, 50, (200,))])
    result = p.run(queries=[QueryData(tensor="CancerFriends", domain_tuple=("x.0", "x.1"), domain_vals=dict(x=coords))])
    assert np.allclose(result[0], cancer_friends[tuple(coords)].todense())
    result = p.run(queries=[QueryData(tensor="SmokesFriends", domain_tuple=("x.0", "s"),
                                      domain_vals=dict(x=coords[0:1], s=slice(0, 20)))])
    assert np.allclose(result[0], smokes_2_friends[coords[0], slice(0, 20)].todense())
