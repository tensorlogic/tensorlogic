from torch_tensor import DenseTensor


class ProgramTensor(object):

    def __init__(self, program, name, shape):

        self.program = program
        self.name = name
        self.shape = shape

        self.tensor = DenseTensor(shape)

        self.inputs = []
        self.initial_inputs = []
        self.initial_data = None

        self.domain = None
        self.empty = None

    def add_input(self, inp): self.inputs.append(inp)
    def add_initial_input(self, inp): self.initial_inputs.append(inp)

    def initialize(self):
        if self.initial_inputs:
            self.initial_data = self.tensor.get_zeros()
            for init_input in self.initial_inputs:
                assert not init_input.subtensor.empty, "Initialization subtensor is empty"
                self.initial_data[init_input.subtensor.index] += init_input.subtensor.data
                init_input.subtensor = None
            self.empty = False

    def reset(self):

        if not self.empty or any(not inp.subtensor.empty for inp in self.inputs):
            self.tensor.reset(self.initial_data)
            self.empty = False
        else:
            self.tensor.data = None
            self.empty = True

    def update_sum(self):

        for inp in self.inputs:
            if not inp.subtensor.empty:
                self.tensor.update_sum(inp.subtensor)
            self.domain = inp.subtensor.domain
            self.empty = self.domain.empty
