from torch_tensor import DenseTensor


class ProgramTensor(object):

    """
    A ProgramTensor is used to abstractly represent a tensor in the program. For "hidden" tensors in the program,
    the Program tensor will just hold the tensor information with no actual data, the data being handled by the
    StackOps. For the leaf tensors, the extensional data which is set by either Constant or Weight inputs during
    compilation or by Inputs during the run stage will be stored in the ProgramTensor and it will be here where it
    gets accessed by GatherOps.

    For the leaf tensors, we use the Domain of the ProgramInput which sets the initial tensor value to determine
    whether a query matches the Domain of the leaf tensor and can gather any extensional data from it.
    """

    def __init__(self, program, name, shape):
        """
        Create a ProgramTensor object.
        :param program: Program
            The program to which the ProgramTensor will be added.
        :param name: str
            The name of the tensor.
        :param shape: tuple[int]
            The shape of the tensor.
        """

        self.program = program
        self.name = name
        self.shape = shape

        self.tensor = DenseTensor(shape)

        self.inputs = []

        self.leaf = None
        self.set_from_input = None

        # the domain and empty properties only apply for leaf tensors. These are used to match queries
        self._domain = None
        self._empty = None

    @property
    def empty(self):
        if self.leaf:
            return self._empty
        raise ValueError("Only leaf tensors have this property")

    @property
    def domain(self):
        if self.set_from_input:
            return self._domain
        raise ValueError("Only tensors set with one input have this property")

    def add_input(self, inp):
        self.inputs.append(inp)

    def initialize(self):
        """
        Initialize the ProgramTensor. Set the flags which indicate whether the ProgramTensor is leaf tensor or not and
        whether it is set from one input only or not.
        :return:
        """
        self.set_from_input = len(self.inputs) == 1
        if self.set_from_input:
            self._domain = self.inputs[0].subtensor.domain
        self.leaf = len(self.inputs) > 0

    def reset(self):
        """
        Reset the value of a leaf ProgramTensor. This function is called by ResetLeafOp and should only be used for
        leaf tensors. The leaf tensor is set to the value of its only input if it has only one input. Otherwise it is
        initialized to zeros after which all of its inputs get aggregated. If all the inputs are empty, the
        leaf ProgramTensor will also become empty.
        :return:
        """
        if self.set_from_input:
            self.tensor.reset(self.inputs[0].subtensor)
            self._empty = self.inputs[0].subtensor.empty
        elif any(not inp.subtensor.empty for inp in self.inputs):
            self.tensor.reset()
            for inp in self.inputs:
                if not inp.subtensor.empty:
                    self.tensor.update_sum(inp.subtensor)
            self._empty = False
        else:
            self.tensor.data = None
            self._empty = True
