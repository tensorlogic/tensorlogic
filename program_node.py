class ProgramNode(object):

    """
    A ProgramNode class is designed to store persistent and constant information about operations being performed
    in the program. At an abstract level, a ProgramNode will operate on a set of right hand side ProgramTensors and
    update one left hand side  ProgramTensor, where either the lhs tensors or rhs tensors are optional(for example,
    ProgramInputs have no rhs tensors, while a Loss operation will have no lhs tensors.
    """
    id_string = None

    def __init__(self, program, lhs_tensor_data, rhs_tensor_data):
        """
        Creates a ProgramNode.
        :param program: Program
            The program to which the node will be added.
        :param lhs_tensor_data: tuple[TensorData]
            A tuple of TensorData objects specifying the information about the lhs tensor(should always have length 1).
            The TensorData is created during parsing and should not be inputted directly.
        :param rhs_tensor_data: tuple[TensorData]
            A tuple of TensorData objects specifying the information about the rhs tensors.
            The TensorData is created during parsing and should not be inputted directly.
        """
        self.program = program

        self.lhs_tensor_data = lhs_tensor_data[0] if lhs_tensor_data else None
        self.rhs_tensor_data = rhs_tensor_data if rhs_tensor_data else None

        self.lhs_tensor = self.program.get_tensor(lhs_tensor_data[0]) if lhs_tensor_data else None
        self.rhs_tensors = tuple(self.program.get_tensor(tensor_data) for tensor_data in rhs_tensor_data) if rhs_tensor_data else None
