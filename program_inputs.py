from program_node import ProgramNode
from subtensor import SubTensor


class ProgramInput(ProgramNode):

    """
    A ProgramInput will be used to represent the inputs to a program: Weights, Constants, and Inputs(data which is
    fed to the program at each iteration). The ProgramInputs correspond to extensional data. The ProgramInputs
    will have a SubTensor object which stores the data and the locations of the tensor to which this data will get
    writen to. The data will be some form of dense tensor provided by the user, while the locations will be
    determined by Ranges forming the Domain of the ProgramInput's SubTensor. The ProgramInput is declared
    in the program specification, while the he data and the ranges are provided either at compilation
    time(Weights, Constants) or at run time(Inputs).
    """
    id_string = None

    def __init__(self, program, lhs_tensor_data):
        """
        Creates a ProgramInput object.
        :param program: Program
            The program to which the input will be added.
        :param lhs_tensor_data: tuple[TensorData]
            A tuple of TensorData objects specifying the information about the lhs tensor(should always have length 1).
        """
        super(ProgramInput, self).__init__(program, lhs_tensor_data, ())
        self.subtensor = None

    @staticmethod
    def init_input(program, id_string, lhs_tensor_data, **kwargs):
        """
        Creates a ProgramInput object from the parsed dictionary. The correct type of ProgramInput to create is
        determined by the id_string.
        :param program: Program
            The program to which the input will be added.
        :param id_string: str
            Unique string identifier for each ProgramInput type.
        :param lhs_tensor_data: tuple[TensorData]
            A tuple of TensorData objects specifying the information about the lhs tensor(should always have length 1).
        :param kwargs: dict
            Other parsed parameters which will be used by the ProgramInput. These are specific to each type of input.
        :return:
        """
        if id_string == Input.id_string:
            return Input(program, lhs_tensor_data, kwargs.pop("type", "data"))
        elif id_string == Weight.id_string:
            return Weight(program, lhs_tensor_data, kwargs.pop("initializer"), kwargs)
        elif id_string == Constant.id_string:
            return Constant(program, lhs_tensor_data, kwargs.pop("type", "data"))
        else:
            raise ValueError

    def initialize(self, *args, **kwargs):
        """
        Adds the ProgramInput as input to a leaf ProgramTensor and initializes other information that will be persistent
        thorough out program runs.
        This is run during compilation.
        :param args:
        :param kwargs:
        :return:
        """
        self.lhs_tensor.add_input(self)


class Constant(ProgramInput):

    """
    A Constant input is declared in the program specification and set during compilation.
    Its value does not change during the program runs.
    """
    id_string = "constant"

    def __init__(self, program, lhs_tensor_data, input_type):
        """
        Creates a Constant object.
        :param program: Program
            The program to which the input will be added.
        :param lhs_tensor_data: tuple[TensorData]
            A tuple of TensorData objects specifying the information about the lhs tensor(should always have length 1).
        :param input_type: str
            String identifier meant to distinguish between data and labels
        """
        super(Constant, self).__init__(program, lhs_tensor_data=lhs_tensor_data)
        self.cast_to_float = input_type == "data"

    def initialize(self, constant_tensor):
        """
        A Constant input is initialized using a constant value. The Constant's data and ranges are provided during
        compilation.
        :param constant_tensor: castable to a torch.Tensor
            Constant data to be inputted into the program.
        :return:
        """
        super(Constant, self).initialize()
        self.subtensor = SubTensor(self.program.initialize_domain(self.lhs_tensor_data, assert_constant=True))
        self.subtensor.set_data_from_input(constant_tensor, self.cast_to_float)


class Weight(ProgramInput):

    """
    A Weight input is declared in the program specification and set during compilation.
    The user specifies the initialization parameters when declaring the program. The initialize function creates
    the weight and adds it to the program during compilation. The weight will be updated during training
    by the program.
    """
    id_string = "weight"

    def __init__(self, program, lhs_tensor_data, initializer, initializer_kwargs):
        """
        Creates a Weight object
        :param program: Program
            The program to which the input will be added
        :param lhs_tensor_data: tuple[TensorData]
            A tuple of TensorData objects specifying the information about the lhs tensor(should always have length 1).
        :param initializer: str
            A unique string identifier specifying the type of initializer to use.
        :param initializer_kwargs: dict
            Other arguments use by the initializer.
        """
        super(Weight, self).__init__(program, lhs_tensor_data=lhs_tensor_data)
        self.initializer = initializer
        self.initializer_kwargs = initializer_kwargs

    def initialize(self):
        """
        A Weight input is initialized during compilation. The user specifies the initializer type and initialization
        parameters when declaring the program, and the ranges needed to create the Domain are provided during
        compilation.
        :return:
        """

        super(Weight, self).initialize()

        self.subtensor = SubTensor(self.program.initialize_domain(self.lhs_tensor_data, assert_constant=True))

        batch_dim = self.initializer_kwargs.get("batch_dim", None)

        self.subtensor.set_data_from_weight_init(self.initializer, **self.initializer_kwargs)
        self.program.weights.append(self.subtensor.data)

        if batch_dim is not None:
            self.subtensor.data = self.subtensor.data.broadcast_to(self.subtensor.shape)


class Input(ProgramInput):

    """
    An Input is declared in the program specification, but as opposed to the other inputs, it is set at every run.
    The Input is used to model extensional information corresponding to one data point.
    """
    id_string = "input"

    def __init__(self, program, lhs_tensor_data, input_type):
        """
        Creates an Input object
        :param program: Program
            The program to which the input will be added
        :param lhs_tensor_data: tuple[TensorData]
            A tuple of TensorData objects specifying the information about the lhs tensor(should always have length 1).
        :param input_type: str
            String identifier meant to distinguish between data and labels
        """
        super(Input, self).__init__(program, lhs_tensor_data=lhs_tensor_data)
        self.cast_to_float = input_type == "data"

    def initialize(self):
        """
        An Input object's Domain is initialized during compilation, although the Ranges in the domain need not be
        set the during compilation. No data is expected by the Input object during compilation
        :return:
        """
        super(Input, self).initialize()
        self.subtensor = SubTensor(self.program.initialize_domain(self.lhs_tensor_data))
