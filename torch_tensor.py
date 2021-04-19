import torch

from torch_utils import maybe_cast_to_tensor


class Tensor(object):

    """
    A Tensor object is used to hold the data pertaining to a tensor. As opposed to a SubTensor, the data runs
    over the whole range of the tensor.

    The Tensor object supports two basic operations, gathering/reading from the Tensor and placing it into
    a SubTensor and writing/updating the Tensor using the data in a SubTensor. The read/write operations are
    performed using the Index in the SubTensor. For now, only dense tensors are supported.
    """

    def __init__(self, shape, data=None):
        """
        :param shape: tuple[int], list[int]
            The shape of the tensor
        :param data: type of data depends on the type of the tensor
            The data that will be stored in the tensor. For a DenseTensor this would be the tensor data itself, while
            for a SparseTensor/MixedTensor it would contain the sparse coordinates along with the data stored at these
            coordinates
        """
        super(Tensor, self).__init__()

        self.shape = shape
        self.ndim = len(shape)
        self.data = data

    def reset(self, subtensor=None):
        """
        Reset the value of the Tensor, potentially setting it to some input SubTensor.
        :param subtensor: SubTensor(optional)
        :return:
        """
        raise NotImplemented

    def update_sum(self, subtensor):
        """
        Update the data stored in the Tensor using the data in the SubTensor. The data is aggregated using a sum.
        The SubTensor contains the information on which data to update through its Domain and the update locations
        are established by the SubTensor's Index.
        :param subtensor: SubTensor
        :return:
        """
        raise NotImplemented

    def gather(self, subtensor):
        """
        Gather data from the Tensor and place it into a SubTensor. The SubTensor contains the information on which data
        to gather through its Domain and the gathering locations are established by the SubTensor's Index.
        :param subtensor: SubTensor
        :return:
        """
        raise NotImplemented


class DenseTensor(Tensor):

    """
    A DenseTensor object holds the data in a torch.Tensor object. The data runs over the whole tensor and the
    dimensions of the tensor are the same as the dimensions of the data. The data is written into/read from a
    DenseTensor using torch's fancy indexing, with the creation of the indices being handled by the Index objects
    stored in the SubTensor objects.
    """

    def __init__(self, shape, data=None):
        data = maybe_cast_to_tensor(data, dtype=torch.float32)
        super(DenseTensor, self).__init__(shape, data)

    def reset(self, subtensor=None):
        if subtensor is None:
            self.data = torch.zeros(size=self.shape, dtype=torch.float32)
        elif subtensor.dummy:
            self.data = subtensor.data
        else:
            self.data = torch.zeros(size=self.shape, dtype=torch.float32)
            self.data[subtensor.index] = subtensor.data

    def update_sum(self, subtensor):
        if subtensor.dummy:
            self.data += subtensor.data
        else:
            self.data[subtensor.index] += subtensor.data

    def gather(self, subtensor):
        if subtensor.dummy:
            subtensor.data = self.data
        else:
            subtensor.data = self.data[subtensor.index]
