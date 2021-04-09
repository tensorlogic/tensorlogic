import torch

from torch_utils import maybe_cast_to_tensor


def _assert_same_shape(subtensors):
    st0_shape = subtensors[0].shape
    assert all(st0_shape == st.shape for st in subtensors[1:])


class Tensor(object):

    def __init__(self, shape, data=None):

        super(Tensor, self).__init__()

        self.shape = shape
        self.ndim = len(shape)

        self.data = data

    @staticmethod
    def init_from_subtensors(*subtensors):
        _assert_same_shape(subtensors)
        return DenseTensor(subtensors[0].shape)

    def reset(self, data): raise NotImplemented
    def get_zeros(self): raise NotImplemented
    def update_sum(self, subtensor): raise NotImplemented
    def gather(self, subtensor): raise NotImplemented


class DenseTensor(Tensor):

    def __init__(self, shape, data=None):
        data = maybe_cast_to_tensor(data, dtype=torch.float32)
        super(DenseTensor, self).__init__(shape, data)

    def reset(self, data=None):
        self.data = self.get_zeros() if data is None else maybe_cast_to_tensor(data, dtype=torch.float32)

    def get_zeros(self): return torch.zeros(size=self.shape, dtype=torch.float32)

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
