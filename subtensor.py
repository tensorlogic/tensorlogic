import torch

from torch_utils import maybe_cast_to_tensor
from misc import any_pred_true, indices_pred_true, reduce_concat
from index import DenseIndex


class SubTensor(object):

    def __init__(self, domain, data=None):

        self.domain = domain

        self.indexed_ranges = self.domain.indexed_ranges
        self.slices = self.domain.slices
        self.coords = self.domain.coords
        self.slice_dims = self.domain.slice_dims
        self.coord_dims = self.domain.coord_dims

        self.ranges_dims_map = self.domain.range_dims_map
        self.slice_dims_map = self.domain.slice_dims_map
        self.coord_dims_map = self.domain.coord_dims_map

        self.ndim = self.domain.ndim
        self.shape = self.domain.shape
        self.dummy = self.domain.dummy

        self.data = data

        self.data_dims = None
        self.data_ranges = None
        self.data_ranges_map = None
        self.data_ndim = None

        self._index = None
        self.set_dims_and_index()

    @property
    def empty(self): return self.domain.empty or self.data is None
    @property
    def index(self): return self._index.index()

    def reset(self): self.data = self.get_zeros()
    def get_zeros(self): return torch.zeros(self.get_data_shape(), dtype=torch.float32)
    def get_tensor_zeros(self): return torch.zeros(self.shape, dtype=torch.float32)

    def get_constant(self, indices=None): return self.domain.get_constant(indices)
    def get_all_constant(self, indices=None): return self.domain.get_constant(indices)
    def get_indexed_ranges(self, indices=None): return self.domain.get_indexed_ranges(indices)
    def get_ranges(self, indices=None): return self.domain.get_ranges(indices)
    def get_shape(self, indices=None): return self.domain.get_shape(indices)
    def get_range_types(self, indices=None): return self.domain.get_range_types(indices)
    def get_data_shape(self):
        return reduce_concat(*(r.data_shape for r in self.data_ranges))

    def _sort_dims(self):

        slice_index_dims = tuple(indices_pred_true(self.domain.get_ranges(), lambda x: x.is_dense))
        coord_index_dims = tuple(indices_pred_true(self.domain.get_ranges(), lambda x: x.is_sparse))

        if len(coord_index_dims) > 1:
            first, last = coord_index_dims[0], coord_index_dims[-1]
            coord_dims = tuple(self.ranges_dims_map[id(r)]
                               for r, c_idx in (self.indexed_ranges[i] for i in coord_index_dims) if c_idx == 0)
            if any_pred_true(slice_index_dims, lambda x: first < x < last):
                return coord_dims + slice_index_dims
            else:
                sl_idx_dims_1 = tuple(idx for idx in slice_index_dims if idx < first)
                sl_idx_dims_2 = tuple(idx for idx in slice_index_dims if first < idx)
                return sl_idx_dims_1 + coord_dims + sl_idx_dims_2
        else:
            return tuple(self.ranges_dims_map[id(r)] for r, c_idx in self.indexed_ranges if c_idx == 0 and not r.is_int)

    def _set_dims(self):
        self.data_dims = self._sort_dims()
        self.data_ranges = tuple(self.indexed_ranges[dims][0]
                                 if isinstance(dims, int) else self.indexed_ranges[dims[0]][0]
                                 for dims in self.data_dims)
        self.data_ranges_map = tuple(id(r) for r in self.data_ranges)
        self.data_ndim = len(self.data_ranges)

    def _set_index(self):
        self._index = DenseIndex(self)

    def set_dims_and_index(self):
        self._set_dims()
        self._set_index()

    def set_data_from_input(self, data):
        if data is None:
            self.data = None
        else:
            self.data = maybe_cast_to_tensor(data, dtype=torch.float32)

    def set_data_from_weight_init(self, initializer, **kwargs):
        self.data = torch.nn.Parameter(data=torch.Tensor(*(r.data_shape for r in self.data_ranges)), requires_grad=True)
        if initializer == "normal":
            self.data.data.normal_(**kwargs)
        elif initializer == "zeros":
            self.data.data.fill_(0.0)
        elif initializer == "uniform":
            self.data.data.uniform_(kwargs.pop("from", 0.0), kwargs.pop("to", 1.0))
        else:
            raise ValueError
