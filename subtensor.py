import torch

from torch_utils import maybe_cast_to_tensor
from misc import any_pred_true, indices_pred_true
from index import DenseIndex


class SubTensor(object):

    """
    A SubTensor object is used to represent data that will be writen or data which was read from a tensor. This
    data corresponds only to a subset of the values in the tensor, subset which is specified by a Domain object.
    The data is a dense torch.Tensor.

    A SubTensor object contains all the information needed to write and read data correctly to a tensor.
    For example, when data is collected from a torch.Tensor via fancy indexing, the dimensions of the collected data
    could be permuted. The SubTensor object keeps track of which of its data dimensions correspond to which tensor
    dimensions and ensures that data gets written and read correctly via an Index object. Other similar properties
    will be added to support sparse computation in the future.

    A SubTensor is also used to input data into the program or to initialize program weights.

    Each data dimension corresponds to a Range in the Domain. However, the data dimensions need not be in the same
    order as the dimensions in the domain and it could be the case that a data dimension corresponds to multiple
    tensor dimensions(e.g. CoordRanges) or that some Ranges are not represented in the data(e.g IntRanges).
    """

    def __init__(self, domain):

        """
        Creates a SubTensor object
        :param domain: Domain
            A Domain object specifying the part of the tensor to which the data will be written to or from which the
            data will be read
        """

        self.domain = domain

        # inherit fields from the domain
        self.indexed_ranges = self.domain.indexed_ranges
        self.slices = self.domain.slices
        self.coords = self.domain.coords
        self.slice_dims = self.domain.slice_dims
        self.coord_dims = self.domain.coord_dims
        self.slice_dims_map = self.domain.slice_dims_map
        self.coord_dims_map = self.domain.coord_dims_map
        self.ndim = self.domain.ndim
        self.shape = self.domain.shape
        self.dummy = self.domain.dummy

        self.data = None

        self.data_dims = None   # maps each data dim to the dimensions in the tensor that it corresponds to.
        self.data_ranges = None    # maps each data dim to the range in the domain that it corresponds to
        self.data_ndim = None

        self._index = None  # index object used to index into a dense(at least for now) torch tensor

        self._set_dims()
        self._set_index()

    @property
    def empty(self): return self.domain.empty or self.data is None
    @property
    def index(self): return self._index.index()

    def set_zeros(self): self.data = self.get_zeros()
    def get_zeros(self): return torch.zeros(self.get_data_shape(), dtype=torch.float32)

    def get_indexed_ranges(self, indices=None): return self.domain.get_indexed_ranges(indices)
    def get_ranges(self, indices=None): return self.domain.get_ranges(indices)
    def get_shape(self, indices=None): return self.domain.get_shape(indices)
    def get_range_types(self, indices=None): return self.domain.get_range_types(indices)
    def get_data_shape(self): return tuple(ds for ds in (r.data_shape for r in self.data_ranges) if ds is not None)

    def _sort_dims(self):
        """
        Given the types of ranges in the SubTensor's Domain, the data selected with the SubTensor Index from a torch
        tensor can get permuted, or some tensor dimensions will not be in the data anymore in case of an IntRange across
        that dimension. In this function, we compute the map from the data dimensions to the ranges and
        dimensions in the tensor it corresponds to.
        :return:
        """
        slice_index_dims = tuple(indices_pred_true(self.domain.get_ranges(), lambda x: x.is_dense))
        coord_index_dims = tuple(indices_pred_true(self.domain.get_ranges(), lambda x: x.is_sparse))

        if len(coord_index_dims) > 1:
            first, last = coord_index_dims[0], coord_index_dims[-1]
            coord_dims = tuple(self.coord_dims_map[id(r)] if r.is_coord else self.slice_dims_map[id(r)]
                               for r, c_idx in (self.indexed_ranges[i] for i in coord_index_dims) if c_idx == 0)
            if any_pred_true(slice_index_dims, lambda x: first < x < last):
                return coord_dims + slice_index_dims
            else:
                sl_idx_dims_1 = tuple(idx for idx in slice_index_dims if idx < first)
                sl_idx_dims_2 = tuple(idx for idx in slice_index_dims if first < idx)
                return sl_idx_dims_1 + coord_dims + sl_idx_dims_2
        else:
            return tuple(self.coord_dims_map[id(r)] if r.is_coord else self.slice_dims_map[id(r)]
                         for r, c_idx in self.indexed_ranges if c_idx == 0 and not r.is_int)

    def _set_dims(self):
        """
        Sets all fields related to keeping track of the data dimensions.
        :return:
        """
        self.data_dims = self._sort_dims()
        self.data_ranges = tuple(self.indexed_ranges[dims][0] if isinstance(dims, int) else
                                 self.indexed_ranges[dims[0]][0] for dims in self.data_dims)
        self.data_ndim = len(self.data_ranges)

    def _set_index(self):
        self._index = DenseIndex(self)

    def set_data_from_input(self, data, cast_to_float=True):
        """
        Set the tensor data from an input to the program
        :param data: castable to a torch.Tensor
            Data to be inputted into the program. The data has to be inputted in the order corresponding to the
            SubTensor's Index. See numpy's fancy indexing rules for clarifications.
        :param cast_to_float: bool
            Flag indicating whether the data should be cast to a float or not
        :return:
        """
        if data is not None:
            if cast_to_float:
                self.data = maybe_cast_to_tensor(data, dtype=torch.float32)
            else:
                self.data = maybe_cast_to_tensor(data, dtype=None)

    def set_data_from_weight_init(self, initializer, **kwargs):
        """
        Initialize the data as a set of weights.
        :param initializer: str
            A unique string identifier specifying the type of torch initializer to use
        :param kwargs: dict
            Other arguments to use by the initializer
        :return:
        """

        batch_dim = kwargs.pop("batch_dim", None)
        self.data = torch.nn.Parameter(data=torch.Tensor(*(r.data_shape for i, r in enumerate(self.data_ranges)
                                                           if i != batch_dim)), requires_grad=True)

        if initializer == "normal":
            torch.nn.init.normal_(self.data, **kwargs)
        elif initializer == "uniform":
            torch.nn.init.uniform_(self.data, kwargs.pop("from", 0.0), kwargs.pop("to", 1.0))
        elif initializer == "xavier_uniform":
            torch.nn.init.xavier_uniform_(self.data, kwargs.pop("gain"))
        elif initializer == "xavier_normal":
            torch.nn.init.xavier_normal_(self.data, kwargs.pop("gain"))
        else:
            raise ValueError("Unknown weight initializer type", initializer, ". Available values: normal, uniform, xavier_uniform, xavier_normal")
