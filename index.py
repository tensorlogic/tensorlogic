from misc import count_pred_true


class Index(object):
    """
    An Index object is used to store all the information needed to index into a Tensor object. More specifically,
    an Index object will be created for each SubTensor and it will be used to read data from the Tensor and into the
    SubTensor or to write the data from the SubTensor to the Tensor.

    For the time being, only DenseIndex is supported(that is an index into a dense torch.Tensor object).
    """
    def __init__(self, subtensor):
        """
        Create an Index object
        :param subtensor: SubTensor
            The SubTensor to which the Index belongs to
        """
        self._index = None

        self.indexed_ranges = subtensor.indexed_ranges
        self.ndim = subtensor.ndim

    def index(self): raise NotImplemented


class DenseIndex(Index):

    """
    A DenseIndex is used to index into a DenseTensor object. Its job is to correctly format the values of the ranges
    in the Domain of the SubTensor in order to collect the data from the correct memory locations.
    """
    def __init__(self, subtensor):
        super(DenseIndex, self).__init__(subtensor)

    def _get_reshapes(self):
        """
        Get the reshapes of the indices corresponding to each Range in the Domain. Sparse-like indices(those coming
        from SetRange or CoordRange) need to be reshaped such that they select the correct parts of a torch.tensor
        as per the fancy indexing rules.
        :return: generator[tuple]
        """
        reshapes_map = {}
        len_reshape = count_pred_true(self.indexed_ranges, lambda r: r[0].is_sparse and r[1] == 0)

        i = 0
        for r, c_idx in self.indexed_ranges:
            if r.is_sparse and len_reshape > 1:
                if c_idx == 0:
                    reshape = tuple(-1 if i == idx else 1 for idx in range(len_reshape))
                    reshapes_map[id(r)] = reshape
                    i += 1
                    yield reshape
                else:
                    yield reshapes_map[id(r)]
            else:
                yield None

    def index(self):
        index = [None] * self.ndim
        for dim, ((r, c_idx), reshape) in enumerate(zip(self.indexed_ranges, self._get_reshapes())):
            index[dim] = r.index(c_idx) if reshape is None else r.index(c_idx).reshape(reshape)
        return tuple(index)
