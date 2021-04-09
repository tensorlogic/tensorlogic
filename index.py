from misc import indices_pred_true, count_pred_true, is_sorted, reduce_concat


class Index(object):

    def __init__(self, subtensor):

        self._index = None

        self.indexed_ranges = subtensor.indexed_ranges
        self.ranges = subtensor.data_ranges
        self.range_dims = subtensor.data_dims
        self.shape = subtensor.shape

        self.needs_sort = not is_sorted(reduce_concat(*self.range_dims))

        self.constant_indexed_ranges_dims = tuple(indices_pred_true(self.indexed_ranges, lambda r: not r[0].empty))
        self.non_constant_indexed_ranges_dims = tuple(indices_pred_true(self.indexed_ranges, lambda r: r[0].empty))

        self.constant_indexed_ranges = tuple(self.indexed_ranges[i] for i in self.constant_indexed_ranges_dims)
        self.non_constant_indexed_ranges = tuple(self.indexed_ranges[i] for i in self.non_constant_indexed_ranges_dims)

        self.constant_range_dims = tuple(indices_pred_true(self.ranges, lambda r: not r.empty))
        self.non_constant_range_dims = tuple(indices_pred_true(self.ranges, lambda r: r.empty))

        self.constant_ranges = tuple(self.ranges[i] for i in self.constant_range_dims)
        self.non_constant_ranges = tuple(self.ranges[i] for i in self.non_constant_range_dims)

        self.constant_dims = tuple(self.range_dims[i] for i in self.constant_range_dims)
        self.non_constant_dims = tuple(self.range_dims[i] for i in self.non_constant_range_dims)

        self.constant = len(self.non_constant_indexed_ranges_dims) == 0

    def _init_index(self): raise NotImplemented
    def _set_index_dims(self, *args, **kwargs): raise NotImplemented
    def index(self): raise NotImplemented


class DenseIndex(Index):

    def __init__(self, subtensor):

        super(DenseIndex, self).__init__(subtensor)

        reshapes = self._get_reshapes()

        self.constant_reshapes = tuple(reshapes[i] for i in self.constant_indexed_ranges_dims)
        self.non_constant_reshapes = tuple(reshapes[i] for i in self.non_constant_indexed_ranges_dims)

        self._init_index()

    def _get_reshapes(self):

        reshapes = []
        reshapes_map = {}

        len_reshape = count_pred_true(self.indexed_ranges, lambda r: r[0].is_sparse and r[1] == 0)

        i = 0
        for r, c_idx in self.indexed_ranges:
            if r.is_sparse and len_reshape > 1:
                if c_idx == 0:
                    reshape = tuple(-1 if i == idx else 1 for idx in range(len_reshape))
                    reshapes_map[id(r)] = reshape
                    i += 1
                else:
                    reshape = reshapes_map[id(r)]
            else:
                reshape = None
            reshapes.append(reshape)
        return tuple(reshapes)

    def _set_index_dims(self, dims, ranges, reshapes):
        for dim, (r, c_idx), reshape in zip(dims, ranges, reshapes):
            self._index[dim] = r.index(c_idx) if reshape is None else r.index(c_idx).reshape(reshape)

    def _init_index(self):
        self._index = [None] * len(self.indexed_ranges)
        self._set_index_dims(self.constant_indexed_ranges_dims, self.constant_indexed_ranges, self.constant_reshapes)
        if self.constant:
            self._index = tuple(self._index)

    def index(self):
        if self.constant:
            return self._index
        self._set_index_dims(self.non_constant_indexed_ranges_dims,
                             self.non_constant_indexed_ranges,
                             self.non_constant_reshapes)
        return tuple(self._index)
