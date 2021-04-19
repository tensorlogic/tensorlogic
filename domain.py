import numpy as np
import sparse as sp
import torch

from sortedcontainers import SortedDict
from functools import partial
from collections import defaultdict
from collections.abc import Iterable
from itertools import product, chain

from range import IntRange, SliceRange, SetRange, CoordRange, DummyRange, range_comp_binary
from misc import min_function_binary, max_function_binary, min_max_function, power_set


class Domain(object):
    """
    A Domain object is used to represent the domain of discourse over which a subtensor runs. More specifically,
    the Domain object will store the parts of the tensor to which a certain subtensor refers to. During backward
    chaining, the domain will store the parts of the tensor being queried. During the forward computation, the Domain
    will be used to index into pytorch tensors in order to update or select the data from the appropriate memory
    locations.

    The Domain object has as many dimensions as the tensor which it indexes. For each dimension, it has a
    corresponding range or one dimension of a CoordRange(the CoordRanges can span across multiple domain dimensions).
    The map between tensor dimensions and the ranges is stored in indexed_ranges.

    The Domain stores the CoordRanges in the coords field, separate from the rest of the ranges which are stored
    in the slices field(the slice dimensions will contain either IntRanges, SliceRanges, SetRanges, DummyRanges).
    """

    def __init__(self, coords, slices, coord_dims, slice_dims, shape):

        """
        Creates a Domain object.
        :param coords: tuple[CoordRange], list[CoordRange]
            list-like containing the CoordRanges that the Domain contains.
        :param slices: tuple[Range], list[Range]
            list-like containing the Ranges(all except CoordRanges) that the Domain contains.
        :param coord_dims: tuple[tuple], list[tuple]
            list-like containing the dimensions of the tensors over which the coords run given as tuples.
            The i-th entry in coords must match the i-th entry in coord_dims. We use tuples because CoordRanges
            generally refer to multiple tensor dimensions.
        :param slice_dims: tuple[int], list[int]
            list-like containing the dimensions of the tensors over which the slices run given as integers.
            The i-th entry in slices must match the i-th entry in slice_dims. We use integers because slice ranges will
            refer to one single tensor dimension.
        :param shape: tuple[int], list[int]
            the shape of the tensor over which the Domain runs.
        """
        super(Domain, self).__init__()

        self.shape = tuple(shape)
        self.ndim = len(shape)

        self.coord_dims, self.coords = tuple(coord_dims or ()), tuple(coords or ())
        self.slice_dims, self.slices = tuple(slice_dims or ()), tuple(slices or ())

        # inverse map from the id of the coords and slices to the tensor dimensions they represent.
        self.coord_dims_map = {id(coords): dims for coords, dims in zip(self.coords, self.coord_dims)}
        self.slice_dims_map = {id(sl): dim for sl, dim in zip(self.slices, self.slice_dims)}

        # create the indexed_ranges. Sort the dimensions using a Sorted dictionary. Only keep the values at the end.
        # each item in indexed_range is a tuple (Range, int) where the integer represents the dimension within the range
        # that the dimension in the domain refers to. For slices, this is always 0 since slices represent only one
        # dimension. For coords, they run from 0 to the ndim of the CoordRange.
        dims = SortedDict()
        for coords, coord_dims in zip(self.coords, self.coord_dims):
            for i, d in enumerate(coord_dims):
                dims[d] = (coords, i)
            coords.shape = tuple(self.get_shape(coord_dims))
        for sl, d in zip(self.slices, self.slice_dims):
            sl.shape = self.shape[d]
            dims[d] = (sl, 0)
        self.indexed_ranges = tuple(dims.values())

        self.dummy = all(isinstance(r, DummyRange) for r in self.get_ranges())

    @staticmethod
    def domain_from_ranges(ranges, shape):

        """
        Given a tuple of (Range, int) objects or Nones(which correspond to DummyRanges) and the shape of the tensor,
        create a Domain. The ranges tuple should contain what will become the indexed_ranges of the new Domain.
        The CoordRanges given as input must not necessarily contain all the dimensions of the CoordRange and only the
        dimensions indicated in the input tuple will be selected
        :param ranges: tuple[(Range, int), (None, int)]
            The ranges and the dimension of each range placed in a tuple such that the i-th entry of the tuple
            corresponds to the i-th dimension of the tensor. None values are treated as DummyRanges.
        :param shape: tuple[int], list[int]
            the shape of the tensor over which the Domain runs.
        :return: Domain
        """
        slices, coords, coords_id, coords_idx = {}, {}, {}, defaultdict(lambda: ([], []))

        for idx, (r, c_idx) in enumerate(ranges):

            if r is None:
                slices[idx] = DummyRange()
            elif not r.is_coord:
                slices[idx] = r
            else:
                r_id = id(r)
                coords_id[r_id] = r
                coords_idx[r_id][0].append(c_idx)
                coords_idx[r_id][1].append(idx)

        for r_id, r in coords_id.items():

            c_idx, idx = coords_idx[r_id]
            c_idx, idx = tuple(c_idx), tuple(idx)

            r = r.select(c_idx)
            coords[idx] = r

        return Domain.domain_from_dicts(coords, slices, shape)

    @staticmethod
    def domain_from_dicts(coords, slices, shape):
        """
        Create a Domain from a dictionary of slices and tuples. The keys will be the dimensions to which the
        Ranges correspond to(ints for slices, tuples for coords).
        :param coords: dict[tuple, CoordRange]
        :param slices: dict[int, Range]
        :param shape: tuple[int], list[int]
        :return: Domain
        """
        return Domain(tuple(coords.values()), tuple(slices.values()), tuple(coords.keys()), tuple(slices.keys()), shape)

    @staticmethod
    def domain_from_query_data(domain_tuple, domain_vals, shape):
        """
        Create a Domain from query data. Query data consists of a domain tuple which has the same format as the
        tensor indexing in the program declaration and the domain_vals is a dictionary matching the names in the
        domain_tuple to values.
        The type of ranges will be inferred from the values:
            None -> DummyRange
            int -> IntRange
            1d array-like -> SetRange
            nd array-like -> CoordRange
            slice -> SliceRange
        :param domain_tuple: tuple[str], list[str]
        :param domain_vals: dict[str, [None, integer-like, slice, Iterable]]
        :param shape: tuple[int], list[int]
        :return: Domain
        """

        coords, slices = {}, {}
        coord_dims, slice_dims = defaultdict(list), {}

        # infer the type of range from the value
        for key, val in domain_vals.items():
            if isinstance(val, (int, np.integer, torch.IntType)):
                slices[key] = IntRange(value=val)
            elif isinstance(val, slice):
                slices[key] = SliceRange(value=val)
            elif isinstance(val, Iterable):
                val = np.array(val, np.long) if not isinstance(val, torch.Tensor) else val
                if val.ndim == 1:
                    slices[key] = SetRange(value=val)
                else:
                    coords[key] = CoordRange(ndim=val.ndim, value=val)
            elif val is None:
                slices[key] = DummyRange()
            else:
                raise ValueError("Wrong query val type", type(val))

        # match the range to the name
        for dim, range_name in enumerate(domain_tuple):
            range_name = range_name.split(".")
            if len(range_name) == 1:
                slice_dims[range_name[0]] = dim
            else:
                coord_dims[range_name[0]].append(dim)

        return Domain(coords=(coords[k] for k in coords.keys()),
                      slices=(slices[k] for k in slices.keys()),
                      coord_dims=(coord_dims[k] for k in coords.keys()),
                      slice_dims=(slice_dims[k] for k in slices.keys()),
                      shape=shape)

    @property
    def empty(self):
        return any(r.empty for r in chain(self.slices, self.coords))

    def get_indexed_ranges(self, indices=None):
        return (self.indexed_ranges[idx] for idx in indices) if indices is not None else iter(self.indexed_ranges)

    def get_ranges(self, indices=None):
        return (self.indexed_ranges[idx][0] for idx in indices) if indices is not None else (d[0] for d in self.indexed_ranges)

    def get_shape(self, indices=None):
        return (self.shape[idx] for idx in indices) if indices is not None else iter(self.shape)

    def get_range_types(self, indices=None):
        return (type(r) for r in self.get_ranges(indices))

    def get_values(self, indices=None):
        return (r.value for r in self.get_ranges(indices))

    def get_sparse_values(self, indices=None):
        return (r.sparse_value for r in self.get_ranges(indices))

    def _get_coords_reshape(self, coord_range, dims):

        coord_dims = iter(self.coord_dims_map[id(coord_range)])
        cur_coord_dim, coord_dims_found = next(coord_dims), 0

        for idx in dims:
            if idx == cur_coord_dim:
                yield coord_range.shape[coord_dims_found]
                cur_coord_dim = next(coord_dims, None)
                coord_dims_found += 1
            else:
                yield 1

    def _get_reshape(self, dims=None):

        dims = dims if dims is not None else range(self.ndim)
        num_slices, num_coords = 0, 0
        reshapes = []

        for idx, (r, c_idx) in enumerate(self.get_indexed_ranges(dims)):

            if not r.is_coord:
                num_slices += 1
                if not r.is_dummy:
                    reshapes.append(tuple(r.shape if i == idx else 1 for i in range(len(dims))))
            elif c_idx == 0:
                num_coords += 1
                reshapes.append(tuple(self._get_coords_reshape(r, dims)))

        return tuple(reshapes) if num_coords + num_slices > 1 else (None, )

    def join_ranges(self, dims=None):

        """
        Joins the given dimensions of the domain. In order to do so, we use the sparse values of the ranges to join.
        The shape of the ranges' sparse values is broadcast in such a way that the sparse.COO arrays involved in the
        join will have the same ndim as the final joined sparse.COO array. The join is performed by
        multiplication(bitwise and) of the sparse arrays. Return True if all the ranges involved are dummies.
        :param dims: tuple[int], list[int]
        :return: sparse.COO 
        """

        reshapes = iter(self._get_reshape(dims))
        value = True

        for r, c_idx in self.get_indexed_ranges(dims):

            if not r.is_dummy and c_idx == 0:

                resh = next(reshapes)
                if resh is not None:
                    r_sparse = r.sparse_value.reshape(resh)
                else:
                    r_sparse = r.sparse_value

                value = r_sparse if value is True else value & r_sparse

        return value


# compare two domains. Return -1 if d1 contains d2, 1 if vice-versa, 0 if the two are equal, and None if the two domains
# are not comparable. For now, the check is fairly restricted and only if two ranges are the same or if one range is a
# DummyRange can containment be established for ranges. If containment can be established for all ranges, and is
# consistent across the whole domain, only then can the comparison be performed.
def domain_comp_binary(d1, d2, dims=None):

    ranges = zip(d1.get_ranges(dims), d2.get_ranges(dims))
    comp = range_comp_binary(*next(ranges))

    if comp is not None:
        for r1, r2 in ranges:
            new_comp = range_comp_binary(r1, r2)
            if new_comp is None:
                return None
            elif new_comp != comp:
                if new_comp in (-1, 1) and comp in (-1, 1):
                    return None
                elif comp == 0:
                    comp = new_comp
    return comp


# other comparison functions
max_domain_binary = partial(max_function_binary, domain_comp_binary)
max_domain = partial(min_max_function, max_domain_binary)

min_domain_binary = partial(min_function_binary, domain_comp_binary)
min_domain = partial(min_max_function, min_domain_binary)


# generate random domains. Test purposes
def generate_domains(pairs=False):

    ndims = 5
    shape_ = (5, 6, 7, 8, 9)
    types = [0, 1, 2]
    pair_sample_prob = 0.33

    def sample_int(i): return np.random.randint(0, shape_[i])

    def sample_sls(i):
        start = np.random.randint(0, shape_[i])
        end = np.random.randint(start + 1, shape_[i] + 1)
        step = np.random.randint(1, end - start + 1)
        return slice(start, end, step)

    def sample_sets(i): return sp.random((shape_[i], ), density=0.5).coords[0]

    def sample_coords(i):
        coords = sp.random(tuple(shape_[d] for d in i), density=0.25).coords
        return coords

    def sample_range(t, i):

        if t is None:
            if isinstance(i, int):
                t = np.random.choice((IntRange, SetRange, SliceRange, DummyRange))
            else:
                t = CoordRange

        if t == IntRange:
            return t(value=sample_int(i))
        elif t == SetRange:
            return t(value=sample_sets(i))
        elif t == SliceRange:
            return t(value=sample_sls(i))
        elif t == DummyRange:
            return DummyRange()
        elif t == CoordRange:
            return t(ndim=len(i), value=sample_coords(i))

    def sample_new_dims(other_dom):

        coos, sls, coo_dims, sl_dims = [], [], [], []

        for sd in other_dom.slice_dims:
            dim = other_dom.indexed_ranges[sd][0]
            if np.random.random() > 0.5 or dim.is_dummy:
                sls.append(dim)
            else:
                t = type(dim)
                sls.append(sample_range(t, sd))
            sl_dims.append(sd)

        for cd in other_dom.coord_dims:
            dim = other_dom.indexed_ranges[cd[0]][0]
            if np.random.random() > 0.5:
                coos.append(dim)
            else:
                coos.append(sample_range(CoordRange, cd))
            coo_dims.append(cd)

        d = Domain(coos, sls, coo_dims, sl_dims, shape_)
        return d

    def increase_dims(other_dom):

        coos, sls = SortedDict(), SortedDict()

        for sd in other_dom.slice_dims:
            dim = other_dom.indexed_ranges[sd][0]
            if np.random.random() > 0.8 or dim.is_dummy:
                sls[sd] = dim
            else:
                sls[sd] = DummyRange()

        for cd in other_dom.coord_dims:
            dim = other_dom.indexed_ranges[cd[0]][0]
            if np.random.random() > 0.8:
                coos[cd] = dim
            else:
                for sd in cd:
                    sls[sd] = DummyRange()

        d = Domain.domain_from_dicts(coos, sls, shape_)
        return d

    def increase_and_decrease_dims(other_dom):

        coos, sls = SortedDict(), SortedDict()

        for sd in other_dom.slice_dims:
            dim = other_dom.indexed_ranges[sd][0]
            if dim.is_dummy:
                sls[sd] = sample_range(None, sd)
            else:
                if np.random.random() > 0.8:
                    sls[sd] = dim
                else:
                    sls[sd] = DummyRange()

        for cd in other_dom.coord_dims:
            dim = other_dom.indexed_ranges[cd[0]][0]
            if np.random.random() > 0.8:
                coos[cd] = dim
            else:
                for sd in cd:
                    sls[sd] = DummyRange()

        d = Domain.domain_from_dicts(coos, sls, shape_)
        return d

    def create_domain(dense_dims=(),
                      int_dims=(), slice_dims=(), set_dims=(), dummy_dims=(),
                      coord1_dims=(), coord2_dims=()):

        coos, sls = SortedDict(), SortedDict()

        if coord1_dims:
            coos[coord1_dims] = sample_range(CoordRange, coord1_dims)

        if coord2_dims:
            coos[coord2_dims] = sample_range(CoordRange, coord2_dims)

        for idx in range(0, ndims):

            if idx in dense_dims:
                sls[idx] = sample_range(None, idx)
            elif idx in int_dims:
                sls[idx] = sample_range(IntRange, idx)
            elif idx in slice_dims:
                sls[idx] = sample_range(SliceRange, idx)
            elif idx in set_dims:
                sls[idx] = sample_range(SetRange, idx)
            elif idx in dummy_dims:
                sls[idx] = sample_range(DummyRange, idx)

        d = Domain.domain_from_dicts(coos, sls, shape_)
        return d

    def domain_generator():

        for sl_type in product(types, repeat=ndims):

            dense_dims, coord1_dims, coord2_dims = [], [], []

            for num, index in enumerate(sl_type):
                if index == 0:
                    dense_dims.append(num)
                elif index == 1:
                    coord1_dims.append(num)
                else:
                    coord2_dims.append(num)

            if len(coord2_dims) > len(coord1_dims):
                continue

            yield create_domain(dense_dims=tuple(dense_dims), coord1_dims=tuple(coord1_dims), coord2_dims=tuple(coord2_dims))

    def pair_generator():

        for d1 in domain_generator():
            print(tuple(d1.get_range_types()))
            for d2 in domain_generator():
                yield 0, d1, d2
                if np.random.random() > pair_sample_prob:
                    t = np.random.choice((1, 2, 3, 4))
                    if t == 1:
                        yield 1, d1, d1
                    elif t == 2:
                        yield 2, d1, sample_new_dims(d1)
                    elif t == 3:
                        yield 3, d1, increase_dims(d1)
                    else:
                        yield 4, d1, increase_and_decrease_dims(d1)

    if pairs:
        return pair_generator()
    else:
        return domain_generator()


if __name__ == "__main__":


    slices_ = (IntRange(value=1),
               SetRange(value=[0, 1, 2], shape=10),
               DummyRange(),
               SliceRange(value=slice(5, None), shape=10),
               DummyRange())

    coords_ = (CoordRange(value=[[3, 4, 5], [13, 14, 15]], shape=(10, 20), ndim=2),
               CoordRange(value=[[13, 14, 15], [23, 24, 25]], ndim=2))

    d = Domain(coords_, slices_, coord_dims=((0, 3), (1, 5)), slice_dims=(2, 4, 6, 7, 8),
               shape=(10, 20, 10, 20, 10, 30, 10, 10, 10))

    print(d.slices)
    print(d.coords)
    print(d.coord_dims)
    print(d.coord_dims)
    print(d.slice_dims)
    print(d.indexed_ranges)
    print(tuple(d.get_indexed_ranges()))
    print(tuple(d.get_ranges()))
    print(tuple(d.get_shape()))
    print(tuple(d.get_range_types()))

    for sl_dims, coo_dims in product(power_set(d.slice_dims), power_set(d.coord_dims)):

        gr = sorted(list(sl_dims) + list(d for dims in coo_dims for d in dims))

        c_dims = []
        for idx, g in enumerate(gr):
            if g in sl_dims:
                c_dims.append(idx)
            else:
                for dims in coo_dims:
                    if g == dims[0]:
                        c_dims.append([gr.index(d) for d in dims])

    d1 = Domain((CoordRange(ndim=2),), (), ((0, 1),), (), shape=(10, 20))
    d2 = Domain((CoordRange(ndim=2),), (), ((0, 1),), (), shape=(10, 20))
    d3 = Domain((), (DummyRange(), DummyRange()), (), (0, 1), shape=(10, 20))
    d4 = Domain((), (DummyRange(), DummyRange()), (), (0, 1), shape=(10, 20))
    d5 = Domain((), (SetRange(), DummyRange()), (), (0, 1), shape=(10, 20))
    d6 = Domain((), (DummyRange(), SetRange()), (), (0, 1), shape=(10, 20))

    s1, s2 = SetRange(), SetRange()
    d7 = Domain((), (s1, s2), (), (0, 1), shape=(10, 20))
    d8 = Domain((), (s1, DummyRange()), (), (0, 1), shape=(10, 20))
    d9 = Domain((), (DummyRange(), DummyRange()), (), (0, 1), shape=(10, 20))

    assert (domain_comp_binary(d1, d2) is None)
    assert (domain_comp_binary(d1, d3) == 1)
    assert (domain_comp_binary(d3, d2) == -1)
    assert (domain_comp_binary(d1, d1) == 0)
    assert (domain_comp_binary(d3, d3) == 0)
    assert (domain_comp_binary(d3, d4) == 0)
    assert (domain_comp_binary(d4, d5) == -1)
    assert (domain_comp_binary(d5, d6) is None)
    assert (max_domain(d1, d3, d5) is d3)
    assert (max_domain(d1, d2, d5) is None)
    assert (min_domain(d7, d8, d9) is d7)
    assert (min_domain(d7, d7, d8) is d7)
    assert (min_domain(d1, d7, d8) is None)

    d = Domain.domain_from_query_data(("x", "y"), dict(x=None, y=None), (20, 20))
    print(list(d.get_values()), list(d.get_range_types()))
    d = Domain.domain_from_query_data(("x", "y"), dict(x=1, y=None), (20, 20))
    print(list(d.get_values()), list(d.get_range_types()))
    d = Domain.domain_from_query_data(("y.0", "x", "y.1"), dict(x=1, y=[[0, 1, 2], [0, 1, 2]]), (20, 20, 20))
    print(list(d.get_values()), list(d.get_range_types()))