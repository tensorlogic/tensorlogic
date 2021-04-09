from sortedcontainers import SortedDict
from functools import partial
from collections import defaultdict
from collections.abc import Iterable
from itertools import product, chain
import numpy as np
import sparse as sp

from range import IntRange, SliceRange, SetRange, CoordsRange, DummyRange, range_comp_binary, equality_range
from misc import min_function_binary, max_function_binary, min_max_function, power_set, reduce_concat


class Domain(object):

    def __init__(self, coords, slices, coord_dims, slice_dims, shape):

        super(Domain, self).__init__()

        self.shape = shape
        self.ndim = len(shape)

        coord_dims = coord_dims or ()
        slice_dims = slice_dims or ()

        coords_dict = SortedDict(zip((tuple(coo_dims) for coo_dims in coord_dims), coords))
        slices_dict = SortedDict(zip(slice_dims, slices))

        self.coord_dims, self.coords = tuple(coords_dict.keys()), tuple(coords_dict.values())
        self.coord_dims_flat = tuple(sorted(reduce_concat(self.coord_dims)))
        self.slice_dims, self.slices = tuple(slices_dict.keys()), tuple(slices_dict.values())

        self.coord_dims_map = {id(coords): dims for coords, dims in zip(self.coords, self.coord_dims)}
        self.slice_dims_map = {id(sl): dim for sl, dim in zip(self.slices, self.slice_dims)}

        dims = SortedDict()

        for coords, coord_dims in zip(self.coords, self.coord_dims):
            for i, d in enumerate(coord_dims):
                if coords.shape is not None:
                    assert coords.shape[i] == self.shape[d]
                dims[d] = (coords, i)
            if coords.shape is None:
                coords.shape = tuple(self.get_shape(coord_dims))

        for sl, d in zip(self.slices, self.slice_dims):
            if sl.shape is not None:
                assert sl.shape == self.shape[d]
            else:
                sl.shape = self.shape[d]
            dims[d] = (sl, 0)

        self.indexed_ranges = tuple(dims.values())
        self.range_dims_map = {id(r): dim for r, dim in zip(chain(self.slices, self.coords), chain(self.slice_dims, self.coord_dims))}

        self.dummy = all(isinstance(r, DummyRange) for r in self.get_ranges())
        self._sparse_value = None

    @staticmethod
    def domain_from_ranges(ranges, shape):

        slices, coords, coords_id, coords_idx = SortedDict(), SortedDict(), {}, defaultdict(lambda: ([], []))

        for idx, (r, c_idx) in enumerate(ranges):

            if r is None:
                slices[idx] = DummyRange()
            elif r.is_slice:
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

        coords, coord_dims = tuple(coords.values()), tuple(coords.keys())
        slices, slice_dims = tuple(slices.values()), tuple(slices.keys())
        return Domain(coords, slices, coord_dims, slice_dims, shape)

    @staticmethod
    def domain_from_query_data(domain_tuple, domain_vals, shape):

        coords, slices = {}, {}
        coord_dims, slice_dims = defaultdict(list), {}

        for key, val in domain_vals.items():

            if isinstance(val, (int, np.integer)):
                slices[key] = IntRange(value=val)
            elif isinstance(val, slice):
                slices[key] = SliceRange(value=val)
            elif isinstance(val, Iterable):
                val = np.array(val, np.long)
                if val.ndim == 1:
                    slices[key] = SetRange(value=val)
                else:
                    coords[key] = CoordsRange(ndim=val.ndim, value=val)
            elif val is None:
                slices[key] = DummyRange()
            else:
                raise ValueError("Wrong query val type", type(val))

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

    # ############### GET STUFF ##################

    @property
    def empty(self): return any(r.empty for r in chain(self.slices, self.coords))
    def get_all_constant(self, indices=None): return all(self.get_constant(indices))
    def get_constant(self, indices=None): return (r.constant for r in self.get_ranges(indices))
    def get_indexed_ranges(self, indices=None): return (self.indexed_ranges[idx] for idx in indices) if indices is not None else iter(self.indexed_ranges)
    def get_ranges(self, indices=None): return (self.indexed_ranges[idx][0] for idx in indices) if indices is not None else (d[0] for d in self.indexed_ranges)
    def get_shape(self, indices=None): return (self.shape[idx] for idx in indices) if indices is not None else iter(self.shape)
    def get_range_types(self, indices=None): return (type(r) for r in self.get_ranges(indices))
    def get_values(self, indices=None): return (r.value for r in self.get_ranges(indices))
    def get_sparse_values(self, indices=None): return (r.sparse_value for r in self.get_ranges(indices))
    def get_torch_values(self, indices=None): return (r.torch_value for r in self.get_ranges(indices))

    # ############### COMPUTE VALUES ##################

    def _get_coords_reshape(self, coord_range, group):

        coord_dims = iter(self.coord_dims_map[id(coord_range)])
        cur_coord_dim, coord_dims_found = next(coord_dims), 0

        for idx in group:
            if idx == cur_coord_dim:
                yield coord_range.shape[coord_dims_found]
                cur_coord_dim = next(coord_dims, None)
                coord_dims_found += 1
            else:
                yield 1

        assert cur_coord_dim is None, "All the dimensions of the coordinate range have to be included in the group"

    def get_reshape(self, group=None):

        group = group if group is not None else range(self.ndim)
        num_slices, num_coords = 0, 0
        reshapes = []

        for idx, (r, c_idx) in enumerate(self.get_indexed_ranges(group)):

            if r.is_slice:
                num_slices += 1
                if not r.is_dummy:
                    reshapes.append(tuple(r.shape if i == idx else 1 for i in range(len(group))))
            elif c_idx == 0:
                num_coords += 1
                reshapes.append(tuple(self._get_coords_reshape(r, group)))

        return tuple(reshapes) if num_coords + num_slices > 1 else (None, )

    def group_ranges(self, group=None, reshapes=None):

        reshapes = iter(reshapes) if reshapes else iter(self.get_reshape(group))

        value = True

        for r, c_idx in self.get_indexed_ranges(group):

            if not r.is_dummy and c_idx == 0:

                r_sparse = r.sparse_value

                resh = next(reshapes)
                if resh is not None:
                    r_sparse = r_sparse.reshape(resh)

                value = r_sparse if value is True else value * r_sparse

        return value


def domain_comp_binary(d1, d2, group=None):

    ranges = zip(d1.get_ranges(group), d2.get_ranges(group))
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


max_domain_binary = partial(max_function_binary, domain_comp_binary)
max_domain = partial(min_max_function, max_domain_binary)

min_domain_binary = partial(min_function_binary, domain_comp_binary)
min_domain = partial(min_max_function, min_domain_binary)

scalar_domain = Domain((), (), (), (), ())


# an assumption made here is that the same coords range must run over the same dimensions of a tensor!!
def equal_domain_dims(domains):
    eq_domains = tuple(equality_range(*ranges) for ranges in zip(*(d.get_ranges() for d in domains)))
    return tuple(i for i, eq in enumerate(eq_domains) if eq), tuple(i for i, eq in enumerate(eq_domains) if not eq)


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
                t = CoordsRange

        if t == IntRange:
            return t(value=sample_int(i))
        elif t == SetRange:
            return t(value=sample_sets(i))
        elif t == SliceRange:
            return t(value=sample_sls(i))
        elif t == DummyRange:
            return DummyRange()
        elif t == CoordsRange:
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
                coos.append(sample_range(CoordsRange, cd))
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
            coos[coord1_dims] = sample_range(CoordsRange, coord1_dims)

        if coord2_dims:
            coos[coord2_dims] = sample_range(CoordsRange, coord2_dims)

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

    coords_ = (CoordsRange(value=[[3, 4, 5], [13, 14, 15]], shape=(10, 20), ndim=2),
               CoordsRange(value=[[13, 14, 15], [23, 24, 25]], ndim=2))

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

    d1 = Domain((CoordsRange(ndim=2), ), (), ((0, 1), ), (), shape=(10, 20))
    d2 = Domain((CoordsRange(ndim=2), ), (), ((0, 1), ), (), shape=(10, 20))
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