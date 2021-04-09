import torch
import sparse as sp
import numpy as np
from functools import partial, reduce

from misc import equality_function_binary, equality_function, min_function_binary, max_function_binary, min_max_function


def _check_type(item, t): return isinstance(item, t) or item == t


class Range(object):

    id_string = None

    is_int = None
    is_slice = None
    is_dense = None
    is_sparse = None
    is_dummy = None

    def __init__(self, value=None, shape=None):

        self._value = None
        self._sparse_value = None
        self._torch_value = None

        self.shape = shape
        self.empty = None

        if value is not None:
            self.value = value
        else:
            self.empty = True

    def reset(self):
        self._value = None
        self._sparse_value = None
        self._torch_value = None
        self.empty = True

    @property
    def data_shape(self): raise NotImplemented

    @property
    def value(self):

        if self._value is None and self._sparse_value is not None:
            self._value = self._value_from_sparse_value()

        if self._value is not None:
            return self._value

        raise ValueError("Value of range is not set")

    @value.setter
    def value(self, v):

        self._value = self._format_value(v)
        self._sparse_value = None
        self._torch_value = None
        self.empty = self._value is None

    @property
    def sparse_value(self):

        if self._sparse_value is None and self._value is not None:
            self._sparse_value = self._sparse_value_from_value()

        if self._sparse_value is not None:
            return self._sparse_value

        raise ValueError("Value of range is not set")

    @sparse_value.setter
    def sparse_value(self, v):
        self._sparse_value = self._format_sparse_value(v)
        self._value = None
        self._torch_value = None
        self.empty = self._sparse_value is None

    @property
    def torch_value(self):

        if self._torch_value is None:
            if self._value is not None:
                self._torch_value = self._torch_value_from_value()
            elif self._sparse_value is not None:
                self._torch_value = self._torch_value_from_sparse_value()
            else:
                raise ValueError("Cannot set torch value without value or sparse value already set")

        if self._torch_value is not None:
            return self._torch_value

        raise ValueError("Value of range is not set")

    def _format_value(self, v): raise NotImplemented
    def _format_sparse_value(self, v): raise NotImplemented
    def _value_from_sparse_value(self): raise NotImplemented
    def _sparse_value_from_value(self): raise NotImplemented
    def _torch_value_from_value(self): raise NotImplemented
    def _torch_value_from_sparse_value(self): raise NotImplemented
    def set_value_from_coords(self, coords): raise NotImplemented

    @classmethod
    def union_type_binary(cls, other): raise NotImplemented
    @classmethod
    def intersection_type_binary(cls, other): raise NotImplemented

    @staticmethod
    def init_range(id_string, **kwargs):

        if id_string in IntRange.id_string:
            return IntRange(**kwargs)
        elif id_string in SetRange.id_string:
            return SetRange(**kwargs)
        elif id_string in SliceRange.id_string:
            return SliceRange(**kwargs)
        elif id_string in CoordsRange.id_string:
            return CoordsRange(**kwargs)
        else:
            raise ValueError("Unknown id_string", id_string)

    def index(self, c_idx=0): raise NotImplemented


class IntRange(Range):

    id_string = "int"

    is_int = True
    is_slice = True
    is_dense = False
    is_sparse = False
    is_dummy = False

    def __init__(self, value=None, shape=None):
        super(IntRange, self).__init__(value, shape)

    @property
    def data_shape(self): return ()

    def _format_value(self, v):

        if isinstance(v, (int, np.ndarray, torch.Tensor, np.integer)):
            return int(v)
        elif v is None:
            return None

        raise ValueError

    def _format_sparse_value(self, v):

        if v.nnz == 1:
            return v
        elif v.nnz == 0:
            return None

        raise ValueError

    def _value_from_sparse_value(self): return self._sparse_value.coords.item()
    def _sparse_value_from_value(self): return sp.COO(coords=np.array([[self._value]], dtype=np.long), data=True, shape=(self.shape,))
    def _torch_value_from_value(self): return torch.tensor(self._value, dtype=torch.long)
    def _torch_value_from_sparse_value(self): return torch.tensor(self._sparse_value.coords.item(), dtype=torch.long)

    def set_value_from_coords(self, coords):
        if coords.size >= 1:
            self.value = coords[0].item()
        else:
            self.value = None

    @classmethod
    def union_type_binary(cls, other):
        if _check_type(other, DummyRange):
            return DummyRange
        elif _check_type(other, IntRange):
            return SetRange
        elif _check_type(other, SetRange):
            return SetRange
        elif _check_type(other, SliceRange):
            return SetRange
        elif _check_type(other, CoordsRange):
            return CoordsRange
        else:
            raise ValueError

    @classmethod
    def intersection_type_binary(cls, other):
        if _check_type(other, DummyRange):
            return IntRange
        elif _check_type(other, IntRange):
            return IntRange
        elif _check_type(other, SetRange):
            return IntRange
        elif _check_type(other, SliceRange):
            return IntRange
        elif _check_type(other, CoordsRange):
            return IntRange
        else:
            raise ValueError

    def index(self, c_idx=0): return self.value


class SetRange(Range):

    id_string = "set"

    is_int = False
    is_slice = True
    is_dense = False
    is_sparse = True
    is_dummy = False

    def __init__(self, value=None, shape=None):
        super(SetRange, self).__init__(value, shape)

    @property
    def data_shape(self):
        if self._value is not None:
            return self._value.shape[0]
        elif self._sparse_value is not None:
            return self._sparse_value.coords.shape[1]

    def _format_value(self, v):

        if v is None:
            return None

        v = np.array(v, dtype=np.long)

        if v.size:
            return v
        return None

    def _format_sparse_value(self, v):

        if v.nnz:
            return v
        return None

    def _value_from_sparse_value(self): return self._sparse_value.coords[0]
    def _sparse_value_from_value(self): return sp.COO(coords=np.expand_dims(self._value, axis=0), data=True, shape=(self.shape,))
    def _torch_value_from_value(self): return torch.tensor(self._value, dtype=torch.long)
    def _torch_value_from_sparse_value(self): return torch.tensor(self._sparse_value.coords[0], dtype=torch.long)
    def set_value_from_coords(self, coords): self.value = coords

    @classmethod
    def union_type_binary(cls, other):
        if _check_type(other, DummyRange):
            return DummyRange
        elif _check_type(other, IntRange):
            return SetRange
        elif _check_type(other, SetRange):
            return SetRange
        elif _check_type(other, SliceRange):
            return SetRange
        elif _check_type(other, CoordsRange):
            return CoordsRange
        else:
            raise ValueError

    @classmethod
    def intersection_type_binary(cls, other):

        if _check_type(other, DummyRange):
            return SetRange
        elif _check_type(other, IntRange):
            return IntRange
        elif _check_type(other, SetRange):
            return SetRange
        elif _check_type(other, SliceRange):
            return SetRange
        elif _check_type(other, CoordsRange):
            return CoordsRange
        else:
            raise ValueError

    def index(self, c_idx=0): return self.value


class SliceRange(Range):

    id_string = "slice"

    is_int = False
    is_slice = True
    is_dense = True
    is_sparse = False
    is_dummy = False

    def __init__(self, value=None, shape=None):
        super(SliceRange, self).__init__(value, shape)

    @property
    def data_shape(self):
        if self._value is not None:
            return (self._value.stop - self._value.start + (self._value.step - 1)) // self._value.step
        elif self._sparse_value is not None:
            return self._sparse_value.coords.shape[1]

    @property
    def torch_value(self):
        if self._torch_value is None:
            self._torch_value = torch.arange(self.value.start, self.value.stop, self.value.step)
        return self._torch_value

    def _format_value(self, v):

        if v is None:
            return None

        v = slice(v.start or 0, v.stop or self.shape, v.step or 1)

        if v.stop > v.start and v.step > 0:
            return v

        return None

    def _format_sparse_value(self, v): raise NotImplementedError
    def _value_from_sparse_value(self): raise NotImplementedError
    def _sparse_value_from_value(self): return sp.COO(coords=np.expand_dims(np.arange(self._value.start, self._value.stop, self._value.step), axis=0), data=True, shape=(self.shape,))
    def _torch_value_from_value(self): raise NotImplementedError
    def _torch_value_from_sparse_value(self): raise NotImplementedError
    def set_value_from_coords(self, coords): raise NotImplementedError

    @classmethod
    def union_type_binary(cls, other):
        if _check_type(other, DummyRange):
            return DummyRange
        elif _check_type(other, IntRange):
            return SetRange
        elif _check_type(other, SetRange):
            return SetRange
        elif _check_type(other, SliceRange):
            return SetRange
        elif _check_type(other, CoordsRange):
            return CoordsRange
        else:
            raise ValueError

    @classmethod
    def intersection_type_binary(cls, other):

        if _check_type(other, DummyRange):
            return SliceRange
        elif _check_type(other, IntRange):
            return IntRange
        elif _check_type(other, SetRange):
            return SetRange
        elif _check_type(other, SliceRange):
            return SetRange
        elif _check_type(other, CoordsRange):
            return CoordsRange
        else:
            raise ValueError

    def index(self, c_idx=0): return self.value


class CoordsRange(Range):

    id_string = "coords"

    is_int = False
    is_slice = False
    is_dense = False
    is_sparse = True
    is_dummy = False

    def __init__(self, ndim, value=None, shape=None, is_child=None):

        self.ndim = ndim
        self.is_child = is_child
        self.child_ranges = {}
        super(CoordsRange, self).__init__(value, shape)

    @property
    def data_shape(self):
        if self._value is not None:
            return self._value.shape[1]
        elif self._sparse_value is not None:
            return self._sparse_value.coords.shape[1]

    @staticmethod
    def _numpy_set_child_from_value(value, dims):
        return value[list(dims)]

    @staticmethod
    def _numpy_set_child_from_sparse_value(sparse_value, dims):
        return sparse_value.coords[list(dims)]

    def _set_child_ranges_from_value(self, value):
        for dims, r in self.child_ranges.items():
            r.set_value_from_coords(self._numpy_set_child_from_value(value, dims))

    def _set_child_ranges_from_sparse_value(self, value):
        for dims, r in self.child_ranges.items():
            r.set_value_from_coords(self._numpy_set_child_from_sparse_value(value, dims))

    def _set_child_ranges_to_empty(self):
        for r in self.child_ranges.values():
            r.reset()

    def _format_value(self, v):

        if v is None:
            self._set_child_ranges_to_empty()
            return v

        v = np.array(v, dtype=np.long)

        if v.size:
            self._set_child_ranges_from_value(v)
            return v

        self._set_child_ranges_to_empty()
        return None

    def _format_sparse_value(self, v):

        if v.nnz:

            self._set_child_ranges_from_sparse_value(v)
            return v

        self._set_child_ranges_to_empty()
        return None

    def _sparse_value_from_value(self): return sp.COO(coords=self._value, data=True, shape=self.shape)
    def _value_from_sparse_value(self): return self._sparse_value.coords

    def _torch_value_from_value(self): return torch.tensor(self._value, dtype=torch.long)
    def _torch_value_from_sparse_value(self): return torch.tensor(self._sparse_value, dtype=torch.long)

    def set_value_from_coords(self, coords): self.value = coords

    @classmethod
    def union_type_binary(cls, other):
        if _check_type(other, DummyRange):
            return CoordsRange
        elif _check_type(other, IntRange):
            return CoordsRange
        elif _check_type(other, SetRange):
            return CoordsRange
        elif _check_type(other, SliceRange):
            return CoordsRange
        elif _check_type(other, CoordsRange):
            return CoordsRange
        else:
            raise ValueError

    @classmethod
    def intersection_type_binary(cls, other):

        if _check_type(other, DummyRange):
            return CoordsRange
        elif _check_type(other, IntRange):
            return IntRange
        elif _check_type(other, SetRange):
            return CoordsRange
        elif _check_type(other, SliceRange):
            return CoordsRange
        elif _check_type(other, CoordsRange):
            return CoordsRange
        else:
            raise ValueError

    def select(self, dims):

        dims = tuple(dims)

        if len(dims) == self.ndim:
            return self

        if dims in self.child_ranges:
            return self.child_ranges[dims]

        shape = None if self.shape is None else tuple(self.shape[idx] for idx in dims)
        r = CoordsRange(ndim=len(dims), shape=shape, is_child=True)

        if self._value is not None:
            r.set_value_from_coords(self._numpy_set_child_from_value(self._value, dims))
        elif self._sparse_value is not None:
            r.set_value_from_coords(self._numpy_set_child_from_sparse_value(self._sparse_value, dims))

        self.child_ranges[dims] = r

        return r

    def project(self):

        if self._value is not None:
            return np.unique(self._value, axis=1).squeeze(axis=0) if self.ndim == 1 else np.unique(self._value, axis=1)
        elif self._sparse_value is not None:
            return np.unique(self._sparse_value.coords, axis=1).squeeze(axis=0) if self.ndim == 1 else np.unique(self._value, axis=1)
        else:
            raise ValueError

    def index(self, c_idx=0): return self.value[c_idx]


class DummyRange(Range):

    is_int = False
    is_slice = True
    is_dense = True
    is_sparse = False
    is_dummy = True

    def __init__(self, shape=None):

        super(DummyRange, self).__init__(None, shape)
        self.empty = False

    @property
    def data_shape(self): return self.shape
    @property
    def value(self): return dummy_value
    @value.setter
    def value(self, v): raise NotImplementedError
    @property
    def sparse_value(self): raise NotImplementedError
    @sparse_value.setter
    def sparse_value(self, v): raise NotImplementedError

    def set_value_from_coords(self, coords): raise NotImplementedError

    @classmethod
    def union_type_binary(cls, other): return CoordsRange if _check_type(other, CoordsRange) else DummyRange
    @classmethod
    def intersection_type_binary(cls, other): return type(other)

    def index(self, c_idx=0): return dummy_value


dummy_value = slice(None, None, None)


def range_comp_binary(r1, r2):

    if r1 is r2 or (isinstance(r1, DummyRange) and isinstance(r2, DummyRange)):
        return 0
    elif isinstance(r1, DummyRange):
        return -1
    elif isinstance(r2, DummyRange):
        return 1
    return None


def range_union_type(*ranges): return reduce(lambda x, y: x.union_type_binary(y), ranges)
def range_intersection_type(*ranges): return reduce(lambda x, y: x.intersection_type_binary(y), ranges)


equality_range_binary = partial(equality_function_binary, range_comp_binary)
equality_range = partial(equality_function, equality_range_binary)
min_range_binary = partial(min_function_binary, range_comp_binary)
min_range = partial(min_max_function, min_range_binary)
max_range_binary = partial(max_function_binary, range_comp_binary)
max_range = partial(min_max_function, max_range_binary)


if __name__ == "__main__":

    import itertools

    i, i_val, i_coord_val = IntRange(shape=10), 1, np.array([1, 1, 1])
    se, se_val, se_coord_val = SetRange(shape=10), [1, 3, 5], np.array([1, 3, 5])
    sl, sl_val, sl_coord_val = SliceRange(shape=10), slice(None, 5), None
    coo, coo_val, coo_coord_val = CoordsRange(shape=(10, 20), ndim=2), \
                                  np.array([[0, 1, 2], [10, 11, 12]]), np.array([[0, 1, 2], [10, 11, 12]])
    du, du_val, du_coord_val = DummyRange(shape=10), None, None

    rans = (i, se, sl, coo, du)
    ran_vals = (i_val, se_val, sl_val, coo_val, du_val)
    ran_coord_vals = (i_coord_val, se_coord_val, sl_coord_val, coo_coord_val, du_coord_val)

    def test_range(r, val, coord_val):

        if val is not None:

            r.value = val

            new_r = type(r)(shape=r.shape, ndim=r.ndim) if isinstance(r, CoordsRange) else type(r)(shape=r.shape)
            new_r.value = r.value

            assert r.value == new_r.value if isinstance(r, SliceRange) else np.allclose(r.value, new_r.value)
            val = r.sparse_value == new_r.sparse_value
            assert val.fill_value and val.nnz == 0

            if not isinstance(r, SliceRange):

                assert np.allclose(r.torch_value, new_r.torch_value)

                new_r = type(r)(shape=r.shape, ndim=r.ndim) if isinstance(r, CoordsRange) else type(r)(shape=r.shape)
                new_r.sparse_value = r.sparse_value

                print(r.value, new_r.value, r.sparse_value, new_r.sparse_value)
                assert np.allclose(r.value, new_r.value)
                val = r.sparse_value == new_r.sparse_value
                assert val.fill_value and val.nnz == 0
                assert np.allclose(r.torch_value, new_r.torch_value)

                new_r = type(r)(shape=r.shape, ndim=r.ndim) if isinstance(r, CoordsRange) else type(r)(shape=r.shape)
                new_r.set_value_from_coords(coord_val)
                assert np.allclose(r.value, new_r.value)
                val = r.sparse_value == new_r.sparse_value
                assert val.fill_value and val.nnz == 0
                assert np.allclose(r.torch_value, new_r.torch_value)


    for (r, v, coo_v) in zip(rans, ran_vals, ran_coord_vals):
        test_range(r, v, coo_v)

    for (r1, r2) in itertools.product(rans, rans):
        print(type(r1).__name__, "intersect", type(r2).__name__, "=", range_intersection_type(r1, r2).__name__)
        print(type(r1).__name__, "union", type(r2).__name__, "=", range_union_type(r1, r2).__name__)

    assert range_comp_binary(i, i) == 0
    assert range_comp_binary(i, se) is None
    assert range_comp_binary(i, du) == 1
    assert range_comp_binary(du, i) == -1
    assert range_comp_binary(du, du) == 0

    assert equality_range(du, du, du)
    assert equality_range(se, se, se)
    assert not equality_range(du, i, i)
    assert not equality_range(i, se, sl)
    assert equality_range(i, i, i)
    assert equality_range(i)

    assert min_range(i, i) is i
    assert min_range(i, du) is i
    assert min_range(du, i, du) is i
    assert min_range(se, sl) is None
    assert min_range(du, du, du) is du

    assert max_range(i, i) is i
    assert max_range(i, du) is du
    assert max_range(du, i, du) is du
    assert max_range(se, sl) is None
    assert max_range(du, du, du) is du

    c_ = CoordsRange(3, sp.random((10, 20, 30)).coords, (10, 20, 30))

    c0 = c_.select((0, ))
    c1 = c_.select((1, ))
    c2 = c_.select((0, ))
    c01 = c_.select((0, 1))

    assert c0 is c2

    print(c0.value, c0.sparse_value, c0.torch_value)
    print(c1.value, c1.sparse_value, c1.torch_value)
    print(c01.value, c01.sparse_value, c01.torch_value)

    c_.value = sp.random((10, 20, 30)).coords

    print(c0.value, c0.sparse_value, c0.torch_value)
    print(c1.value, c1.sparse_value, c1.torch_value)
    print(c01.value, c01.sparse_value, c01.torch_value)