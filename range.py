import torch
import sparse as sp
import numpy as np
from functools import partial, reduce

from misc import equality_function_binary, equality_function, min_function_binary, max_function_binary, min_max_function


def _check_type(item, t): return isinstance(item, t) or item == t


class Range(object):

    """
    A Range object is used to index into one or multiple dimensions(CoordRange) of a tensor.
    The values of the range will specify the locations in the tensor dimensions which have data or locations which
    we are querying. From a logic programming stand point, ranges can be though of as atoms.
    A range can have its value set from certain types of python/numpy objects(this is usually done during compilation
    or the run stage of the program, with the user setting these values). These values are referred to as "values".
    Ranges can be used to perform operations like union or intersection. To perform these types of operations,
    the values will be cast to sparse.COO objects. The coordinates of the sparse.COO array will be correspond to the
    locations along the tensor dimensions that the Range is indexing into and the values in the sparse.COO object will
    be the boolean True. The sparse values are not manipulated by the user and are referred to as "sparse_values".
    The transformation between the two types of values is done internally. A Range which doesn't
    have a set value or which after an operation like intersection turns out to be empty will have the value and
    sparse_value None. The shape of a Range refers to the shape of the tensor dimensions the range indexes into.
    """

    id_string = None

    is_int = None   # True only of IntRange
    is_coord = None  # True only for CoordRange
    is_dense = None   # True for "dense-like" Ranges: SliceRange, DummyRange
    is_sparse = None  # True for "sparse-like" Ranges: SetRange, CoordRange
    is_dummy = None   # True only for DummyRange

    def __init__(self, value=None, shape=None):
        """
        Create a Range Object.
        :param value: type of value depends on the type of Range(optional)
            value of the Range, i.e. the locations in the tensor dimension that the Range if referring to.
        :param shape: tuple(optional)
            shape of tensor dimensions that the Range is indexing into.
        """
        self._value = None
        self._sparse_value = None

        self.shape = shape

        self.empty = True
        if value is not None:
            self.value = value

    def reset(self):
        """
        Reset the Range values and set to empty.
        :return:
        """
        self._value = None
        self._sparse_value = None
        self.empty = True

    @property
    def data_shape(self):
        """
        Return the number of locations in the tensor that the range is referring to. For example, for a SetIndex,
        this would be the length of the set. For a SliceIndex, it will be the number of values in the slice.
        :return: int, None
        """
        raise NotImplemented

    @property
    def value(self):
        """
        Get the value of the Range. Each subclass will implement its own function for transforming the sparse value
        into the value and this will be used when the sparse value is available and the value is not
        :return: type of value depends on the type of Range.
        """
        if self._value is None and self._sparse_value is not None:
            self._value = self._value_from_sparse_value()

        if self._value is not None:
            return self._value

        raise ValueError("Value of range is not set")

    @value.setter
    def value(self, v):
        """
        Set the value of the range. Value set to None will set the range to empty. Each subclass will implement its
        own functions formatting the value.
        :return:
        """
        self._value = self._format_value(v)
        self._sparse_value = None
        self.empty = self._value is None

    @property
    def sparse_value(self):
        """
        Get the sparse value of the Range. Each subclass will implement its own function for transforming the
        value into the sparse value and this will be used when the value is available and the sparse value is not.
        :return: sparse.COO
        """
        if self._sparse_value is None and self._value is not None:
            self._sparse_value = self._sparse_value_from_value()

        if self._sparse_value is not None:
            return self._sparse_value

        raise ValueError("Value of range is not set")

    @sparse_value.setter
    def sparse_value(self, v):
        """
        Set the sparse value of the range. This should not be called by the user directly, but should only be set
        during operations like union or intersection. Cannot be set to None, but the format_sparse_value function
        should detect an empty range from the sparse value and return None which will set the Range to empty.
        Each subclass will implement its own functions formatting the sparse value.
        :return:
        """
        self._sparse_value = self._format_sparse_value(v)
        self._value = None
        self.empty = self._sparse_value is None

    def index(self, c_idx=0):
        """
        Given the value or sparse value(the function should handle both cases) of the Range, return an index into a
        numpy/torch tensor that will access the data at the locations the range corresponds to. c_idx is used only
        for CoordRanges which index into multiple tensor dimensions
        :param c_idx: int
            Not required by any Range subclass, except CoordRanges. In the latter case, c_idx will select the dimension
            of the multi-index.
        :return: type of value depends on the type of Range
            Index into a numpy/torch tensor that will access the data at the locations the range corresponds to
        """
        raise NotImplemented

    def _format_value(self, v):
        """
        Formats v. Returns None if v represents something empty like. Each subclass implements this.
        :param v: type depends on the type of Range
            The value to be formatted into the range value.
        :return: type depends on the type of Range
            Formatted value, None if v was empty-like.
        """
        raise NotImplemented

    def _format_sparse_value(self, v):
        """
        Formats v. Returns None if v is an empty sparse.COO.
        :param v: sparse.COO
            The sparse value to be formatted into the range value.
        :return: sparse.COO
            Formatted sparse value, None if v was empty-like
        """
        raise NotImplemented

    def _value_from_sparse_value(self):
        """
        Return a value from the already existing sparse value of the Range.
        :return: type depends on the type of Range
            Value to set the Range to.
        """
        raise NotImplemented

    def _sparse_value_from_value(self):
        """
        Return a sparse value from the already existing value of the Range.
        :return: sparse.COO
            Sparse value to set the Range to.
        """
        raise NotImplemented

    def set_value_from_coords(self, coords):
        """
        Set the value from a subset of dimensions of a sparse.COO array. The result of operations like union and
        intersection is generally a sparse array. Sometimes, we want to set the value of a range from a subset of
        the dimensions of this sparse array, and to do so, we select the relevant dimensions of the coordinates of
        the sparse array.
        :param coords: np.ndarray
            Coordinates selected from a subset of dimensions of a sparse.COO array
        :return:
        """
        raise NotImplemented

    @classmethod
    def union_type_binary(cls, other):
        """
        Return the Range type resulting from the union of two Ranges.
        :param other: class, Range
        :return: class
        """
        raise NotImplemented

    @classmethod
    def intersection_type_binary(cls, other):
        """
        Return the Range type resulting from the intersection of two Ranges.
        :param other: class, Range
        :return: class
        """
        raise NotImplemented

    @staticmethod
    def init_range(id_string, **kwargs):
        """
        Creates a Range object from the parsed dictionary obtained when parsing the TensorLogic program.
        The correct type of the Range to create is determined by the id_string.
        :param id_string: str
            Unique string identifier for each Range type
        :param kwargs: dict
            Parameters parsed which will be used for initialization
        :return: Range
        """
        if id_string in IntRange.id_string:
            return IntRange(**kwargs)
        elif id_string in SetRange.id_string:
            return SetRange(**kwargs)
        elif id_string in SliceRange.id_string:
            return SliceRange(**kwargs)
        elif id_string in CoordRange.id_string:
            return CoordRange(**kwargs)
        else:
            raise ValueError("Unknown id_string", id_string)


class IntRange(Range):

    """
    The IntRange corresponds to a single integer location along the dimension of a tensor.
    The value will be an integer or None if the Range is empty.
    The sparse value will be a single dimensional sparse.COO object with a single coordinate corresponding to the
    integer value. The shape corresponds to the tensor dimension the Range is indexing into.
    """
    id_string = "int"

    is_int = True
    is_coord = False
    is_dense = False
    is_sparse = False
    is_dummy = False

    def __init__(self, value=None, shape=None):
        super(IntRange, self).__init__(value, shape)

    @property
    def data_shape(self): return None

    def set_value_from_coords(self, coords):
        if coords.size >= 1:
            self.value = coords[0].item()
        else:
            self.value = None

    def index(self, c_idx=0): return self.value

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

    def _value_from_sparse_value(self):
        return self._sparse_value.coords.item()

    def _sparse_value_from_value(self):
        return sp.COO(coords=np.array([[self._value]], dtype=np.long), data=True, shape=(self.shape,))

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
        elif _check_type(other, CoordRange):
            return CoordRange
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
        elif _check_type(other, CoordRange):
            return IntRange
        else:
            raise ValueError


class SetRange(Range):
    """
    The SetRange corresponds to a set(no repetitions) of locations along the dimension of a tensor.
    The value will be a 1d np.ndarray or None if the Range is empty.
    The sparse value will be a single dimensional sparse.COO object with coordinates corresponding to the
    set of locations the Range is indexing. The shape corresponds to the tensor dimension the Range is indexing into.
    """

    id_string = "set"

    is_int = False
    is_coord = False
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

    def set_value_from_coords(self, coords):
        self.value = coords

    def index(self, c_idx=0): return self.value

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

    def _value_from_sparse_value(self):
        return self._sparse_value.coords[0]

    def _sparse_value_from_value(self):
        return sp.COO(coords=np.expand_dims(self._value, axis=0), data=True, shape=(self.shape,))

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
        elif _check_type(other, CoordRange):
            return CoordRange
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
        elif _check_type(other, CoordRange):
            return CoordRange
        else:
            raise ValueError


class SliceRange(Range):
    """
    The SliceRange corresponds to a slice along the dimension of a tensor.
    The value will be a python slice object or None if the Range is empty.
    The sparse value will be a single dimensional sparse.COO object with coordinates corresponding to the
    locations the slice is indexing. The shape corresponds to the tensor dimension the Range is indexing into.
    """

    id_string = "slice"

    is_int = False
    is_coord = False
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

    def set_value_from_coords(self, coords): raise NotImplementedError

    def index(self, c_idx=0): return self.value

    def _format_value(self, v):

        if v is None:
            return None

        v = slice(v.start or 0, v.stop or self.shape, v.step or 1)

        if v.stop > v.start and v.step > 0:
            return v

        return None

    def _format_sparse_value(self, v): raise NotImplementedError
    def _value_from_sparse_value(self): raise NotImplementedError

    def _sparse_value_from_value(self):
        return sp.COO(coords=np.expand_dims(np.arange(self._value.start, self._value.stop, self._value.step), axis=0),
                      data=True, shape=(self.shape,))

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
        elif _check_type(other, CoordRange):
            return CoordRange
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
        elif _check_type(other, CoordRange):
            return CoordRange
        else:
            raise ValueError


class CoordRange(Range):
    """
    The CoordRange corresponds to coordinates along an arbitrary subset of tensor dimensions. The coordinates are not
    necessarily unique. However, non-unique coordinates should only be created when a subset of dimensions from a
    CoordRange with unique coords is selected and the user should only create CoordRanges with unique coordinates.
    The value will be a n-dimensional np.ndarray or None if the Range is empty.
    The sparse value will be a n-dimensional sparse.COO object with coordinates corresponding to the
    locations the Range is indexing. The shape corresponds to the tensor dimensions the Range is indexing into.
    The number of dimensions is specified by ndim.
    The other main difference from the other types of ranges is that we can select a subset of the dimensions
    of a CoordRange. This will create a child CoordRange. The child Range will not have unqiue coordinates and
    will only have its value updated when the parent CoordRange gets updated. If a CoordRange is create in such a way,
    the is_child flag is set to True.
    """

    id_string = "coords"

    is_int = False
    is_coord = True
    is_dense = False
    is_sparse = True
    is_dummy = False

    def __init__(self, ndim, value=None, shape=None, is_child=False):
        """
        :param ndim: int
        :param value: np.ndarray(optional)
        :param shape: tuple(optional)
        :param is_child: bool
        """

        self.ndim = ndim
        self.is_child = is_child

        # a dictionary with the child ranges of this range, indexed by the subset of dimensions to which the child range
        # corresponds
        self.child_ranges = {}

        super(CoordRange, self).__init__(value, shape)

    @property
    def data_shape(self):
        if self._value is not None:
            return self._value.shape[1]
        elif self._sparse_value is not None:
            return self._sparse_value.coords.shape[1]

    def set_value_from_coords(self, coords): self.value = coords

    def index(self, c_idx=0):
        return self.value[c_idx]

    def _set_child_ranges_from_value(self, value):
        for dims, r in self.child_ranges.items():
            r.value = value[list(dims)]

    def _set_child_ranges_from_sparse_value(self, sparse_value):
        for dims, r in self.child_ranges.items():
            r.value = sparse_value.coords[list(dims)]

    def _set_child_ranges_to_empty(self):
        for r in self.child_ranges.values():
            r.value = None

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

    def _sparse_value_from_value(self):
        return sp.COO(coords=self._value, data=True, shape=self.shape)

    def _value_from_sparse_value(self):
        return self._sparse_value.coords

    @classmethod
    def union_type_binary(cls, other):
        if _check_type(other, DummyRange):
            return CoordRange
        elif _check_type(other, IntRange):
            return CoordRange
        elif _check_type(other, SetRange):
            return CoordRange
        elif _check_type(other, SliceRange):
            return CoordRange
        elif _check_type(other, CoordRange):
            return CoordRange
        else:
            raise ValueError

    @classmethod
    def intersection_type_binary(cls, other):

        if _check_type(other, DummyRange):
            return CoordRange
        elif _check_type(other, IntRange):
            return IntRange
        elif _check_type(other, SetRange):
            return CoordRange
        elif _check_type(other, SliceRange):
            return CoordRange
        elif _check_type(other, CoordRange):
            return CoordRange
        else:
            raise ValueError

    def select(self, dims):
        """
        Create a child Range which is a new CoordRange from a subset of the dimensions of this CoordRange. The child
        Range will not have unique coordinates and will only have its value updated when the parent CoordRange gets
        updated.
        :param dims: tuple, list
        :return: CoordRange
        """
        dims = tuple(dims)

        # when we select all dimensions, return self
        if len(dims) == self.ndim:
            return self

        # if a child range for the current subset is already a child, return it
        if dims in self.child_ranges:
            return self.child_ranges[dims]

        # otherwise, create a new child range.
        shape = None if self.shape is None else tuple(self.shape[idx] for idx in dims)
        r = CoordRange(ndim=len(dims), shape=shape, is_child=True)

        if self._value is not None:
            r.set_value_from_coords(self._value[list(dims)])
        elif self._sparse_value is not None:
            r.set_value_from_coords(self._sparse_value.coords[list(dims)])

        self.child_ranges[dims] = r

        return r


class DummyRange(Range):
    """
    The DummyRange corresponds to the a whole tensor dimensions. Or an ellipsis in python parlance.
    The value will be a python slice object with all Nones. A dummy range cannot be empty.
    A dummy range does not have a sparse value.
    """

    is_int = False
    is_coord = False
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

    def index(self, c_idx=0): return dummy_value

    @classmethod
    def union_type_binary(cls, other): return CoordRange if _check_type(other, CoordRange) else DummyRange
    @classmethod
    def intersection_type_binary(cls, other): return type(other)


dummy_value = slice(None, None, None)


# compare two ranges. For now, equality only holds between a range compared with itself or between two dummy ranges.
# a dummy range always contains everything else.
# return -1, 0, 1 if the r1 and r2 can be ordered(-1 if r1 is greater, 0 if they are equal, 1 if r2 is greater) and
# None if the items cannot be ordered.
def range_comp_binary(r1, r2):

    if r1 is r2 or (isinstance(r1, DummyRange) and isinstance(r2, DummyRange)):
        return 0
    elif isinstance(r1, DummyRange):
        return -1
    elif isinstance(r2, DummyRange):
        return 1
    return None


# compute the out-of-place union and intersection types of multiple ranges
def range_union_type(*ranges): return reduce(lambda x, y: x.union_type_binary(y), ranges)
def range_intersection_type(*ranges): return reduce(lambda x, y: x.intersection_type_binary(y), ranges)


# create comparison functions between ranges
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
    coo, coo_val, coo_coord_val = CoordRange(shape=(10, 20), ndim=2), \
                                  np.array([[0, 1, 2], [10, 11, 12]]), np.array([[0, 1, 2], [10, 11, 12]])
    du, du_val, du_coord_val = DummyRange(shape=10), None, None

    rans = (i, se, sl, coo, du)
    ran_vals = (i_val, se_val, sl_val, coo_val, du_val)
    ran_coord_vals = (i_coord_val, se_coord_val, sl_coord_val, coo_coord_val, du_coord_val)

    def test_range(r, val, coord_val):

        if val is not None:

            r.value = val

            new_r = type(r)(shape=r.shape, ndim=r.ndim) if isinstance(r, CoordRange) else type(r)(shape=r.shape)
            new_r.value = r.value

            assert r.value == new_r.value if isinstance(r, SliceRange) else np.allclose(r.value, new_r.value)
            val = r.sparse_value == new_r.sparse_value
            assert val.fill_value and val.nnz == 0

            if not isinstance(r, SliceRange):

                new_r = type(r)(shape=r.shape, ndim=r.ndim) if isinstance(r, CoordRange) else type(r)(shape=r.shape)
                new_r.sparse_value = r.sparse_value

                print(r.value, new_r.value, r.sparse_value, new_r.sparse_value)
                assert np.allclose(r.value, new_r.value)
                val = r.sparse_value == new_r.sparse_value
                assert val.fill_value and val.nnz == 0

                new_r = type(r)(shape=r.shape, ndim=r.ndim) if isinstance(r, CoordRange) else type(r)(shape=r.shape)
                new_r.set_value_from_coords(coord_val)
                assert np.allclose(r.value, new_r.value)
                val = r.sparse_value == new_r.sparse_value
                assert val.fill_value and val.nnz == 0

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

    c_ = CoordRange(3, sp.random((10, 20, 30)).coords, (10, 20, 30))

    c0 = c_.select((0, ))
    c1 = c_.select((1, ))
    c2 = c_.select((0, ))
    c01 = c_.select((0, 1))

    assert c0 is c2

    print(c0.value, c0.sparse_value)
    print(c1.value, c1.sparse_value)
    print(c01.value, c01.sparse_value,)

    c_.value = sp.random((10, 20, 30)).coords

    print(c0.value, c0.sparse_value)
    print(c1.value, c1.sparse_value)
    print(c01.value, c01.sparse_value,)