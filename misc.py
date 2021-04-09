from itertools import chain, combinations
from functools import reduce
from collections.abc import Iterable
from sortedcontainers import SortedSet


def any_pred_true(lst, predicate, *args, **kwargs):
    return any(True for item in lst if predicate(item, *args, **kwargs))


def count_pred_true(lst, predicate, *args, **kwargs):
    return sum(1 for item in lst if predicate(item, *args, **kwargs))


def indices_pred_true(lst, pred, *args, **kwargs):
    return (i for i, l in enumerate(lst) if pred(l, *args, **kwargs))


def map_indices_pred_true(ma, lst, pred, *args, **kwargs):
    return (m for m, l in zip(ma, lst) if pred(l, *args, **kwargs))


def select_from_iterable(indices, it):
    indices = iter(indices)
    cur_idx = next(indices, None)
    if cur_idx is not None:
        for i, val in enumerate(it):
            if i == cur_idx:
                yield val
                cur_idx = next(indices, None)
            if cur_idx is None:
                break


def max_function_binary(func, i1, i2, **kwargs):
    comp = func(i1, i2, **kwargs)
    if comp in (-1, 0):
        return i1
    elif comp == 1:
        return i2
    return None


def min_function_binary(func, i1, i2, **kwargs):
    comp = func(i1, i2, **kwargs)
    if comp in (0, 1):
        return i1
    elif comp == -1:
        return i2
    return None


def min_max_function(func, *items, **kwargs):

    items = iter(items)
    min_max_items = [next(items)]

    for item in items:

        is_new_min_max = True
        add_to_min_max_items = True

        for i, prev_item in enumerate(min_max_items):

            comp_item = func(prev_item, item, **kwargs)
            if comp_item is not None:
                min_max_items[i] = comp_item
                if comp_item is prev_item:
                    is_new_min_max = False
                add_to_min_max_items = False
            else:
                is_new_min_max = False

        if is_new_min_max:
            min_max_items = [item]
        elif add_to_min_max_items:
            min_max_items.append(item)

    if len(min_max_items) == 1:
        return min_max_items[0]

    return None


def equality_function_binary(func, i1, i2, **kwargs):
    return True if func(i1, i2, **kwargs) == 0 else False


def equality_function(func, *items, **kwargs):

    items = iter(items)
    i0 = next(items)

    for i in items:
        if not func(i0, i, **kwargs):
            return False
    return True


def power_set(lst):
    return chain.from_iterable(combinations(lst, r) for r in range(len(lst)+1))


def reduce_sum(*values): return reduce(lambda x, y: x + y, values, 0)


def reduce_product(*values): return reduce(lambda x, y: x * y, values, 1)


def _to_tuple(x): return tuple(x) if isinstance(x, Iterable) else (x, )


def _to_set(x): return set(x) if isinstance(x, Iterable) else {x}


def _to_sorted_set(x): return SortedSet(x) if isinstance(x, Iterable) else SortedSet({x})


def reduce_concat(*values): return reduce(lambda x, y: _to_tuple(x) + _to_tuple(y), values, ())


def reduce_union(*values): return reduce(lambda x, y: _to_set(x).union(_to_set(y)), values, set())


def reduce_union_sorted(*values): return reduce(lambda x, y: _to_sorted_set(x).union(_to_sorted_set(y)), values, SortedSet())


def is_sorted(lst): return all(lst[i] <= lst[i+1] for i in range(len(lst)-1))


def assert_sorted(lst): assert is_sorted(lst)


def stride_from_shape(shape):
    stride = []
    acc = 1
    for s in reversed(shape):
        stride.insert(0, acc)
        acc *= s
    return tuple(stride)
