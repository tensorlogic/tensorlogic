from itertools import chain, combinations
from functools import reduce
from collections.abc import Iterable
from sortedcontainers import SortedSet


# check if predicate(item, *args, **kwargs) hold for any item in lst
def any_pred_true(lst, predicate, *args, **kwargs):
    return any(True for item in lst if predicate(item, *args, **kwargs))


# count the number of items in lst where predicate(item, *args, **kwargs) holds
def count_pred_true(lst, predicate, *args, **kwargs):
    return sum(1 for item in lst if predicate(item, *args, **kwargs))


# return the indices in lst where predicate(item, *args, **kwargs) holds
def indices_pred_true(lst, pred, *args, **kwargs):
    return (i for i, l in enumerate(lst) if pred(l, *args, **kwargs))


# let func be a function which returns -1, 0, 1 if the elements i1 and i2 can be ordered(-1 if i1 is greater, 0 if they
# are equal, 1 if i2 is greater) and None if the items cannot be ordered.

# return the maximum of the two elements or return None if they are not comparable
def max_function_binary(func, i1, i2, **kwargs):
    comp = func(i1, i2, **kwargs)
    if comp in (-1, 0):
        return i1
    elif comp == 1:
        return i2
    return None


# return the minimum of the two elements or return None if they are not comparable
def min_function_binary(func, i1, i2, **kwargs):
    comp = func(i1, i2, **kwargs)
    if comp in (0, 1):
        return i1
    elif comp == -1:
        return i2
    return None


# return the maximal or minimal(depending on whether func is the min or max function above) elements of a set of items
# or return None if the elements are not comparable.
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


# return True if the elements i1 and i2 are equal(func(i1, i2) == 0) and False otherwise
def equality_function_binary(func, i1, i2, **kwargs):
    return True if func(i1, i2, **kwargs) == 0 else False


# return True if all elements in items are equal(func(i1, i2) == 0) and False otherwise
def equality_function(func, *items, **kwargs):

    items = iter(items)
    i0 = next(items)

    for i in items:
        if not func(i0, i, **kwargs):
            return False
    return True


# compute the power set of elements in lst
def power_set(lst): return chain.from_iterable(combinations(lst, r) for r in range(len(lst)+1))


def _to_tuple(x): return tuple(x) if isinstance(x, Iterable) else (x, )
def _to_set(x): return set(x) if isinstance(x, Iterable) else {x}
def _to_sorted_set(x): return SortedSet(x) if isinstance(x, Iterable) else SortedSet({x})


# various reductions of lists
def reduce_sum(*values): return reduce(lambda x, y: x + y, values, 0)
def reduce_product(*values): return reduce(lambda x, y: x * y, values, 1)
def reduce_concat(*values): return reduce(lambda x, y: _to_tuple(x) + _to_tuple(y), values, ())
def reduce_union(*values): return reduce(lambda x, y: _to_set(x).union(_to_set(y)), values, set())
def reduce_union_sorted(*values): return reduce(lambda x, y: _to_sorted_set(x).union(_to_sorted_set(y)), values, SortedSet())
