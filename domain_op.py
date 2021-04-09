from functools import reduce
from sortedcontainers import SortedDict

from domain import Domain, min_domain, generate_domains
from range import CoordsRange, range_intersection_type, min_range
from misc import reduce_union


def _assert_same_shape(domains):
    d0_shape = domains[0].shape
    assert all(d0_shape == d.shape for d in domains[1:])


def _get_reshapes(groups, domains):
    return tuple(tuple(d.get_reshape(group) for d in domains) for group in groups)


class IntersectionOp(object):

    @staticmethod
    def intersection_sparse(*values):
        return reduce(lambda x, y: x & y, values) if values else True

    @staticmethod
    def _get_slice_type_and_min_domain(domains, i):
        min_sl = min_range(*(domain.indexed_ranges[i][0] for domain in domains))
        if min_sl:
            return i, min_sl, None
        else:
            return i, None, range_intersection_type(*(domain.indexed_ranges[i][0] for domain in domains))

    @staticmethod
    def _intersection_groups(domains):
        coord_group = reduce_union(*(dims for domain in domains for dims in domain.coord_dims))
        slice_dims_and_types = (IntersectionOp._get_slice_type_and_min_domain(domains, i)
                                for i in range(domains[0].ndim) if i not in coord_group)
        return coord_group, slice_dims_and_types

    @staticmethod
    def _get_group_sparse_values(domains, group, group_reshapes):
        for domain, reshape in zip(domains, group_reshapes):
            yield domain.group_ranges(group, reshape)

    @staticmethod
    def _get_slice_sparse_values(domains, dim):
        for domain in domains:
            r = domain.indexed_ranges[dim][0]
            if not r.is_dummy:
                yield r.sparse_value

    @staticmethod
    def op(*domains):

        min_dom = min_domain(*domains)

        if min_dom is not None:
            return min_dom

        coord_group, slice_dims_and_types = IntersectionOp._intersection_groups(domains)

        slices = SortedDict()
        coords = SortedDict()

        for dim, min_sl, sl_type in slice_dims_and_types:

            if min_sl is not None:
                sl = min_sl
            else:
                sl = sl_type()

                if not sl.is_dummy:

                    sl_sparse_values = IntersectionOp._get_slice_sparse_values(domains, dim)
                    value = IntersectionOp.intersection_sparse(*sl_sparse_values)
                    if value.nnz == 0:
                        return None
                    sl.sparse_value = value

            slices[dim] = sl

        if coord_group:

            min_dom = min_domain(*domains, group=coord_group)

            if min_dom:

                for dim in coord_group:
                    r, c_idx = min_dom.indexed_ranges[dim]
                    if r.is_slice:
                        slices[dim] = r
                    elif c_idx == 0:
                        coords[min_dom.coord_dims_map[id(r)]] = r

            else:

                coord_reshape = tuple(d.get_reshape(coord_group) for d in domains)
                coord_sparse_values = IntersectionOp._get_group_sparse_values(domains, coord_group, coord_reshape)

                coord_range = CoordsRange(ndim=len(coord_group))
                value = IntersectionOp.intersection_sparse(*coord_sparse_values)
                if value.nnz == 0:
                    return None
                coord_range.sparse_value = value

                coords[tuple(coord_group)] = coord_range

        return Domain.domain_from_dicts(coords, slices, shape=domains[0].shape)


Intersection = IntersectionOp.op


if __name__ == "__main__":

    import numpy as np

    multi_prob = 0.2

    def run_pair_test(op_func,
                      check_func_self, check_func_other, check_func_some_eq, check_func_larger_d,
                      check_func_larger_or_smaller_d, check_func_multi):

        multi_doms, empty_doms = [], []

        for t, d1, d2 in generate_domains(pairs=True):

            result = op_func(d1, d2)

            if t == 0:
                check_func_other(result, d1, d2)
            elif t == 1:
                check_func_self(result, d1, d2)
            elif t == 2:
                check_func_some_eq(result, d1, d2)
            elif t == 3:
                check_func_larger_d(result, d1, d2)
            elif t == 4:
                check_func_larger_or_smaller_d(result, d1, d2)

            if np.random.random() > multi_prob:
                multi_doms.append(d2)
                if len(multi_doms) == 3:
                    result = op_func(*multi_doms)
                    check_func_multi(result, *multi_doms)
                    multi_doms = []

    def test_intersection():

        def check_intersection(result, *ds):

            vals = tuple(d.group_ranges() for d in ds)
            vals = IntersectionOp.intersection_sparse(*vals)

            if result is None:
                assert not vals.fill_value and vals.nnz == 0
            else:
                vals = result.group_ranges() == vals
                assert vals is True or vals.fill_value and vals.nnz == 0

        def check_func_self(result, d1, d2):
            check_intersection(result, d1, d2)
            assert result is d1 and result is d2

        def check_func_other(result, d1, d2):
            check_intersection(result, d1, d2)

        def check_func_some_eq(result, d1, d2):
            check_intersection(result, d1, d2)

        def check_func_larger_d(result, d1, d2):
            check_intersection(result, d1, d2)
            assert min_domain(d1, d2) is result and result is d1

        def check_func_larger_or_smaller_d(result, d1, d2):
            check_intersection(result, d1, d2)
            for rr, r1, r2 in zip(result.get_ranges(), d1.get_ranges(), d2.get_ranges()):
                min_ = min_range(r1, r2)
                assert rr is min_ or rr.is_dummy and min_.is_dummy

        def check_func_multi(result, *ds):
            check_intersection(result, *ds)

        run_pair_test(IntersectionOp.op, check_func_self, check_func_other, check_func_some_eq,
                      check_func_larger_d, check_func_larger_or_smaller_d, check_func_multi)

    while True:
        test_intersection()
