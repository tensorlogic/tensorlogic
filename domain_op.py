from functools import reduce

from domain import Domain, min_domain, generate_domains
from range import CoordRange, range_intersection_type, min_range
from misc import reduce_union


class DomainIntersection(object):

    """
    Class containing all the functions required to compute Domain intersection.
    The approach is the following:
        - we begin by checking if we can find a minimal domain(contained by all or equal to some other domains in the
        intersection) and return it if that's the case.
        - if that is not the case we compute the union of the dimensions which are CoordRanges over all the domains in
        the intersection. The union of the dimensions will be the single CoordRange in the result domain. All other
        slice ranges can be resolved separately, performing the intersection on a per dimension basis.
        - after the previous step, the intersection operation can be split into groups(the CoordRange group and one
        group for each dimension where we have only slice ranges for all domains). We again try to find a minimal
        range to solve each group. If that is not possible, we resort to computing the sparse values of each group
        and perform the sparse operation(bitwise-and) to perform the intersection.
    """

    @staticmethod
    def intersection_sparse(*values):
        return reduce(lambda x, y: x & y, values)

    @staticmethod
    def _get_slice_type_or_min_domain(domains, i):
        min_sl = min_range(*(domain.indexed_ranges[i][0] for domain in domains))
        if min_sl:
            return i, min_sl, None
        else:
            return i, None, range_intersection_type(*(domain.indexed_ranges[i][0] for domain in domains))

    @staticmethod
    def op(*domains):
        """
        Given a set of Domains to intersect, compute their intersection. If the intersection is empty, return None.
        :param domains: Domain
            The set of Domains to perform the intersection of. Empty domains should not be included.
        :return: Domain, None
            The result Domain of the intersection. If the intersection is empty, return None.
        """

        # scalar domains need no intersection
        if domains[0].ndim == 0:
            return domains[0]

        # try to compute the minimal domain and return it if found.
        min_dom = min_domain(*domains)
        if min_dom is not None:
            return min_dom

        # find the union of all the dimensions which are dimensions of CoordRanges over all domains.
        # this union will be the dimensions over which the only CoordRange of the result will span.
        coord_group = reduce_union(*(dims for domain in domains for dims in domain.coord_dims))

        # for the remaining dimensions over which all domains have slice Ranges(non-Coord Ranges),
        # compute the type of range resulting from performing the intersection on a per dimension basis or the minimal
        # range over this dimension.
        slice_dims_and_types = (DomainIntersection._get_slice_type_or_min_domain(domains, i)
                                for i in range(domains[0].ndim) if i not in coord_group)

        slices, coords = {}, {}

        # solve the slice dimensions separately
        for dim, min_sl, sl_type in slice_dims_and_types:

            # if a minimal range is found, return it
            if min_sl is not None:
                sl = min_sl
            # otherwise, create a new range of the inferred type and set the value of the range from the sparse value
            # of the intersection
            else:
                sl = sl_type()
                if not sl.is_dummy:
                    sl_sparse_values = (r.sparse_value for r in (d.indexed_ranges[dim][0] for d in domains) if not r.is_dummy)
                    value = DomainIntersection.intersection_sparse(*sl_sparse_values)
                    if value.nnz == 0:
                        return None
                    sl.sparse_value = value
            slices[dim] = sl

        # solve the coordinate dimensions
        if coord_group:

            # check for a minimal domain
            min_dom = min_domain(*domains, dims=coord_group)

            # if a minimal domain is found, set the ranges of the result domain to the values of that domain.
            # note not all of these dimensions have to be CoordRanges.
            if min_dom:
                for dim in coord_group:
                    r, c_idx = min_dom.indexed_ranges[dim]
                    if not r.is_coord:
                        slices[dim] = r
                    elif c_idx == 0:
                        coords[min_dom.coord_dims_map[id(r)]] = r

            # if not, compute the intersection using the sparse values. We first have to perform a join of the ranges
            # involved, then compute the sparse intersection.
            else:

                value = DomainIntersection.intersection_sparse(*(d.join_ranges(coord_group) for d in domains))

                if value.nnz == 0:
                    return None

                coord_range = CoordRange(ndim=len(coord_group))
                coord_range.sparse_value = value
                coords[tuple(coord_group)] = coord_range

        return Domain.domain_from_dicts(coords, slices, shape=domains[0].shape)


Intersection = DomainIntersection.op


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

            vals = tuple(d.join_ranges() for d in ds)
            vals = DomainIntersection.intersection_sparse(*vals)

            if result is None:
                assert not vals.fill_value and vals.nnz == 0
            else:
                vals = result.join_ranges() == vals
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

        run_pair_test(DomainIntersection.op, check_func_self, check_func_other, check_func_some_eq,
                      check_func_larger_d, check_func_larger_or_smaller_d, check_func_multi)

    while True:
        test_intersection()

