from collections import namedtuple

# data structure used to store parsed tensor information:
#   - name: string. the tensor name
#   - ranges: tuple[(str, int)]. the name of the range and the dimension of that range corresponding to the dimension of
#             the tensor
TensorData = namedtuple('TensorData', ['name', 'ranges'])


class QueryData(object):

    def __init__(self, tensor_name, domain_tuple, domain_vals):
        """
        Data structure used to represent a query
        :param tensor_name: str
            The name of the tensor to query.
        :param domain_tuple: tuple[str], list[str]
            A domain tuple which has the same format as the tensor indices in the program declaration.
        :param domain_vals: dict[str, [None, integer-like, slice, Iterable]]
            A dictionary of the values which maps each range name to its value. The following value types should be used
            for each Range type:
                None -> DummyRange
                int -> IntRange
                1d array-like -> SetRange
                nd array-like -> CoordRange
                slice -> SliceRange
            The type of Range need not be specified and will be inferred from the type of the value
        """
        self.tensor = tensor_name
        self.domain_tuple = domain_tuple
        self.domain_vals = domain_vals
