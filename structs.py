from collections import namedtuple

TensorData = namedtuple('TensorData', ['name', 'ranges'])


class QueryData(object):

    def __init__(self, tensor, domain_tuple, domain_vals):

        self.tensor = tensor
        self.domain_tuple = domain_tuple
        self.domain_vals = domain_vals
