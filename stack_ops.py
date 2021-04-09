from torch_ops import aggregate_sum, einsum, gather, loss
from subtensor import SubTensor
from domain import Domain, scalar_domain
from domain_op import Intersection


class StackOp(object):

    def __init__(self, program):

        self.program = program
        self.can_apply = None

    def add_op(self, op):
        self.program.push_stack(op)

    def expand(self): raise NotImplemented
    def apply(self): raise NotImplemented


class UpdateLeafOp(StackOp):

    def __init__(self, program, tensor):
        super(UpdateLeafOp, self).__init__(program)
        self.tensor = tensor
        self.can_apply = True

    def expand(self): raise NotImplementedError

    def apply(self):
        self.tensor.reset()
        self.tensor.update_sum()


class QueryOp(StackOp):

    def __init__(self, program, tensor, domain, final_query=False):

        super(QueryOp, self).__init__(program)

        self.tensor = tensor
        self.domain = domain

        self.subtensor = SubTensor(self.domain)
        self.subgoals = []

        self.final_query = final_query

    @staticmethod
    def from_query_data(program, query_data, final_query):

        tensor = program.get_tensor(query_data.tensor)
        domain = Domain.domain_from_query_data(query_data.domain_tuple, query_data.domain_vals, tensor.shape)
        assert not domain.empty, "Cannot run a query with empty domain"
        return QueryOp(program, tensor, domain, final_query)

    def add_op(self, op):
        super(QueryOp, self).add_op(op)
        self.subgoals.append(op)

    def expand(self):

        if self.tensor.name in self.program.leaf_tensors and not self.tensor.empty and not Intersection(self.subtensor.domain, self.tensor.domain) is None:
            self.add_op(GatherOp(self.program, self.tensor, self.domain))

        for program_op in self.program.rules[self.tensor.name]:
            stack_op = program_op.get_stack_op(self.domain)
            if stack_op is not None:
                self.add_op(stack_op)

        return len(self.subgoals) > 0 or self.final_query

    def apply(self):
        aggregate_sum(self.subtensor, *(subgoal.subtensor for subgoal in self.subgoals if subgoal.subtensor.data is not None))
        if self.final_query and self.subtensor.data is None:
            self.subtensor.reset()


class EinsumOp(StackOp):

    def __init__(self, program, lhs_domain, rhs_domains, rhs_tensors, einsum_string, activation):

        super(EinsumOp, self).__init__(program)

        self.rhs_tensors = rhs_tensors
        self.rhs_domains = rhs_domains
        self.einsum_string = einsum_string
        self.activation = activation

        self.subgoals = []
        self.subtensor = SubTensor(lhs_domain)

    def add_op(self, op):
        super(EinsumOp, self).add_op(op)
        self.subgoals.append(op)

    def expand(self):
        for rhs_tensor, rhs_domain in zip(self.rhs_tensors, self.rhs_domains):
            self.add_op(QueryOp(self.program, rhs_tensor, rhs_domain))
        return True

    def apply(self):
        if all(not subgoal.subtensor.empty for subgoal in self.subgoals):
            einsum(self.einsum_string, self.activation, self.subtensor, *(subgoal.subtensor for subgoal in self.subgoals))


class GatherOp(StackOp):

    def __init__(self, program, tensor, domain):

        super(GatherOp, self).__init__(program)

        self.tensor = tensor
        self.subtensor = SubTensor(domain)
        self.can_apply = True

    def expand(self): raise NotImplementedError
    def apply(self):
        gather(self.subtensor, self.tensor.tensor)


class LossOp(StackOp):

    def __init__(self, program, rhs_tensor_1, rhs_domain_1, rhs_tensor_2, rhs_domain_2):

        super(LossOp, self).__init__(program)

        self.rhs_tensor_1 = rhs_tensor_1
        self.rhs_domain_1 = rhs_domain_1
        self.rhs_tensor_2 = rhs_tensor_2
        self.rhs_domain_2 = rhs_domain_2

        self.subgoals = []
        self.subtensor = SubTensor(scalar_domain)

    def add_op(self, op):
        super(LossOp, self).add_op(op)
        self.subgoals.append(op)

    def expand(self):
        if not self.rhs_domain_1.empty and not self.rhs_domain_2.empty:
            self.add_op(QueryOp(self.program, tensor=self.rhs_tensor_1, domain=self.rhs_domain_1))
            self.add_op(QueryOp(self.program, tensor=self.rhs_tensor_2, domain=self.rhs_domain_2))
            return True
        else:
            return False

    def apply(self):
        if all(not subgoal.subtensor.empty for subgoal in self.subgoals):
            loss(self.subtensor, *(subgoal.subtensor for subgoal in self.subgoals))