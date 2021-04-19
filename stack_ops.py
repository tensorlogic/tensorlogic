from torch_ops import aggregate_sum, einsum, gather, loss
from subtensor import SubTensor
from domain import Domain
from domain_op import Intersection


class StackOp(object):

    """
    A StackOp object is the main engine of a TensorLogic program and is used to compute all the operations during the
    run stage of the program. The program keeps a stack of StackOps and at each step it pops one operation until no
    operations are left.

    Each StackOp has two stages: expand and apply. In the expand stage, the StackOp pushes unto the stack subgoals
    that it needs evaluated before the apply stage(for example, an einsum StackOp will create Query ops for all tensors
    on the rhs before computing the einsum). In the apply stage, the StackOp uses the information in the subgoals to
    compute the operation and stores the result such that other StackOps can use it.
    """

    def __init__(self, program):
        """
        Create a StackOp object
        :param program: Program
            The Program to add the StackOp to.
        """
        self.program = program
        self.can_apply = False
        self.subgoals = []

    def add_subgoal(self, op):
        self.program.push_stack(op)
        self.subgoals.append(op)

    def expand(self):
        """
        This is the first stage of the StackOp in which subgoals specific to each StackOp are created and added to the
        StackOp's list of subgoals and to the Program's stack. These subgoals have to be computed before the apply
        stage can be evaluated.
        :return: bool
            If the expand stage fails return False, otherwise return True.
        """
        raise NotImplemented

    def apply(self):
        """
        This is the second stage of the StackOp in which the actual operation is computed using the results of the
        subgoals. The result of the op will be stored in such a way that it is accessible to other ops which require
        the result.
        :return:
        """
        raise NotImplemented


class ResetLeafOp(StackOp):

    """
    A ResetLeafOp resets the leaf ProgramTensors. It does not require an expand stage and it is always added to
    the stack by the program at the beginning of a run step.
    """

    def __init__(self, program, tensor):
        """
        Create a ResetLeafOp.
        :param program: Program
            The Program the StackOp will be added to.
        :param tensor: ProgramTensor
            The ProgramTensor to reset.
        """
        super(ResetLeafOp, self).__init__(program)
        self.tensor = tensor
        self.can_apply = True

    def expand(self): raise NotImplementedError
    def apply(self): self.tensor.reset()


class QueryOp(StackOp):

    """
    A QueryOp is used to query information about a tensor. The part of the tensor the query is interested in
    is specified by the domain. Queries are created either by the user or by other operations. Queries will be
    matched to either rules(intensional data) or to leaf tensors(extensional data). When a match is successful,
    a QueryOp will push unto the stack a GatherOp for the leaf tensors or an EinsumOp for the rules.
    After these subgoals are evaluated, the query op aggregates the results and places them into its subtensor.
    """
    def __init__(self, program, tensor, domain, user_query=False):
        """
        Create a QueryOp
        :param program: Program
            The Program the StackOp will be added to.
        :param tensor: ProgramTensor
            The ProgramTensor to query.
        :param domain: Domain
            The domain specifying the subpart of the tensor the query is interested in.
        :param user_query: bool
            Flag specifying whether the query is created by the user(True) or if it is created by another op(False).
        """
        super(QueryOp, self).__init__(program)

        self.tensor = tensor
        self.domain = domain

        self.subtensor = None

        self.user_query = user_query

    @property
    def empty(self): return self.subtensor is None or self.subtensor.empty

    @staticmethod
    def from_query_data(program, query_data):

        """
        Create a QueryOp from a QueryData structure.
        :param program: Program
        :param query_data: QueryData
        :return: QueryOp
        """

        tensor = program.get_tensor(query_data.tensor)
        domain = Domain.domain_from_query_data(query_data.domain_tuple, query_data.domain_vals, tensor.shape)

        assert not domain.empty, "Cannot run a query with empty domain"
        return QueryOp(program, tensor, domain, True)

    def expand(self):

        # If the tensor is a non-empty leaf tensor, we can check if the query matches some part of the tensor
        # If it does, we push a GatherOp unto the stack.
        if self.tensor.set_from_input and not self.tensor.empty and not Intersection(self.domain, self.tensor.domain) is None:
            self.add_subgoal(GatherOp(self.program, self.tensor, self.domain))

        # For all the rules that match the tensor of the query we attempt to push an EinsumOp unto the stack.
        for program_op in self.program.rules[self.tensor.name]:
            # We attempt to create the Einsum op. When this fails, it means that the domain of the query has an empty
            # intersection with the lhs domain of the rule. This means that the rule is not applicable, so there is
            # no point in creating this subgoal.
            stack_op = program_op.stack_op(self.domain)
            if stack_op is not None:
                self.add_subgoal(stack_op)

        return len(self.subgoals) > 0 or self.user_query

    def apply(self):

        # gather the subtensors from the subgoals which are not empty.
        inputs = tuple(sg.subtensor for sg in self.subgoals if not sg.empty)
        if inputs:
            self.subtensor = aggregate_sum(self.domain, *inputs)

        # if this is a user query which turns out to be empty(no intensional or extensional data was found to match
        # the query), set the query to all 0s.
        if self.user_query and self.empty:
            if self.subtensor is None:
                self.subtensor = SubTensor(self.domain)
            self.subtensor.set_zeros()


class EinsumOp(StackOp):

    """
    An EinsumOp is used to update tensors with intensional information. The EinsumOp is created by the Einsum
    ProgramOp after it has matched some query operation. The EinsumOp's lhs_domain is the intersection of the
    lhs_domain with the query op and the rhs_domains are the domains obtained after substitution is performed.
    In the expand stage, the EinsumOp creates QueryOps for each of the SubTensors on the rhs. If all the subgoals
    are computed to be non-empty, the einsum is then evaluated in the apply stage.
    """

    def __init__(self, program, lhs_domain, rhs_domains, rhs_tensors, einsum_string, activation):
        """
        Create an EinsumOp.
        :param program: Program
            The Program the StackOp will be added to.
        :param lhs_domain: Domain
            The domain which matched the query.
        :param rhs_domains: list[Domain]
            The substituted domains on the rhs which will be queried.
        :param rhs_tensors: list[ProgramTensor]
            The tensors on the rhs.
        :param einsum_string: str
            The einsum equation writen for the whole tensors. This will be modified in the einsum computation function.
        :param activation: function
            The activation function to be applied after the einsum.
        """
        super(EinsumOp, self).__init__(program)

        self.lhs_domain = lhs_domain
        self.rhs_tensors = rhs_tensors
        self.rhs_domains = rhs_domains
        self.einsum_string = einsum_string
        self.activation = activation

        self.subtensor = None

    @property
    def empty(self): return self.subtensor is None or self.subtensor.empty

    def expand(self):
        # for each substituted domain and tensor on the rhs, add a QueryOp subgoal.
        for rhs_tensor, rhs_domain in zip(self.rhs_tensors, self.rhs_domains):
            self.add_subgoal(QueryOp(self.program, rhs_tensor, rhs_domain))
        return True

    def apply(self):
        # if all subgoals are not empty, compute the einsum.
        if all(not subgoal.empty for subgoal in self.subgoals):
            self.subtensor = einsum(self.einsum_string, self.activation, self.lhs_domain, *(sg.subtensor for sg in self.subgoals))


class GatherOp(StackOp):
    """
    A GatherOp extracts data from a leaf tensor. It does not have an expand stage and can be applied directly.
    """
    def __init__(self, program, tensor, domain):
        """
        Create a GatherOp.
        :param program: Program
            The Program the GatherOp will be added to.
        :param tensor: ProgramTensor
            The ProgramTensor to gather from.
        :param domain: Domain
            The domain specifying the subpart of the tensor the gather op is interested in.
        """
        super(GatherOp, self).__init__(program)

        self.tensor = tensor
        self.domain = domain

        self.can_apply = True
        self.subtensor = None

    @property
    def empty(self): return False
    def expand(self): raise NotImplementedError
    def apply(self): self.subtensor = gather(self.domain, self.tensor.tensor)


class LossOp(StackOp):

    """
    A LossOp computes the specified loss function between the SubTensors obtained after querying tensors on the rhs
    with the specified domains. The loss function and the domains of interest are created by the Loss ProgramOp.
    The result of the loss will be stored in the data field.
    """
    def __init__(self, program, rhs_tensor_1, rhs_tensor_2, rhs_domain_1, rhs_domain_2, loss_func):
        """
        Create a LossOp
        :param program: Program.
            The Program that the GatherOp will be added to.
        :param rhs_tensor_1: ProgramTensor
            Tensor 1 to query.
        :param rhs_tensor_2: ProgramTensor
            Tensor 2 to query.
        :param rhs_domain_1: Domain
            Domain 1 to query.
        :param rhs_domain_2: Domain
            Domain 2 to query.
        :param loss_func: function
            The torch loss function to compute.
        """
        super(LossOp, self).__init__(program)

        self.rhs_tensor_1 = rhs_tensor_1
        self.rhs_tensor_2 = rhs_tensor_2
        self.rhs_domain_1 = rhs_domain_1
        self.rhs_domain_2 = rhs_domain_2

        self.loss_func = loss_func

        self.data = None

    def expand(self):
        # add a query subgoal for both tensors on the rhs.
        self.add_subgoal(QueryOp(self.program, tensor=self.rhs_tensor_1, domain=self.rhs_domain_1))
        self.add_subgoal(QueryOp(self.program, tensor=self.rhs_tensor_2, domain=self.rhs_domain_2))
        return True

    def apply(self):
        # if both the query subgoals are not empty, then we can compute the loss.
        if all(not subgoal.empty for subgoal in self.subgoals):
            self.data = loss(*(subgoal.subtensor for subgoal in self.subgoals), self.loss_func)
