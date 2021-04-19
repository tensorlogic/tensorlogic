from collections import defaultdict
from opt_einsum import contract

from misc import reduce_concat, reduce_union_sorted
from domain import domain_comp_binary
from subtensor import SubTensor
from torch_tensor import DenseTensor


def aggregate_sum(target_domain, *inputs):

    """
    Aggregate with a sum multiple SubTensors and place the result in a new SubTensor with the Domain set to
    target_domain.
    :param target_domain: Domain
        The target Domain.
    :param inputs: SubTensors
        The SubTensors to aggregate.
    :return: SubTensor
        The result SubTensor.
    """

    # create the target subtensor
    target = SubTensor(target_domain)

    # if all the domains are equal to the target one, simply add the data from the input domains to the
    # data of the target domain
    if all(domain_comp_binary(target.domain, inp.domain) == 0 for inp in inputs):
        target.data = inputs[0].data
        for inp in inputs[1:]:
            target.data += inp.data

    # otherwise, create a temporary tensor, write the data to it, then perform a gather to extract the aggregated
    # data into the target subtensor
    else:
        temp_tensor = DenseTensor(target.shape)
        temp_tensor.reset()
        for inp in inputs:
            temp_tensor.update_sum(inp)
        temp_tensor.gather(target)
        del temp_tensor

    return target


def _einsum_reorder_string(eq_string, *subtensor_dims):

    eq_items = reduce_concat((eq_item for s in eq_string.split("->") for eq_item in s.split(",")))
    eq_last_letter = max(reduce_concat(*eq_items))

    letter_groups_map = defaultdict(list)

    for eq_item, st_dims in zip(eq_items, subtensor_dims):
        for dims in st_dims:
            letter_group = {eq_item[dims]} if isinstance(dims, int) else set(eq_item[dim] for dim in dims)
            for letter in letter_group:
                letter_groups_map[letter].append(letter_group)

    letter_groups_map = {k: reduce_union_sorted(*v) for k, v in letter_groups_map.items()}
    new_letter_map = dict()

    for letter_group in letter_groups_map.values():
        letter_group = "".join(letter_group)
        if letter_group not in new_letter_map:
            if len(letter_group) > 1:
                letter = chr(ord(eq_last_letter) + 1)
            else:
                letter = letter_group
            new_letter_map[letter_group] = letter

    new_eq_items = []

    for eq_item, st_dims in zip(eq_items, subtensor_dims):

        new_eq_item = []

        for dims in st_dims:

            key = eq_item[dims if isinstance(dims, int) else dims[0]]
            letter_group = "".join(letter_groups_map[key])
            letter = new_letter_map[letter_group]
            new_eq_item.append(letter)

        new_eq_items.append("".join(new_eq_item))

    equation = ",".join(new_eq_items[:-1]) + "->" + new_eq_items[-1]

    return equation


def einsum(einsum_string, activation_func, target_domain, *inputs):
    """
    Perform the einsum of the input SubTensors as specified by the einsum equation in einsum_string and store the
    result in a new SubTensor created with the target_domain. The original einsum_string corresponds to the full tensors
    involved in the computation and not the subtensors. Thus, the string needs to be modified depending on the input and
    target SubTensors since some dimensions of the full einsum operation might have to be removed(if they are indexed
    by integers for example), or some dimensions might get permuted during gathering or will be permuted in the the
    target SubTensor when updating. This is done in the _einsum_reorder_string function.
    :param einsum_string: str
        Einsum equation for the full tensors. This will be modified for the SubTensors.
    :param activation_func: function
        The activation function to be applied after the einsum.
    :param target_domain: Domain
        The target Domain.
    :param inputs: SubTensors
        The SubTensors to perform the einsum operation on.
    :return: SubTensor
        The result SubTensor.
    """
    target = SubTensor(target_domain)

    einsum_string = _einsum_reorder_string(einsum_string, *(st.data_dims for st in inputs), target.data_dims)
    target.data = activation_func(contract(einsum_string, *(inp.data for inp in inputs)))

    return target


def gather(target_domain, tensor):
    """
    Gather data from a Tensor and place into a newly created SubTensor with the target_domain.
    :param tensor: Tensor
        The tensor to gather data from.
    :param target_domain: Domain
        The target Domain.
    :return: SubTensor
        The result SubTensor.
    """
    subtensor = SubTensor(target_domain)
    tensor.gather(subtensor)
    return subtensor


def loss(input_1, input_2, loss_func):
    """
    Compute the torch loss operation between the data stored in two SubTensors. The data is assumed to have the
    dimensions requited by the loss function.
    :param input_1: SubTensor
    :param input_2: SubTensor
    :param loss_func: function
        Torch function used to compute the loss.
    :return: torch.Tensor
        The value of the loss
    """
    return loss_func(input_1.data, input_2.data)
