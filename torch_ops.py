from misc import reduce_concat, reduce_union
from collections import defaultdict
from opt_einsum import contract

from domain import domain_comp_binary


def aggregate_sum(target, *inputs):

    if inputs:
        if all(domain_comp_binary(target.domain, inp.domain) == 0 for inp in inputs):
            target.data = inputs[0].data
            for inp in inputs[1:]:
                target.data += inp.data
        else:
            temp_tensor = target.get_tensor_zeros()
            for inp in inputs:
                temp_tensor[inp.index] += inp.data
            target.data = temp_tensor[target.index]
            del temp_tensor


def _einsum_reorder_string(eq_string, *subtensor_dims):

    eq_items = reduce_concat((eq_item for s in eq_string.split("->") for eq_item in s.split(",")))
    eq_maps = tuple(dict(zip(range(len(eq_item)), eq_item)) for eq_item in eq_items)

    eq_last_letter = max(reduce_concat(*eq_items))

    letter_groups_map = defaultdict(list)

    for eq_map, st_dims in zip(eq_maps, subtensor_dims):
        for dims in st_dims:
            if isinstance(dims, int):
                dims = (dims,)
            letter_group = set(eq_map[dim] for dim in dims)
            for letter in letter_group:
                letter_groups_map[letter].append(letter_group)

    letter_groups_map = {k: reduce_union(*v) for k, v in letter_groups_map.items()}
    new_letter_map = dict()
    new_eq_items = []

    for eq_map, st_dims in zip(eq_maps, subtensor_dims):

        new_eq_item = []

        for dims in st_dims:

            key = eq_map[dims if isinstance(dims, int) else dims[0]]
            letter_group = "".join(letter_groups_map[key])

            if letter_group not in new_letter_map:
                if len(letter_group) > 1:
                    letter = chr(ord(eq_last_letter) + 1)
                else:
                    letter = letter_group
                new_letter_map[letter_group] = letter
            else:
                letter = new_letter_map[letter_group]
            new_eq_item.append(letter)

        new_eq_items.append("".join(new_eq_item))

    equation = ",".join(new_eq_items[:-1]) + "->" + new_eq_items[-1]

    return equation


def einsum(einsum_string, activation_func, target, *inputs):

    einsum_string = _einsum_reorder_string(einsum_string, *(st.data_dims for st in inputs), target.data_dims)
    target.data = activation_func(contract(einsum_string, *(inp.data for inp in inputs)))


def gather(subtensor, tensor): tensor.gather(subtensor)


def loss(target, input_1, input_2, loss_func):
    target.data = loss_func(input_1.data, input_2.data)
