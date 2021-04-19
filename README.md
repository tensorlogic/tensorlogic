
**TensorLogic syntax**

- Each line begins with an id_string which allows the parser to identify which type of line it has to parse.
  - Possible id_strings: _range_, _tensor_, _input_, _constant_, _weight_, _einsum_, _loss_.

- Depending on the type of line, the id_string is followed by at least one space and either a set of names
  separated by spaces(when declaring ranges and tensors), or by a formula whose lhs and rhs are tensor names
  indexed by range names and whose lhs and rhs are separated by the symbol ":-"(when creating program inputs,
  clauses/rules, or losses).

- After the names or formulas and at least one space, the flags corresponding to each line are writen down.
  The flags are writen as flag=value and are separated by at least one space(not by commas). The flag name
  is always a string and is not writen with quotation marks. The values of the flags can be integers, floats,
  strings and tuples(of ints, floats, or strings). Tuple values are separated by commas,
  an empty tuple is writen as (), a tuple of dimension 1 is writen as (item).

- Tensors are indexed in the following way: TensorName[range_name_1, range_name_2, ...]. The TensorName
  has to have been declared before the tensor is writen in a formula(rule, input, loss).
  When a coord range is used, the RangeName is followed by a dot and an integer between 0 and (# of dims - 1) of
  the coord range(this is the case even if the coordinates are 1 dimensional). The integer indicates which
  dimension of the range indexes into the tensor dimension.
  - e.g: 
    - indexing without coordinates: T[x, y].
    - indexing with coordinates: T[x.0, x.1].
    - indexing with both coordinates and non-coordinates: T[x.0, y, x.1].

- To declare ranges, the following type of lines have to be written:
  - _range [int|set|slice|coords] range_name_1 range_name_2 ... flag_1_key=flag_1_value flag_2_key=flag_2_value ..._
  - e.g: 
    - _range int A B_
    - _range set some_set1 some_set2_
    - _range slice Slice_
    - _range coords C ndim=2_
  - Note: coord ranges require ndim as a flag.
  - Note: the names are not separated by commas, only by spaces.
  - Note: the names are not inside quotes.
  - Note: ranges which are not declared, but appear in a rule are considered as dummy ranges. See semantics bellow.

- To declare a tensor, the following type of lines have to be written before the tensors are used inside a rule:
  - _tensor tensor_name_1 tensor_name_2 ... shape=(dimension_1, dimension_2, ...)_
  - e.g:
    - _tensor T shape=(100, 200)_
    - _tensor A B shape=(100, 200)_
    - _tensor 2dVector shape=(2)_
    - _tensor ScalarT shape=()_
  - Note: the shape is a required flag.
  - Note: a scalar tensor has the shape=(), a tensor with only one dimension has the shape=(dim_length).

- To declare an input to the program, the following type of lines have to be writen:
  - _[constant|weight|input] Tensor_name[range_dim_0, range_dim_1, ...] :- flag_1_key=flag_1_value flag_2_key=flag_2_value ..._
  - e.g:
    - _constant Constant_Tensor[x, y] :-_
    - _weight Weight_Tensor[w] :- initializer=uniform from=-0.1 to=0.1_
    - _input Input[coords.0, coords.1] :- type=data_
    - _input Label[x, coords.0, coords.1] :- type=label_
  - Note: see the tensor indexing rules above.
  - Note: the tensor has to have been declared before an input using that tensor is created.
  - Note: each type of input has its own set of flags:
    - flags for constant:
      - type(optional): _data_ or _label_. If data, the input will always be cast to a float. If label,
        the input will not be cast to anything, but should be given as a torch.Tensor.
    - flags for weight:
      - initializer(required): _normal_, _uniform_, _xavier_uniform_, _xavier_normal_. A string identifying the
        type of weight initializer to use.
      - batch_dim(optional): int. Indicates which dimension, if any, of the weight is to be treated as
        a batch dimension. For example, biases in a neural network will have the 0th dimension as a batch
        dimension.
      - other flags required by the torch initializers. See torch documentation for these.
    - flags for input:
      - type(optional): _data_ or _label_. If data, the input will always be cast to a float. If label,
        the input will not be cast to anything, but should be given as a torch.Tensor.

- To declare a clause/rule, an einsum line has to be writen as:
  - _einsum LHS[range_dim_0, range_dim_1, ...] :- RHS1[range_dim_0, range_dim_1, ...]RHS2[range_dim_0, range_dim_1, ...] ... flag_1_key=flag_1_value   flag_2_key=flag_2_value_
  - e.g:
    - _Mortal[x] :- Human[x]_
    - _Cat[x] :- Meows[x]Has_Tail[x]Is_Fluffy[x]_
    - _HasFunnyFriends[x] :- Friends[x,y]Funny[y]_
    - _Feature2[x] :- Weight[x, f]Feature1[f] activation=sigmoid_
  - Note: see the tensor indexing rules above.
  - Note: A rule cannot have the same tensor on the RHS as on the LHS, recursion is not supported.
  - Note: The tensors in the rule have to have been declared before a rule using the tensor is created.
  - Note: A tensor index on the LHS has to appear on the RHS. The opposite is not true as these dimensions will be projected.
  - Note: the only flag an einsum rule has is an activation:
      - activation(optional): _relu_, _sigmoid_, _tanh_, _elu_, _softmax_, _log_softmax_. Softmaxes are applied on the last dimension. If no value is given, no activation function is applied after the einsum is computed.

- To declare a loss, a loss line has to be writen as:
  - _loss :- RHS1[range_dim_0, range_dim_1, ...]RHS2[range_dim_0, range_dim_1, ...] flag_1_key=flag_1_value flag_2_key=flag_2_value_
  - e.g: 
    - _loss :- Output[x, y]Label[x, y] loss=nll name=loss_
  - Note: see the tensor indexing rules above.
  - Note: The tensors in the loss have to have been declared before the loss is created.
  - Note: The ranges in the loss need not match as long the tensors obtained after slicing the tensors on RHS have the shapes expected by the torch loss.
  - Note: A loss line accepts the following flags:
    - name(required): str. Unique identifier of the loss that the user can use the access the loss.
    - loss(required): str. Unique identifier of the type of torch loss to compute.
    - other flags required by the torch losses. See torch documentation for these.
