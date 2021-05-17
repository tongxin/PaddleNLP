# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from itertools import chain
import re
import paddle

__all__ = ['einsum']

def parse_all_op_labels(label_str, operands):
    '''
    Parses out labels for all input operands in the label string.

    Parameters
    ----------
    label_str:
        the label string of einsum equation
    operands:
        the input operands

    Returns
    -------
    A list of parsed labels for all the operands
    '''
    # Sanity checks
    assert label_str.count(',') + 1 == len(operands), "Invalid equation: the input labels do not match the number of input operands."
    # assert any(not labels for labels in op_labels_list), "Invalid equation: subscripts split by ',' got empty strings."
    # yields a pair of labels, operand
    return [parse_op_labels(op_labels, op) for op_labels, op in zip(label_str.split(','), operands)]
        
def visit_op_labels(f, op_labels_list, operands=None):
    '''
    Traverse the list of op labels, applying the visitor callback to each along the way.
    Returns the results in a generator.

    Parameters
    ----------

    f:
        the visitor callback
    
    op_labels_list:
        the list of labels.

    operands:
        the input operands
    '''
    if operands:
        for op_labels, op in zip(op_labels_list, operands):
            yield f(op_labels, op)
    else:
        for op_labels in op_labels_list:
            yield f(op_labels)

def parse_op_labels(labels, operand):
    '''
    Parses labels for an input operand.
    Returns an extended label string with all missing labels filled as dots
    '''
    # Sanity checks
    assert all(c in '.' or c.isalpha() for c in labels), f"Invalid equation: a label is expected to be in [a-Z] but found {c}."
    # Expand the labels
    op_rank = len(operand.shape) # Note, in Paddle a tensor rank is always nonzero
    assert op_rank > 0

    dot_pos = labels.find('.')
    if dot_pos == -1:
        # No ellipsis, implying the labels should straightly match the operand dimensions
        assert len(labels) == op_rank, f"Invalid equation: missing labels for input operand '{operand.name}''."
        extended_labels = labels
    if dot_pos >= 0:
        ell_pos, extra_dot = labels.find('...', dot_pos), labels.find('...', dot_pos+3)  
        # Note, the order of the following two asserts matters
        assert dot_pos == ell_pos, "Invalid equation: ellipsis is expected but not found."
        assert extra_dot == -1, "Invalid equation: `.` is only expected to be included in an ellipsis."
        assert len(labels) - 3 <= op_rank, "Invalid equation: more labels are found than the available dimensions in operand '{operand.name}'."
        ell_rank = op_rank - len(labels) + 3
        extended_labels = ('.' * ell_rank).join(labels.split('...'))
        # return a tuple of expanded subscripts, ellipsis masked rank for each operand
    # Do we need to handle at this point the trace and diag cases
    return extended_labels

def has_bcast_dims(extended_labels, operand=None):
    '''
    Returns whether there are non-labeled dimensions by checking the extended labels 
    '''
    return '.' in extended_labels

def num_bcast_dims(extended_labels, operand=None):
    '''
    Returns the number of broadcast dimensions
    '''
    return extended_labels.count('.')

def get_bcast_dims_indices_and_shape(op_shape, op_labels):
    '''
    Returns the indices and shape of the broadcast dimensions.

    Parameters
    ----------
    op_shape:
        the tensor shape of the operand
    op_labels:
        the extended label string for the operand. Broadcast dimensions are labeled with dots.
    

    Returns
    -------
    indices:
        the indices of the broadcast dimensions
    shape:
        the sizes of the broadcast dimensions
    '''
    assert len(op_shape) == len(op_labels)

    indices, shape = [], []
    for i, size, label in zip(range(len(op_shape)), op_dims, op_labels):
        if label == '.':
            indices.append(i)
            shape.append(size)

    return indices, shape

def bcastable_test(args, f=None):
    '''
    Tests if the two operands can perform a broadcast operation on the given ranges of dimensions. 
    We follow the Numpy broadcasting convention which states that, by lining up the shape arrays
    starting from the right most dimension, all the aligned dimensions either have equal sizes or
    one of them is sized one.

    Parameters
    ----------
    args:
        *args unpacks into operand one's axes range, shape, operand two's axes range, shape

    f: 
        if available, is used as a callback for postprocessing the aligned operand dimensions.
    '''
    xran, xshape, yran, yshape = *args

    xran_inv, yran_inv = xran[::-1], yran[::-1]

    for xi, yi in zip(xran_inv, yran_inv):
        xs, ys = xshape[xi], yshape[yi]
        cond = xs == ys or xs == 1 or ys == 1
        if not cond:
            return False

    if not f:
        return True

    # Apply the callback to each aligned dimension pair
    for xi, yi in zip(xran_inv, yran_inv):
        f(xi, yi)

def gather_avail_labels(labels_list):
    '''
    Returns a sorted string of all the available labels in the list
    '''
    labelset = set()

    for _ in visit_op_labels(labelset.update, labels_list):
        pass
    
    return ''.join(sorted(labelset))

def gather_singular_labels(labels_list, alphabet_only=True):
    '''
    Returns the labels that appear only once
    Parameter alphabet_only indicates whether to count labels in [a-z] only
    '''
    all_labels = sorted(''.join(labels_list))    

    _off = 0
    if alphabet_only:
        for i, l in enumerate(all_labels):
            if l.isalpha():
                _off = i
                break

    all_labels = all_labels[_off:]

    singular_labels = []
    last_label, count = None, 0
    for l in all_labels:
        if (l != last_label):
            # new label, the last label is singular is count is one
            if count == 1:
                singular_labels.append(l)
            label, count = l, 1
        else:
            count += 1
    if count == 1:
        singular_labels.append(all_labels[-1])

    return ''.join(singular_labels)


def parse_output_labels(rhs, avail_labels, n_bcast_dims):
    '''
    Parse explicit output labels given on the right hand side of '->' and the available
    input labels.

    Parameters
    ----------
    rhs:
        the output label string, given by the right hand side of the einsum equation
    avail_labels:
        the available labels to check with
    n_bcast_dims:
        the number of broadcast dimensions

    Returns
    -------
    The output labels in a string
    '''
    # Sanity check. Only alphabet is allowed if not '.'
    assert all(c in avail_labels for c in rhs), f"Invalid equation: an output label is expected to be included in the input labels but `{c}` is found."

    # Syntax sanity check. Verify there's no duplicate labels
    for i, l in enumerate(rhs.replace('.', '')):
        if rhs.find(l, 0, i) >= 0:
            assert False, f"Invalid equation: duplicate output label {l}."

    if '.' in avail_labels:
        # Syntax sanity check. Verify that dots exist if and only if they show up in an ellipsis
        dot_pos = rhs.find('.')
        ell_pos, extra_dot = rhs.find('...', dot_pos), rhs.find('.', dot_pos+3)
        assert ell_pos == dot_pos and extra_dot == -1, "Invalid equation: `.` is only expected to be included in an ellipsis."
        out_labels = ('.'*n_bcast_dims).join(rhs.split('...'))
    else:
        assert n_bcast_dims == 0,  "Invalid equation: more output dimensions than labels, missing '...'."
        out_labels = rhs

    return out_labels

def infer_output_labels(list_op_labels, n_bcast_dims):
    '''
    Infer output labels in case no explicit output labels are given on the right hand side of '->'.
    The output labels are those that appear only once, put in alphabetical order. 
    Returns the output labels in a string
    '''
    output_labels = ''
    # Broadcast labels come first
    output_labels += '.' * n_bcast_dims
    # Followed by singular labels
    output_labels += gather_singular_labels(list_op_labels)

    return output_labels

def dim_strides(shape):
    '''
    Returns the dimension strides for a tensor shape
    '''
    strides = []
    stride = 1
    for size in shape[::-1]:
        strides.append(stride)
        stride = stride * size
    return strides

def create_op_view(operand, *view_def):
    '''
    Create and materialize a view of an operand.
    
    Parameters
    ----------

    operand:
        the base tensor operand

    view_def: 
        include two lists which define the view's dimension sizes and strides
    '''
    view_sizes, view_strides = view_def
    return operand.create_view(view_sizes, view_strides)    

def has_duplicated_labels(labels, *args):
    '''
    Returns True if there is any duplicate label.
    '''
    labels = labels.replace('.', '')
    return any(l in labels[i+1:] for i, l in enumerate(labels))

def diagonalize(labels, operand):
    '''
    Merges dimensions if there are duplicated labels. 
    
    For those dimensions with duplicate labels, merge them into one dimension
    which represents the diagonal elements. That requires the duplicate labeled 
    dimensions have the same size. The order of dimensions is kept unchanged
    up to the left-most appearance of each label.

    Examples
    -------- 

    'ijj...i' would be merged into 'ij...'

    '''
    op_strides = dim_strides(operand.shape)
    op_shape = operand.shape
    new_op_labels = []
    new_op_sizes = []
    new_op_strides = []
    
    for i, l in enumerate(labels):
        newi = new_op_labels.index(l)
        if newi < 0 or l == '.':
            # not duplicate
            new_op_labels.append(l)
            new_op_strides.append(op_strides[i])
            new_op_sizes.append(op_shape[i])
        else:
            # duplicated label
            new_op_strides[newi] += op_strides[i]

    # call framework API to build a new tensor
    new_op = create_op_view(operand, new_op_sizes, new_op_strides)
    return new_op, new_op_labels

def inv_map(in_labels, out_labels):
    '''
    Build an inverse map of dimension indices. Following prerequisites must hold to make
    the result meaningful. First, there's no duplicate alphabet labels in either parameters.
    Second, the broadcast dimensions in out_labels, are at least as many as in in_labels.
    Third, indices of broadcast dimension are contiguous.

    Parameters
    ----------
    in_labels:
        The dimension labels to map to
    out_labels:
        The dimension labels to map from
    

    Returns
    -------
    The inverse map from out_labels to in_labels. The length of the inverse map equals that of
    out_labels. -1 is filled if there's no matching intput dimension for a specific label.

    Examples
    --------
    in_labels = 'ij..', out_labels = '..ji'
    inv_map = [2, 3, 1, 0]

    in_labels = 'ij..', out_labels = '..kji'
    inv_map = [2, 3, -1, 1, 0]
    '''
    inv_map = [-1] * len(out_labels)
    
    # First build the broadcast dimension mapping
    # Find the broadcast index range in out_labels
    r = re.search('\.+', out_labels)
    if r:
        start, end = r.start(), r.end()
        s = re.search('\.+', in_labels)
        # fill the broadcast dimension indices from right to left.
        if s:
            inv_map[end:start:-1] = range(s.end(), s.start())
        
    # Now work on non-broadcast dimensions 
    if start:
        it = chain(range(start), range(end, len(out_labels)))
    else:
        it = iter(range(len(out_labels)))
        
    for i in it:
        inv_map[i] = in_labels.Find(out_labels[i])

    return inv_map
    

def join(x, y, xlabels, ylabels, out_labels):
    '''
    Joins two tensor operands x and y following the input and output labels
    Returns [tensor, dimension labels] pair

    For each input operand, a dimension falls into 4 categories based on its subscript's properties:
    - subscript is shared and in output
    - subscript is shared and not in output
    - subscript is not shared, and in output
    - subscript is not shared, and not in output

    For not shared and not output subscripts, the dimensions will be reduced first.
    After that local reduction step, all operand tensors are permuted into shape
    [batch_dims..., index_dims..., sum_dims...], where
    batch_dims... are dimensions with shared *and* output subscripts,
    index_dims... dimensions with output and not shared subscripts,
    sum_dims... dimensions with shared and not output subscripts

    For all shared dimensions, their size and order must be consistent.
    The operation cannot proceed if there's inconsistent dimensions

    Finally, the summed result is returned along with its dimension subscripts
    '''


def einsum(equation, *operands):
    r"""
    Executes the sum of product of provided operands based on the Einstein summation convention.
    Einsum can be used to complete a variety of operations, such as sum, transpose,
    batch matrix multiplication.

    Args:
        equation (`str`):
            The equation uses uncased letters to indicate the dimensions for summation. 
            These letters are called dimension labels or dimension subscripts. The dimension labels
            are comma separated to correspond to all the input operands. The equation uses `->` to indicate 
            explicitly the output dimensions which otherwise would be collapsed as the result of summation.
            In case the explicit output is not given, Einsum will deduce the output dimensions automatically.
            Dimensions with the same label should be broadcastable. The equation uses ellipsis ('...') to 
            specify broadcast dimensions.

        operands (`Tensor`):
            The operands to compute the Einstein sum of. The number of operands should be the same as the
            the operands described in input equation.
    
    Returns:
        `Tensor`: The result of Einstein sum product.
    
    Example:
    .. code-block::

        import paddle
        import paddlenlp
        import numpy as np

        np.random.seed(102)

        x = paddle.to_tensor(np.random.rand(4))
        y = paddle.to_tensor(np.random.rand(5))
        # sum
        print(paddlenlp.ops.einsum('i->', x))
        # Tensor(shape=[], dtype=float64, place=CUDAPlace(0), stop_gradient=True, 2.30369050)

        # dot
        print(paddlenlp.ops.einsum('i,i->', x, x))
        # Tensor(shape=[], dtype=float64, place=CUDAPlace(0), stop_gradient=True, 1.43773247)

        # outer
        print(paddlenlp.ops.einsum("i,j->ij", x, y)),
        # Tensor(shape=[4, 5], dtype=float64, place=CUDAPlace(0), stop_gradient=True,
        #         [[0.34590188, 0.48353496, 0.09996135, 0.18656330, 0.21392910],
        #         [0.39122025, 0.54688535, 0.11305780, 0.21100591, 0.24195704],
        #         [0.17320613, 0.24212422, 0.05005442, 0.09341929, 0.10712238],
        #         [0.42290818, 0.59118179, 0.12221522, 0.22809690, 0.26155500]])

        A = paddle.to_tensor(np.random.rand(2, 3, 2))
        B = paddle.to_tensor(np.random.rand(2, 2, 3))
        # transpose
        print(paddlenlp.ops.einsum('ijk->kji', A))
        #  Tensor(shape=[2, 3, 2], dtype=float64, place=CUDAPlace(0), stop_gradient=True,
        #        [[[0.49174730, 0.33344683],
        #          [0.89440989, 0.26162022],
        #          [0.36116209, 0.12241719]],

        #         [[0.49019824, 0.51895050],
        #          [0.18241053, 0.13092809],
        #          [0.81059146, 0.55165734]]])

        # batch matrix multiplication
        print(paddlenlp.ops.einsum('ijk, ikl->ijl', A,B))
        # Tensor(shape=[2, 3, 3], dtype=float64, place=CUDAPlace(0), stop_gradient=True,
        #     [[[0.13654339, 0.39331432, 0.65059661],
        #      [0.07171420, 0.57518653, 0.77629221],
        #      [0.21250688, 0.37793541, 0.73643411]],

        #     [[0.56925339, 0.65859030, 0.57509818],
        #      [0.30368265, 0.25778348, 0.21630400],
        #      [0.39587265, 0.58031243, 0.51824755]]])

        # Ellipsis transpose
        print(paddlenlp.ops.einsum('...jk->...kj', A))
        # Tensor(shape=[2, 2, 3], dtype=float64, place=CUDAPlace(0), stop_gradient=True,
        #     [[[0.49174730, 0.89440989, 0.36116209],
        #         [0.49019824, 0.18241053, 0.81059146]],

        #         [[0.33344683, 0.26162022, 0.12241719],
        #         [0.51895050, 0.13092809, 0.55165734]]])

        # Ellipsis batch matrix multiplication
        print(paddlenlp.ops.einsum('...jk, ...kl->...jl', A,B))
        # Tensor(shape=[2, 3, 3], dtype=float64, place=CUDAPlace(0), stop_gradient=True,
        # [[[0.13654339, 0.39331432, 0.65059661],
        #     [0.07171420, 0.57518653, 0.77629221],
        #     [0.21250688, 0.37793541, 0.73643411]],

        #     [[0.56925339, 0.65859030, 0.57509818],
        #     [0.30368265, 0.25778348, 0.21630400],
        #     [0.39587265, 0.58031243, 0.51824755]]])
    """

    def _mul_sum(left, right, sum_dims):
        assert left.rank() == right.rank(), "number of rank should be equal."
        if len(sum_dims) == 0:
            return left * right
        sum_dims_set = set(sum_dims)
        batch_dims = []
        left_out_dims = []
        right_out_dims = []
        batch_size = summed_size = left_size = right_size = 1
        dim = len(left.shape)
        for i in range(dim):
            is_left_summed_dim = left.shape[i] > 1  # not broadcast dim
            is_right_summed_dim = right.shape[i] > 1
            if i in sum_dims_set:
                if is_left_summed_dim and is_right_summed_dim:
                    assert left.shape[i] == right.shape[
                        i], "Non-brocast dim should be equal."
                    summed_size *= left.shape[i]
                elif is_left_summed_dim:
                    left = left.sum(axis=i, keepdim=True)
                elif is_right_summed_dim:
                    right = right.sum(axis=i, keepdim=True)
            elif is_left_summed_dim and is_right_summed_dim:
                assert left.shape[i] == right.shape[
                    i], "Non-brocast dim should be equal."
                batch_dims.append(i)
                batch_size *= left.shape[i]
            elif is_left_summed_dim:
                left_out_dims.append(i)
                left_size *= left.shape[i]
            else:
                right_out_dims.append(i)
                right_size *= right.shape[i]
        out_shape = [left.shape[i] for i in batch_dims + left_out_dims]
        out_shape.extend([1] * len(sum_dims))
        out_shape.extend([right.shape[i] for i in right_out_dims])

        left_perm = list(batch_dims)
        left_perm.extend(left_out_dims)
        left_perm.extend(sum_dims)
        left_perm.extend(right_out_dims)

        right_perm = list(batch_dims)
        right_perm.extend(sum_dims)
        right_perm.extend(right_out_dims)
        right_perm.extend(left_out_dims)

        output_perm = [-1] * (len(batch_dims) + len(left_out_dims) +
                              len(sum_dims) + len(right_out_dims))
        for i, j in enumerate(batch_dims + left_out_dims + sum_dims +
                              right_out_dims):
            output_perm[j] = i

        left = paddle.reshape(
            paddle.transpose(
                left, perm=left_perm), (batch_size, left_size, summed_size))
        right = paddle.reshape(
            paddle.transpose(
                right, perm=right_perm), (batch_size, summed_size, right_size))
        result = paddle.matmul(left, right)
        result = paddle.reshape(result, out_shape)
        result = paddle.transpose(result, output_perm)
        return result




    if len(operands) == 1 and isinstance(operands[0], (list, tuple)):
        operands = operands[0]
    # Equation is case insensitive
    equation = equation.lower().replace(' ', '')
    # 1. Parse the equation
    lhs, *rhs = equation.split('->') # eqns = equation.split("->")
    assert len(rhs) < 2, " Invalid equation: multiple `->` were found."
    rhs = rhs[0] if rhs else ''

    # Parse the input equation and get the list of extended op_labels for all the input operands
    # e.g. ['ij', 'i.', '.k']
    all_op_labels = parse_all_op_labels(lhs, operands)

    # Get maximum number of broadcast dimensions
    # e.g. 1 for ['ij', 'i.', '.k']
    n_bcast_dims = max(visit_op_labels(num_bcast_dims, all_op_labels, operands))

    # Get all available labels
    # e.g. '.ijk' for ['ij', 'i.', '.k']
    avail_labels = gather_avail_labels(all_op_labels)

    # Parse and infer output labels. The n_bcast_dims of dots would be added for sure.
    if rhs:
        output_labels = parse_output_labels(rhs, avail_labels, n_bcast_dims)
    else:
        output_labels = infer_output_labels(all_op_labels, n_bcast_dims)

    # Replace an operand with its diagonal in case it has duplicate labels
    proprocess = lambda l, o: diagonalize(l, o) if has_duplicated_labels(l) else (l, o)
    operands, all_op_labels = list(zip(visit_op_labels(proprocess, all_op_labels, operands)))

    # Append output labels with all combined labels. There will be not dots left in combined labels.
    # The labels in the resulting order will be used to guide the axes ordering in the subsequent
    # operand join operations.  
    combined_labels = ''.join(c for c in avail_labels if c not in output_labels)
    all_labels_ordered = output_labels + combined_labels

    # Build op axes maps
    op_axes = list(visit_op_labels(lambda labels: map_op_dims(labels, all_labels_ordered), \
                                   all_op_labels))

    # Actions start here. Sum up the operands following certain order 
    # which is up to an optional order decision algorithm
    work_queue = list()
    for op_subs, op in zip(op_subscripts_list, operands):
        work_queue.append([op_subs, op])
    
    subscripts, result = work_queue.pop()
    while work_queue:
        subscripts, result = join(subscripts, result, *work_queue.pop())
    
    return result
