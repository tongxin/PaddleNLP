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

import paddle

__all__ = ['einsum']


def einsum(equation, *operands):
    r"""
    Executes the sum of product of provided operands based on the Einstein summation convention.
    Einsum can be used to complete a variety of operations, such as sum, transpose,
    batch matrix multiplication.

    Args:
        equation (`str`):
            Uses uncased letters to specify the dimension of the operands and result. The input
            equation is on the left hand before `->` while the output equation is on the right side.
            Einsum can infer the result shape so that the `->` and the result label letters can be omitted.
            Operands in the input equation are splited by commas (','), e.g. 'abc,cde' describes two 3D
            operands. The dimensions labeled with same letter should be same or be 1. Ellipsis ('...') can
            be used to specify the broadcast dimensions.

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

    def expand_eqn_lhs(lhs, operands):
        '''
        Parse the left hand side of the input equation, returning expanded subscripts 
        for each input operand, where the subscripts' length equals the operand tensor's rank.
        Along with subscripts string, the number of ellipsis dimensions is also returned 
        '''
        op_eqn_list = lhs.split(',') # operand_eqns = input_eqn.split(",")
        # Sanity checks 
        assert all(c == '.' or c == ',' for c in lhs if not c.isapha()) "Invalid equation: no special characters other than ',' and '.' are allowed in the equation."
        assert len(op_eqn_list) == len(operands) "Invalid equation: the subscripts groups do not match the number of input operands."
        assert any(not s for s in op_eqn_list) "Invalid equation: subscripts split by ',' got empty strings."
        # Expand the equation for each input operand 
        for eqn, op in zip(op_eqn_list, operands):
            dot_pos, ell_pos = eqn.find('.'), eqn.find('...')
            # Note, the order of the following two asserts matters
            assert dot_pos == ell_pos "Invalid equation: equation includes '.' but no '...'."
            assert ell_pos >= 0 and eqn.find('.', ell_pos+3) < 0 "Invalid equation: equation includes '.' behind '...'."
            op_rank = int(op.rank().numpy())
            ell_rank = 0
            if ell_pos >= 0:
                ell_rank = op_rank - len(eqn) + 3
                op_subscripts = ('.' * ell_rank).join(eqn.split('...'))
            # return a tuple of expanded subscripts, ellipsis masked rank for each operand
            yield op_subscripts, ell_rank

    def expand_eqn_rhs(eqn, ell=False):
        # Sanity check. only alphabet is allowed if not '.'
        assert all(c == '.' for c in eqn if not c.isalpha()) "Invalid equation: non-alphabet char is found other than '.'."
        
        # Syntax sanity check
        dot_pos, ell_pos = rhs.find('.'), eqn.find('...')
        assert dot_pos == ell_pos "Invalid equation: equation includes '.' but no '...'."
        assert ell_pos >= 0 and eqn.find('.', ell_pos+3) < 0 "Invalid equation: equation includes '.' behind '...'."

        out_subscripts = eqn
        if ell:
            # None-zero ell_rank value implies non-zero hidden output dimensions, 
            # and that in turn requires including an '...' in rhs in case rhs is provided.
            assert ell_pos >= 0 "Invalid equation: output has more dimensions than what rhs provides, missing '...'."
            out_subscripts = '.'.join(eqn.split('...')
        return out_subscripts

    def join(x_subs, x, y_subs, y, out_subs=out_subscripts):
        '''
        Joins two tensor operands. The out_subs is referenced for querying output subscripts and their order

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

    if len(operands) == 1 and isinstance(operands[0], (list, tuple)):
        operands = operands[0]
    # Equation is case insensitive
    equation = equation.lower().replace(' ', '')
    # 1. Parse the equation
    lhs, *rhs = equation.split('->') # eqns = equation.split("->")
    assert len(rhs) < 2, " Invalid equation: multiple `->` were found."
    rhs = rhs[0] if rhs else ''
    # Parse the input equation
    op_subscripts_list = list(expand_eqn_lhs(lhs, operands))
    # TODO: diagnalize 

    # collect the set of subscripts not shared across operands and build initial output subscripts
    subscript_count = dict()
    for op_subs in op_subscripts_list:
        for ch in set(op_subs):
            subscript_count[ch]++
    outset = set(ch for ch in subscript_count if subscript_count[ch] == 1)
    out_subscripts = ''.join(outset)
    if rhs:
        # check output subscripts must appear in the left hand side of the equation
        output_chars = set(rhs).remove('.')
        assert all(lhs.find(ch) >= 0 for ch in output_chars)
            "Invalid equation: subscripts in right hand side of the equation must appear in the left hand side."
        # check no subscripts appear more than once 
        assert all(rhs.count(ch) == 1 for ch in output_chars)
            "Invalid equation: right hand side includes the same subscript multiple times."
        ell_found = any(r > 0 for _, r in op_subscripts_list)
        out_subscripts = expand_eqn_rhs(rhs, ell_found)

    # Actions start here. Sum up the operands following certain order 
    # which is up to an optional order decision algorithm
    work_queue = list()
    for op_subs, op in zip(op_subscripts_list, operands):
        work_queue.append([op_subs, op])
    
    subscripts, result = work_queue.pop()
    while work_queue:
        subscripts, result = join(subscripts, result, *work_queue.pop())
    
    return result
