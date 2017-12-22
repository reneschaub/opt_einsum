import numpy as np
from . import paths
#from accelerate.cuda.blas import Blas   not certified with latest numba versions
from pyculib.blas import Blas

from numba import cuda, float32 #, int32
import setperm


bla =  Blas()

def can_blas(inputs, result, idx_removed):
    """
    Checks if we can use a BLAS call.

    Parameters
    ----------
    inputs : list of str
        Specifies the subscripts for summation.
    result : str
        Resulting summation.
    idx_removed : set
        Indices that are removed in the summation


    Returns
    -------
    type : str or bool
        The type of BLAS call to be used or False if none.

    Notes
    -----
    We assume several operations are not efficient such as a transposed
    DDOT, therefore 'ijk,jki->' would return False.

    Examples
    --------
    >>> _can_blas(['ij', 'jk'], 'ik', set('j'))
    'GEMM'

    >>> _can_blas(['ijj', 'jk'], 'ik', set('j'))
    False

    """

    # Gotta remove indices
    if len(idx_removed) == 0:
        return False

    # Can only do two
    if len(inputs) != 2:
        return False

    # Make sure there is overlap
    if len(set(inputs[0]) & set(inputs[1])) == 0:
        return False

    # Build a few temporaries
    sets = [set(x) for x in inputs]
    keep_left = sets[0] - idx_removed
    keep_right = sets[1] - idx_removed
    input_left = inputs[0]
    input_right = inputs[1]
    rs = len(idx_removed)

    if any(len(l) != len(s) for l, s in zip(inputs, sets)):
        return False

    # Cannot handle partial inner
    if len(keep_left & keep_right):
        return False

    # DDOT
    elif inputs[0] == inputs[1]:
        return 'DOT'

    # DDOT doesnt make sense if you have to tranpose
    elif sets[0] == sets[1]:
        return False

    # GEMM no transpose
    elif input_left[-rs:] == input_right[:rs]:
        return 'GEMM'

    # GEMM transpose both
    elif input_left[:rs] == input_right[-rs:]:
        return 'GEMM'

    # GEMM transpose right
    elif input_left[-rs:] == input_right[-rs:]:
        return 'GEMM'

    # GEMM tranpose left
    elif input_left[:rs] == input_right[:rs]:
        return 'GEMM'

    # Einsum is faster than vectordot if we have to copy
    elif (len(keep_left) == 0) or (len(keep_right) == 0):
        return False

    # Conventional tensordot
    else:
        return 'TDOT'


def tensor_blas(view_left, input_left, view_right, input_right, index_result, idx_removed):
    """
    Computes the dot product between two tensors, attempts to use np.dot and
    then tensordot if that fails.

    Parameters
    ----------
    view_left : array_like
        The left hand view
    input_left : str
        Indices of the left view
    view_right : array_like
        The right hand view
    input_right : str
        Indices of the right view
    index_result : str
        The resulting indices
    idx_removed : set
        Indices removed in the contraction

    Returns
    -------
    type : array
        The resulting BLAS operation.

    Notes
    -----
    Interior function for tensor BLAS.

    GPU device arrays must be in C contiguous order

    This function will attempt to use `np.dot` by the iterating through the
    four possible transpose cases. If this fails all inner and matrix-vector
    operations will be handed off to einsum while all matrix-matrix operations will
    first copy the data, perform the DGEMM, and then copy the data to the required
    order.

    Examples
    --------

    >>> a = np.random.rand(4, 4)
    >>> b = np.random.rand(4, 4)
    >>> tmp = tensor_blas(a, 'ij', b, 'jk', 'ik', set('j'))
    >>> np.allclose(tmp, np.dot(a, b))

    """

    idx_removed = set(idx_removed)
    keep_left = set(input_left) - idx_removed
    keep_right = set(input_right) - idx_removed
    

    # We trust this must be called correctly
    dimension_dict = {}
    for i, s in zip(input_left, view_left.shape):
        dimension_dict[i] = s
    for i, s in zip(input_right, view_right.shape):
        dimension_dict[i] = s

    # Do we want to be able to do this?

    # Check for duplicate indices, cannot do einsum('iij,jkk->ik') operations here
    # if (len(set(input_left)) != len(input_left)):
    #     new_inds = ''.join(keep_left) + ''.join(idx_removed)
    #     view_left = np.einsum(input_left + '->' + new_inds, view_left, order='C')
    #     input_left = new_inds

    # if (len(set(input_right)) != len(input_right)):
    #     new_inds = ''.join(idx_removed) + ''.join(keep_right)
    #     view_right = np.einsum(input_right + '->' + new_inds, view_right, order='C')
    #     input_right = new_inds

    # Tensordot guarantees a copy for ndim > 2, should avoid skip if possible
    rs = len(idx_removed)  
   
    dim_left = paths.compute_size_by_dict(keep_left, dimension_dict)
    dim_right = paths.compute_size_by_dict(keep_right, dimension_dict)
    dim_removed = paths.compute_size_by_dict(idx_removed, dimension_dict)
  
    tensor_result = input_left + input_right
    for s in idx_removed:
        tensor_result = tensor_result.replace(s, "")


    #Permutate indices to a default order that can be handled by gemm matrix multiplication, 
    #if contraction indices are not same block at beginning or end of inputs.
    #Formerly tensordot fallback cases
    result_left, result_right = input_left, input_right
    for s in idx_removed:
        result_left, result_right = result_left.replace(s, ""), result_right.replace(s, "")
    # Find indices to contract over
    left_pos, right_pos = np.zeros( rs, dtype=int ), np.zeros( rs, dtype=int ) 
    for num, s in enumerate(idx_removed):
        left_pos[num], right_pos[num] = input_left.find(s), input_right.find(s)
    order_removed = ''.join(idx_removed)  #arbitrary but fixed sequence of removed indices
   
    result_left, result_right = result_left + order_removed, result_right + order_removed 

    if (left_pos - right_pos).min() != (left_pos - right_pos).max() :   #not shifted versions of each other
        #default both
        import pdb; pdb.set_trace()
        hold = cuda.device_array_like( view_right ) #!cache this for repeated calls
        setperm.perm( view_right, input_right, hold, result_right )
        input_right, view_right = result_right, hold

        hold = cuda.device_array_like( view_left ) #!cache this for repeated calls
        setperm.perm( view_left, input_left, hold, result_left )
        input_left, view_left = result_left, hold

    else: 
        cs = left_pos.sum()
        ln = len(input_left) 
        if  rs*(rs-1)/2  !=  cs  and  (2*(ln-1)-(rs-1))*rs/2  !=  cs :  #removed indices not at beginning or end
            #default left
            hold = cuda.device_array_like( view_left ) #!cache this for repeated calls
            setperm.perm( view_left, input_left, hold, result_left )
            input_left, view_left = result_left, hold
        cs = right_pos.sum()
        ln = len(input_right) 
        if  rs*(rs-1)/2  !=  cs  and  (2*(ln-1)-(rs-1))*rs/2  !=  cs :  
            #default right
            hold = cuda.device_array_like( view_right ) #!cache this for repeated calls
            setperm.perm( view_right, input_right, hold, result_right )
            input_right, view_right = result_right, hold

    D = cuda.device_array( shape=(dim_left, dim_right), dtype=np.float32, order='F' )

    # This is ugly, but can vastly speed up certain operations
    # Vectordot
    #-need to support this, but also potentially a bug: what if some indices are fixed? 
    if input_left == input_right:
        new_view = np.dot(view_left.ravel(), view_right.ravel())

    # Matrix multiply
    # No transpose needed
    elif input_left[-rs:] == input_right[:rs]:
        print('ij,jk')

        #General example works for all: ijkl,klmn->ijmn. C order implies lkji,nmlk viewed as F order. 
        #gemm-transpose both results in jilk,lknm arguments. Index order within contracting block is irrelevant.
        #gemm result is jinm in F order, which is mnij in C, therefor final transpose gives ijmn.

        #!!!fix anac bug here too, and all other cases
        #view C as F arrays with flipped blocks
        d_lv = view_left.reshape( dim_removed, dim_left, order='F' )
        d_rv = view_right.reshape( dim_right, dim_removed, order='F' )

        #flip transpose for C order. Gemm interprets as F order, ie mirror order and strides
        bla.gemm('T', 'T', dim_left, dim_right, dim_removed, 1.0, d_lv, d_rv, 0.0, D)  
        new_view = c_mirror_transpose_back( D, dim_left, dim_right )
 
#        new_view = np.dot(view_left.reshape(dim_left, dim_removed),
#                          view_right.reshape(dim_removed, dim_right))

    # Transpose both
    elif input_left[:rs] == input_right[-rs:]:
        print('ij,ki transp both')

        #view C as F arrays with flipped blocks
        lv = view_left.reshape(dim_left, dim_removed, order='F')
        rv = view_right.reshape(dim_removed, dim_right, order='F')

        bla.gemm('N', 'N', dim_left, dim_right, dim_removed, 1.0, lv, rv, 0.0, D)  
        new_view = c_mirror_transpose_back( D, dim_left, dim_right )

#        new_view = np.dot(view_left.reshape(dim_removed, dim_left).T,
#                          view_right.reshape(dim_right, dim_removed).T)

    # Transpose right
    elif input_left[-rs:] == input_right[-rs:]:
        print('ij,kj transp right')

        #view C as F arrays with flipped blocks
        lv = view_left.reshape( dim_removed, dim_left, order='F' )
        rv = view_right.reshape( dim_removed, dim_right, order='F' )

        bla.gemm('T', 'N', dim_left, dim_right, dim_removed, 1.0, lv, rv, 0.0, D)  
        new_view = c_mirror_transpose_back( D, dim_left, dim_right )

#        new_view = np.dot(view_left.reshape(dim_left, dim_removed),
#                          view_right.reshape(dim_right, dim_removed).T)

    # Tranpose left
    elif input_left[:rs] == input_right[:rs]:
        print( 'ij,ik transp left' )

        #view C as F arrays with flipped blocks
        lv = view_left.reshape( dim_left, dim_removed, order='F' )
        rv = view_right.reshape( dim_right, dim_removed, order='F' )

        bla.gemm('N', 'T', dim_left, dim_right, dim_removed, 1.0, lv, rv, 0.0, D)  
        #!ANACONDA BUG: if shape remains same (here left=right), it won't convert from F to C. So flatten first.
        D_C = D.reshape( dim_right * dim_left, order='F' )  
        D_C = D_C.reshape( dim_right, dim_left, order='C' )  #flipped blocks as C order
        D_C = cuda.kernels.transpose.transpose( D_C )  #transpose those two blocks
        new_view = D_C

#        new_view = np.dot(view_left.reshape(dim_removed, dim_left).T,
#                          view_right.reshape(dim_removed, dim_right))

    # Conventional tensordot
    else:
        assert(False)
#        print( 'tensordot' )
#        # Find indices to contract over
#        left_pos, right_pos = (), ()
#        for s in idx_removed:
#            left_pos += (input_left.find(s),)
#            right_pos += (input_right.find(s),)
#        new_view = np.tensordot(view_left, view_right, axes=(left_pos, right_pos))

    # Make sure the resulting shape is correct
    tensor_shape = tuple(dimension_dict[x] for x in tensor_result)
    if (new_view.shape != tensor_shape):
        if (len(tensor_result) > 0):
            print( 'shape set' )
            #new_view.shape = tensor_shape  #!this does not work for device array, also need strides
            new_view = new_view.reshape( tensor_shape, order='C' )
        else:
            print( 'result set empty wtf' )
            new_view = np.squeeze(new_view)

    if tensor_result != index_result:
        print( 'permutate end result' )
        assert( len(tensor_result) == len(index_result) )  #just in case this is an open pit
        hold = cuda.device_array_like( new_view ) #!cache this for repeated calls
        #important that new_view shape was correctly set to tensor_shape above, for setperm
        setperm.perm( new_view, tensor_result, hold, index_result )
        new_view = hold

        #new_view = np.einsum(tensor_result + '->' + index_result, new_view)

    return new_view


def c_mirror_transpose_back( new_view, dim_left, dim_right ) :
  new_view = new_view.reshape( dim_right * dim_left, order='F' )   #anaconda bug 
  new_view = new_view.reshape( dim_right, dim_left, order='C' )  #flipped blocks as C order
  new_view = cuda.kernels.transpose.transpose( new_view )  #transpose those two blocks
  return new_view

