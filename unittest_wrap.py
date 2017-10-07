from __future__ import division, absolute_import, print_function

import numpy as np
import pytest
from opt_einsum import contract
from numba import cuda, float32, int32
import permutate
import setperm

blas_tests = [
    # DOT
    ((['k', 'k'], '', set('k')),            'DOT'),  # DDOT
    ((['ijk', 'ijk'], '', set('ijk')),      'DOT'),  # DDOT

    # GEMV?

    # GEMM
    ((['ij', 'jk'], 'ik', set('j')),        'GEMM'), # GEMM N N
    ((['ijl', 'jlk'], 'ik', set('jl')),     'GEMM'), # GEMM N N Tensor
    ((['ij', 'kj'], 'ik', set('j')),        'GEMM'), # GEMM N T
    ((['ijl', 'kjl'], 'ik', set('jl')),     'GEMM'), # GEMM N T Tensor 
    ((['ji', 'jk'], 'ik', set('j')),        'GEMM'), # GEMM T N
    ((['jli', 'jlk'], 'ik', set('jl')),     'GEMM'), # GEMM T N Tensor
    ((['ji', 'kj'], 'ik', set('j')),        'GEMM'), # GEMM T T
    ((['jli', 'kjl'], 'ik', set('jl')),     'GEMM'), # GEMM T T Tensor

    # GEMM with final transpose
    ((['ij', 'jk'], 'ki', set('j')),        'GEMM'), # GEMM N N
    ((['ijl', 'jlk'], 'ki', set('jl')),     'GEMM'), # GEMM N N Tensor
    ((['ij', 'kj'], 'ki', set('j')),        'GEMM'), # GEMM N T
    ((['ijl', 'kjl'], 'ki', set('jl')),     'GEMM'), # GEMM N T Tensor 
    ((['ji', 'jk'], 'ki', set('j')),        'GEMM'), # GEMM T N
    ((['jli', 'jlk'], 'ki', set('jl')),     'GEMM'), # GEMM T N Tensor
    ((['ji', 'kj'], 'ki', set('j')),        'GEMM'), # GEMM T T
    ((['jli', 'kjl'], 'ki', set('jl')),     'GEMM'), # GEMM T T Tensor

   # Tensor Dot (requires copy), lets not deal with this for now
   ((['ilj', 'jlk'], 'ik', set('jl')),     'TDOT'), # FT GEMM N N Tensor
   ((['ijl', 'ljk'], 'ik', set('jl')),     'TDOT'), # ST GEMM N N Tensor
   ((['ilj', 'kjl'], 'ik', set('jl')),     'TDOT'), # FT GEMM N T Tensor 
   ((['ijl', 'klj'], 'ik', set('jl')),     'TDOT'), # ST GEMM N T Tensor 
   ((['lji', 'jlk'], 'ik', set('jl')),     'TDOT'), # FT GEMM T N Tensor
   ((['jli', 'ljk'], 'ik', set('jl')),     'TDOT'), # ST GEMM T N Tensor
   ((['lji', 'jlk'], 'ik', set('jl')),     'TDOT'), # FT GEMM T N Tensor
   ((['jli', 'ljk'], 'ik', set('jl')),     'TDOT'), # ST GEMM T N Tensor

   # Tensor Dot (requires copy), lets not deal with this for now with transpose
   ((['ilj', 'jlk'], 'ik', set('lj')),     'TDOT'), # FT GEMM N N Tensor
   ((['ijl', 'ljk'], 'ik', set('lj')),     'TDOT'), # ST GEMM N N Tensor
   ((['ilj', 'kjl'], 'ik', set('lj')),     'TDOT'), # FT GEMM N T Tensor 
   ((['ijl', 'klj'], 'ik', set('lj')),     'TDOT'), # ST GEMM N T Tensor 
   ((['lji', 'jlk'], 'ik', set('lj')),     'TDOT'), # FT GEMM T N Tensor
   ((['jli', 'ljk'], 'ik', set('lj')),     'TDOT'), # ST GEMM T N Tensor
   ((['lji', 'jlk'], 'ik', set('lj')),     'TDOT'), # FT GEMM T N Tensor
   ((['jli', 'ljk'], 'ik', set('lj')),     'TDOT'), # ST GEMM T N Tensor

   # Other
   ((['ijk', 'ikj'], '', set('ijk')),       False), # Transpose DOT
   ((['ijj', 'jk'], 'ik', set('j')),        False), # Double index
   ((['i', 'j'], 'ij', set()),              False), # Outer 
   ((['ijk', 'j'], 'ij', set()),            False), # Index sum 1
   ((['ijk', 'k'], 'ij', set()),            False), # Index sum 2
]


def test_blas_out():
    N = 4*5*6*7
    a = np.arange(N, dtype='float32').reshape(4,5,6,7,  order='C') 
    b = (np.arange(N, dtype='float32')+10).reshape(6,4,7,5,  order='C') 
    R = 6*7*6*7
    #R = 4*5*4*5
    #d = np.empty(R, dtype='float32').reshape(4,5,4,5,  order='C')
    d = np.empty(R, dtype='float32').reshape(6,7,6,7,  order='C')


#    a = np.random.rand(4, 5, 6, 7)
#    a = np.array( a, dtype=np.float32, order='C')
    d_a = cuda.to_device( a )

#    b = np.random.rand(4, 5, 6, 7)
#    b = np.array( b, dtype=np.float32, order='C')
    d_b = cuda.to_device( b )
#    c = np.random.rand(4, 5, 6, 7)
#    d = np.empty((6, 7, 6, 7))
#    d = np.array( d, dtype=np.float32, order='C')
    d_d = cuda.to_device( d )

#    contract('ij,jk->ik', a, b, out=d)
#
#    assert np.allclose(d, np.dot(a, b))
    
#    contract('ij,jk,kl', a, b, c, out=d)
#    assert np.allclose(d, np.dot(a, b).dot(c))


#opt_einsum first swaps the arguments, hence the mirror images
#1    contract('ij,ki->kj', d_a, d_b, out=d_d)   #none
#2    contract('ki,ij->jk', d_a, d_b, out=d_d)   #both
#3    contract('ki,ji->jk', d_a, d_b, out=d_d)   #right
#4    contract('ji,jk->ki', d_a, d_b, out=d_d)   #left
#    contract('ji,jk->ki', d_a, d_b, out=d_d)   #left
#    contract('ijkl,ijmn->mnkl', d_a, d_b, out=d_d)   #left
#5    contract('ijkl,ijmn->klmn', d_a, d_b, out=d_d)   #left
#6    contract('ijkl,mnkl->ijmn', d_a, d_b, out=d_d)   #right
#7    contract('ijkl,mnij->klmn', d_a, d_b, out=d_d)   #none
    contract('ijkl,minj->klmn', d_a, d_b, out=d_d)   #none
    d_d.copy_to_host( d )
#    validate =  np.einsum('ijkl,ijmn->klmn', a, b)
    v =  np.einsum('ijkl,minj->klmn', a, b)
#7    v =  np.einsum('ijkl,mnij->klmn', a, b)
#6    v =  np.einsum('ijkl,mnkl->ijmn', a, b)
#5    v =  np.einsum('ijkl,ijmn->klmn', a, b)
#    v =  np.einsum('ijkl,ijmn->mnkl', a, b)
#    assert np.allclose(d, np.einsum('ijkl,ijmn->mnkl', a, b))
#4    assert np.allclose(d.transpose(), np.dot(a.transpose(), b))
#3    assert np.allclose(d.transpose(), np.dot(a, b.transpose()))
#2    assert np.allclose(d.transpose(), np.dot(a, b))
#1    assert np.allclose(d.transpose(), np.dot(a.transpose(), b.transpose()))
#    assert np.allclose(d.transpose(), np.dot(a.transpose(), b.transpose()))

    import pdb; pdb.set_trace() 
    assert np.allclose(d, v)
    print('success')


test_blas_out()
