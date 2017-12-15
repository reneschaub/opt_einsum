import numpy as np
import record
import math
import numpy.random as random

def  checkperm( A, shape_a, B, shape_b, out ) :

  A = A.reshape( shape_a )
  B = B.reshape( shape_b )

  N = 100
  elements =  np.zeros( (N, len(shape_a) ), dtype=int )
  #generate a number of random indices, and check the value of the permutation
  for i in range(N) :
    for j, m in enumerate( shape_a ) :
      elements[ i, j ] = random.randint( 0, m ) 
    
    permuted = elements[ i, out ] 
    assert( np.allclose( A[ tuple(elements[i,:]) ], B[ tuple(permuted) ] ) )

