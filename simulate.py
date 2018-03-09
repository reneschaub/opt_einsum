#threadblock size is assumed to be number of warps in a SM * 32, the total parallel cores per shared mem thread block
#no funny name business, just numbers for the permutation. But lets assume left major ordering, I'm tired of doing these
#reverse loops.
#divisors of N in increasing order

import numpy as np

def rwblock( shape, out, N, divisors ) :
  assert(divisors[0] == 1)  #need 1 in case first dim is spillover dim

  #calculate all block sizes

  D = len(shape)
  #block[i] indicates size of the block to the left of dimension i
  block, blockOut = np.zeros(D, dtype=int), np.zeros(D, dtype=int)
  block[0] = 1.0
  for i in range(1,D) :
    block[i] = shape[i-1] * block[i-1]

  blockOut[0] = 1.0
  for i in range(1,D) :
    blockOut[i] = shape[out[i-1]] * blockOut[i-1]
  
  #spillover dimensions

  spillover = np.where( block >= N )[0][0]  #returns a list of arrays, each for one index
  spillover = np.argmax( block >= N ) #same: the max is '1', and arg returns the first occurrence
  spilloverOut = np.where( blockOut >= N )[0][0]  #wrt out order
  #if N is smaller than first dimension, the spillover dim is the first dim, everything works same

  #so now the tensor should be block padded to a divisor of N, at the block P next to spillover.
  p = spillover-1  #to get the next smaller block not spilling over 
  padded = divisors[ np.argmax( divisors >= block[p] ) ]
  padding = padded - block[p]

  return padding, padded, p 
