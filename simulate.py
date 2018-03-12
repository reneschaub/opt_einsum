#threadblock size is assumed to be number of warps in a SM * 32, the total parallel cores per shared mem thread block
#no funny name business, just numbers for the permutation. But lets assume left major ordering, I'm tired of doing these
#reverse loops.
#divisors of N in increasing order

import numpy as np

def rwblock( shape, out, N, divisors, block=None ) :
  assert(divisors[0] == 1)  #need 1 in case first dim is spillover dim
  #calculate all block sizes

  D = len(shape)
  #block[i] indicates size of the block to the left of dimension i
  if block is None :
    block = np.zeros(D, dtype=int)
    block[0] = 1.0
    for i in range(1,D) :
      block[i] = shape[i-1] * block[i-1]

  blockOut = np.zeros(D, dtype=int)
  blockOut[0] = 1.0
  for i in range(1,D) :
    blockOut[i] = shape[out[i-1]] * blockOut[i-1]
  
  #spillover dimensions

  def calc( block ) :
    class struct : pass
    r = struct()


    r.spillover = -1 + np.where( block >= N )[0][0]   #returns a list of arrays, each for one index
    r.spillover = -1 + np.argmax( block >= N )  #same: the max is '1', and arg returns the first occurrence
    #everything still works out if first dim is spillover dim

    #so now the tensor should be block padded to a divisor of N, at the block P next to spillover.
    def Padding( p ) :  #block size to the left of dim p 
      padded = divisors[ np.argmax( divisors >= block[p] ) ]  #first divisor as big as block to left of p
      return padded - block[p]
    r.padding    = Padding( r.spillover )

    #finally, size of spillover interval
    r.paddedBlock = r.padding + block[r.spillover]
    r.interval =  N / r.paddedBlock
    r.block =  block

    #note spillover dimension is wrt to block order
    #return spillover, padding, paddedBlock, interval,  block
    return r 

  a, b = calc(block), calc(blockOut)

  #but if spillover dimensions are the same, need gcd for interval (always possible when N is power of two)
  if a.spillover == out[b.spillover] :
    assert( a.interval % b.interval == 0  or  b.interval % a.interval == 0 ) #require power of 2, one is divisor of the other
    a.interval, b.interval = max(a.interval, b.interval), max(a.interval, b.interval)
    #now one interval block may be a multiple of N. The size is interval * block(spillover) 

  return a, b


def bank_padding( a, b, out, bankN ) :
  #The write interval block in global write order gives the needed result of padding: e.g. ABCD, A needs +1 bank modulo shift for 
  #each increment, each increment of B needs A+1 shift, each increment of C needs AB+1 shift, etc.
  # Padding is only needed for the write interval block, which is a multiple of N.  
  # Padding is calculated in the global read order of the write block, including interspersed read dimensions. Whether a dim is shared is 
  #irrelevant. 
  #E.g. mDnBoCA. D is first and needs ABC+1 (modulo Banks) shift, so we pad m like so. Then B needs a CD+1 shift, so pad the mDn block.
  #Note that padded Tensors are defined by block strides, not individual dimensions.

  #blockOut already has the strides, needed up to spillover dim (the write block).
 
  #Scan read order dimensions for write dimensions: map from input to output dim (out is map from output to input dim)

  #Note: even if I didn't pad the input tensor so interval blocks are aligned with N, I can pretend they are and just idle threads
  #at the intended zero padding addresses, while using the true block strides. Simply means 2 accesses per global read.
  #whether or not they are aligned follows from the wrblock() padding number (assuming I run that twice, once on unpadded then padded tensor). 
  D = len(a.block)

  rout = np.zeros(D, dtype=int)
  for i in range(D) : rout[out[i]] = i

  paddedBlock = np.zeros(D, dtype=int)
  for i in range(D) : paddedBlock[i] = a.block[i]
 
#  from pdb import set_trace; set_trace()
 
  for r in range(D) : #i is global read dimension
    w = rout[r] 
    if w <= b.spillover :  #r is a write block dimension

      #pad read order block to the left of r, paddedBlock[r], so result is b.block[w] shift
      # need  b.block[w]  % bankN  ==  (padded)a.block[r] + x % bankN 
      padding = (b.block[w] - paddedBlock[r])  %  bankN   # % precedence is high
    
      if padding < 0 : 
        from pdb import set_trace; set_trace()
      assert( padding >= 0 )  #never trust % 
 
      #adjust the current and downstream read order blocks to the new padded sizes
      expand_block( paddedBlock, padding, r )
#      ob = paddedBlock[r]
#      for i in range(r,D) : paddedBlock[i] = paddedBlock[i] * (ob+padding) / ob

  return paddedBlock 

def expand_block( block, padding, where ) :
      ob = block[where]
      for i in range(where,len(block)) : block[i] = block[i] * (ob+padding) / ob

#Can validate now that sm write block is contiguous wrt bankN %, by iterating through any write block in write order.
#For that I need C) map

def map_write( s, a, b, out, paddedBlock ) :
  #get scalar sm entry corresponding to all indices 

  #split s into write block indices, given fixed read-only dimensions of the rw block
   
  #split s into write block indices
#  from pdb import set_trace; set_trace()
  index = np.zeros(b.spillover+1, dtype=int)
  r = s
  for i in range(b.spillover, -1, -1) :
    index[i] = r / b.block[i]
    r = r % b.block[i]

  #sm scalar entry  (leaving read only indexes at 0 for now)
  sm = 0
  for i in range(b.spillover+1):
    sm = sm + paddedBlock[out[i]] * index[i]

  return sm 


def map_read( s, a, b, out, paddedBlock ) :
  index = np.zeros(a.spillover+1, dtype=int)
  r = s
  for i in range(a.spillover, -1, -1) : 
    index[i] = r / a.block[i]
    r = r % a.block[i]

  #sm scalar entry  (leaving read only indexes at 0 for now)
  sm = 0
  for i in range(a.spillover+1):
    sm = sm + paddedBlock[i] * index[i]

  return sm






def test_bank( N, bankN, a, b, out, paddedBlock ) :
  print( ('write sm % bankN', 'sm'))
  for s in range(N):  #actually this could be multiple of N. !
    sm = map_write( s, a, b, out, paddedBlock )
    print( sm % bankN, sm ) 

  print( ('read sm % bankN', 'sm'))
  for s in range(N):  
    sm = map_read( s, a, b, out, paddedBlock )
    print( sm % bankN, sm ) 

