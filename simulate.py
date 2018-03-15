#threadblock size is assumed to be number of warps in a SM * 32, the total parallel cores per shared mem thread block
#no funny name business, just numbers for the permutation. But lets assume left major ordering, I'm tired of doing these
#reverse loops.
#divisors of N in increasing order

import numpy as np
class struct : pass

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

  #If spillover dimensions are the same, need gcd for interval (always possible when N is power of two).
  #This requires an extra loop over at least one of the blocks, to iterate over the 
  #multiple of N, in addition to the complementary outer loop.
  # If a spillover dimension is a shared non-spillover dimension in the other block, the interval is the 
  #full dimension and the extra loop last iteration is partial because the spillover won't divide the dim. 


#  if a.spillover == out[b.spillover] :
#    assert( a.interval % b.interval == 0  or  b.interval % a.interval == 0 ) #require power of 2, one is divisor of the other
#    a.interval, b.interval = max(a.interval, b.interval), max(a.interval, b.interval)
#    #now one interval block may be a multiple of N. The size is interval * block(spillover) 

  # Instead of gcd above could also just do partial loop on the smaller spillover. Actually this is simpler
  #and I can just keep the intervals as is.
  rout = reverse_permutation( out )
  if rout[a.spillover] <=     b.spillover  and a.interval < b.interval :  #a spillover shared, and a needs loop
    pass
  if      a.spillover  >= out[b.spillover] and a.interval > b.interval :  #b spillover shared, and b needs loop
    pass

  c = struct()
  combined = np.zeros( D, dtype=int )
  z = 0
  #indices of combined read write block, wrt original indices, ordered small to large. Noting the positions of the spillover dims.
  for i in range(D) :
    if i <= a.spillover  or  rout[i] <= b.spillover :
      combined[z] = i
      #where in combined array are the spillover dims
      if i       == a.spillover : c.a_spillover = z  
      if rout[i] == b.spillover : c.b_spillover = z
      z = z + 1 
  c.indices = combined[:z]

  #the final shape of combined rw
  c.shape = np.zeros( z, dtype=int )
  for i in range(z) : c.shape[i] = shape[c.indices[i]]
  if rout[a.spillover] == b.spillover :
    if    a.interval <  b.interval :
      c.shape[ c_a_spillover ] = b.interval;  
      a.loop, b.loop = true, false
    elif  a.interval >  b.interval :
      c.shape[ c.a_spillover ] = a.interval;  
      a.loop, b.loop = false, true 
    else : #==
      c.shape[ c.a_spillover ] = a.interval;  
      a.loop, b.loop = false, false
  #now forward expand the block sizes
  c.block = np.zeros( z, dtype=int )
  c.block[0] = 1.0
  for i in range(1,z) :
    c.block[i] = c.shape[i-1] * c.block[i-1]


  #the respective blocks are a.block up to a.spillover dim, with dim size a.interval.
  #To simulate, 
  return a, b,  c 


def bank_padding( a, b, c, out, bankN ) :
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
  z = len(c.indices)

  rout = reverse_permutation( out ) 

#  paddedBlock = np.zeros(D, dtype=int)
  paddedBlock = np.zeros(z, dtype=int)
#  for i in range(D) : paddedBlock[i] = a.block[i]
  for i in range(z) : paddedBlock[i] = c.block[i]
 
#  from pdb import set_trace; set_trace()

#  for r in range(D) : #i is global read dimension
  for i,r in enumerate(c.indices) : #i is global read dimension
    w = rout[r] 
    if w <= b.spillover :  #r is a write block dimension

      #pad read order block to the left of r, paddedBlock[r], so result is b.block[w] shift
#      # need  b.block[w]  % bankN  ==  (padded)a.block[r] + x % bankN 
      # need  b.block[w]  % bankN  ==  (padded)c.block[r] + x % bankN 
#      padding = (b.block[w] - paddedBlock[r])  %  bankN   # % precedence is high
      padding = (b.block[w] - paddedBlock[i])  %  bankN   # % precedence is high
    
      if padding < 0 : 
        from pdb import set_trace; set_trace()
      assert( padding >= 0 )  #never trust % 
 
      #adjust the current and downstream read order blocks to the new padded sizes
#      expand_block( paddedBlock, padding, r )
      expand_block( paddedBlock, padding, i )
      #ob = paddedBlock[r]
      #for i in range(r,D) : paddedBlock[i] = paddedBlock[i] * (ob+padding) / ob

  return paddedBlock 

def expand_block( block, padding, where ) :
      ob = block[where]
      for i in range(where,len(block)) : block[i] = block[i] * (ob+padding) / ob

#Can validate now that sm write block is contiguous wrt bankN %, by iterating through any write block in write order.
#For that I need C) map


def map_write( s, a, b, c, out, paddedBlock ) :
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
#  for i in range(b.spillover+1):
#    sm = sm + paddedBlock[out[i]] * index[i]

  
  for ci, i in enumerate(c.indices) :
    #exhausting: if read order index i (at position ci in combined block and wi in write block) is part of the write block,
    if i in out[ :b.spillover+1] :
      wi = np.argmax( i == np.array(out) )  #position in out
      #add write block index wi as multiple of paddedBlock at position ci 
      sm = sm + paddedBlock[ci] * index[wi]

  return sm 


def map_read( s, a, b, c, out, paddedBlock ) :
  index = np.zeros(a.spillover+1, dtype=int)
  r = s
  for i in range(a.spillover, -1, -1) : 
    index[i] = r / a.block[i]
    r = r % a.block[i]

  #sm scalar entry  (leaving read only indexes at 0 for now)
  sm = 0
#  for i in range(a.spillover+1):
#    sm = sm + paddedBlock[i] * index[i]

  for ci, i in enumerate(c.indices) :
    #exhausting: if read order index i (at position ci in combined block and i in read block) is part of the read block,
    if i <= a.spillover : 
      #add read block index i as multiple of paddedBlock at position ci (even that simplifies as ci == i for read block)
      sm = sm + paddedBlock[ci] * index[i]

  return sm


def reverse_permutation( p ) :
  rp = np.zeros(len(p), dtype=int)
  for i in range(len(p)) : rp[p[i]] = i
  return rp


def test_bank( N, bankN, a, b, c, out, paddedBlock ) :
  print( ('write sm % bankN', 'sm'))
  for s in range(N):  #actually this could be multiple of N. !
    sm = map_write( s, a, b, c, out, paddedBlock )
    print( sm % bankN, sm ) 

  print( ('read sm % bankN', 'sm'))
  for s in range(N):  
    sm = map_read( s, a, b, c, out, paddedBlock )
    print( sm % bankN, sm ) 

