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
  a.shape_block, b.shape_block = block, blockOut

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
      a.loop, b.loop = True, False
    elif  a.interval >  b.interval :
      c.shape[ c.a_spillover ] = a.interval;  
      a.loop, b.loop = False, True 
    else : #==
      c.shape[ c.a_spillover ] = a.interval;  
      a.loop, b.loop = False, False


  #now forward expand the block sizes
  #(this still will be padded later)
  #(read padding is not carried over here because there is no point: bank memory does not have to be aligned, only 
  # consecutive) 
  c.shape_block = np.zeros( z, dtype=int )
  c.shape_block[0] = 1.0
  for i in range(1,z) :
    c.shape_block[i] = c.shape[i-1] * c.shape_block[i-1]


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
  for i in range(z) : paddedBlock[i] = c.shape_block[i]
 
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
      if padding > 0 : print( "padding" + str(padding) )
 
      #adjust the current and downstream read order blocks to the new padded sizes
#      expand_block( paddedBlock, padding, r )

      #on first read dim padding has multiplicative effect: disrupts contiguous sm writes and not enough memory.
      #If it is also first write dim there is no padding anyhow, so at least the first write dim is write bank contiguous
      if r != 0 :  
        expand_block( paddedBlock, padding, i )
      #ob = paddedBlock[r]
      #for i in range(r,D) : paddedBlock[i] = paddedBlock[i] * (ob+padding) / ob

  c.block = paddedBlock

def expand_block( block, padding, where ) :
      ob = block[where]
      for i in range(where,len(block)) : block[i] = block[i] * (ob+padding) / ob

#Can validate now that sm write block is contiguous wrt bankN %, by iterating through any write block in write order.
#For that I need C) map


def map_write( s, a, b, c, out ) :
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
      sm = sm + c.block[ci] * index[wi]

  return sm 


def map_read( s, a, b, c, out ) :
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
      sm = sm + c.block[ci] * index[i]

  return sm


def reverse_permutation( p ) :
  rp = np.zeros(len(p), dtype=int)
  for i in range(len(p)) : rp[p[i]] = i
  return rp


def test_bank( N, bankN, a, b, c, out ) :
  print( ('write sm % bankN', 'sm'))
  for s in range(N):  #actually this could be multiple of N. !
    sm = map_write( s, a, b, c, out )
    print( sm % bankN, sm ) 

  print( ('read sm % bankN', 'sm'))
  for s in range(N):  
    sm = map_read( s, a, b, c, out )
    print( sm % bankN, sm ) 



def loops( N, bankN, a, b, c, out ) :
  #according to shape elements, not blocks (skip threads)

  #size of read interval block, the number of threads in thread block that can read (padded or not), not N
  #(padded only keeps global reads aligned) 
  TR = a.interval * a.block[a.spillover] 

  #outer scalar is a multiple of that size.
  #global or local positioning of spillover dimensions have a partial final iteration, for both r and w blocks,
  #and for both global read and global write. So each loop has two parts, last iteration partial.

  #contiguous scalars in partial iteration

  #count up a d block of global interval indices, like c block, where there is a virtual dimension and block next to the interval dim.
  #these can be counted up several times, so partial can happen several times. What changes in the partial is TR, the number of active
  #contiguous threads, that's it. Note there are two partials: to complete the dimension (global block), and to catch up to larger spillover
  #(read or write block match)
  #Note on global there are typically two completion loops, one for each spillover.
  #So each thread block has to know if it is a final completion iteration. So fuck what if both are the final completion 
  #iteration at the same time. In particular, what is the meaning of the write spillover in the read loop? Well nothing because
  #there is no completion then <wrong>. You do a completion if read spillover is also a write dim and vice versa, or if they are shared,
  #and the meaning is TR both times. For global, the question is the meaning of the second completion. It is cutting
  #down the outer loop of the rw block on the spillover dimension. So that interval dim is shorter depending on global block,
  #and whether we are in the read or write loop. If a spillover is in other, the dim is not part of global block.
  # Tricky: when spillover is shared, behavior in global partial, if read spill < write: basically the write spill changes here,
  #so its changing the inner completion loop dim. But basically can just make all inner spills completion loops: you have a 
  #spill interval, and you have a dim. when interval > dim, you do a partial TR, and otherwise loop adding interval.

  #the global block counter is always a product of dimensions, so get those first (d) with corresponding blocks 

  #>>>>This is all that's needed:
  #-d.indices, step (for spillover dims, jump several blocks - no need for new dim), block (read global) for counting the blocks,
  #as well as block write global - that's just from the read and write full blocks then (nothing in sm)
  #-a,b each spillover has a completion interval (e.g. whole dim, same as interval, same as larger spill, or somewhere in between if partial block - whether or not its the tricky case)
  #-global passes down partial as new completion intervals, whichever or both spillovers happen to be partial in the thread block

  #remove all dims in left r or w block. In the remaining dims, set step of spillover dim (largest if same)

  # not(inner b)  and  not(inner a)
  d = struct()
  notb = np.array(out)[b.spillover:]
  d.indices = notb[ np.where(notb >= a.spillover) ]
  d.indices = np.extract(notb >= a.spillover, notb)
  d.steps = np.ones(len(d.indices), dtype=int)

  #set steps to spillover if in remaining dims
  if a.spillover in d.indices :
    aspi, = np.where( d.indices == a.spillover )  #I don't want to select first index and tuple every time 
    aspi = list(d.indices).index(a.spillover)
    if a.spillover  ==  out[b.spillover] :
      d.steps[aspi] = max( a.interval, b.interval )
    else :
      d.steps[aspi] = a.interval 
  if b.spillover in d.indices  and  a.spillover  !=  out[b.spillover] :  
    bspi = list(d.indices).index(b.spillover)
    d.steps[bspi] = b.interval 

  #size of global block
  GN = 1
  for i, ii in enumerate(d.indices) :
    GN = GN * int(np.ceil( 1.0 * shape[ii] / d.steps[i] ))
  d.GN = GN

 
  def blocken( indices, steps ) :
    block = np.zeros(len(indices), dtype=int)
    block[0] = 1
    for i, ii in enumerate(indices[:-1]) :
      block[i+1] = block[i] * int(np.ceil( 1.0 * shape[ii] / steps[i] ))
    return block

  d.block = blocken( d.indices )

  def indexen( s, block ) :
    index = np.zeros(len(block), dtype=int)
    r = s
    for i in reversed(range(len(block))) :
      index[i] = r / block[i] 
      r = r % block[i]
    return index 

  #here the block is supposed to be larger than indices (e.g. full global read block)
  def scalar( block, indices, index ) :
    s = 0
    for i in range(len(indices)) :
      s = s + block[indices[i]] * index[i]
    return s 

  #scalar thread block to scalar global read
  def map_global_read( s ) :

  #*global block only, to break up s, and global read block to get new scalar
  #*create function that takes shape (or indices) and returns corresponding block 
  #*create function that takes block and splits up number into corresponding index values
  #create function that takes block, indices and values and returns scalar for those
  #create block for global read
  #
  index = np.zeros(len(d.indices), dtype=int)
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
      sm = sm + c.block[ci] * index[i]

  return sm




