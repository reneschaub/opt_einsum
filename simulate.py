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

  #the final shape of combined rw shared memory, also intervals and completion intervals for counting through precalc
  c.shape = np.zeros( z, dtype=int )
  c.interval = np.zeros( z, dtype=int )
  c.completion = np.zeros( z, dtype=int )
  a.completion = np.zeros( a.spillover+1, dtype=int )
  b.completion = np.zeros( b.spillover+1, dtype=int )
  for i in range(z) : c.shape[i]      = shape[c.indices[i]]
  for i in range(z) : c.interval[i]   = shape[c.indices[i]]
  for i in range(z) : c.completion[i] = shape[c.indices[i]]
  for i in range(a.spillover+1) : a.completion[i] = shape[i]
  for i in range(b.spillover+1) : b.completion[i] = shape[out[i]]
  if rout[a.spillover] == b.spillover :
    if    a.interval <  b.interval :
      c.shape     [ c.a_spillover ] = b.interval;  
      a.loop, b.loop = True, False
      c.interval  [c.a_spillover] = a.interval
      c.completion[c.a_spillover] = b.interval
      a.completion[a.spillover] = b.interval
      b.completion[b.spillover] = b.interval
    elif  a.interval >  b.interval :
      c.shape[ c.a_spillover ] = a.interval;  
      a.loop, b.loop = False, True 
      c.interval  [c.a_spillover] = b.interval 
      c.completion[c.a_spillover] = a.interval
      #!does this even make sense on combined c level. Depending on read vs write, completion vs interval.
      #rather, distinguish. Ok now I have int/cint everywhere, just need to figure out how to use it.
      a.completion[a.spillover] = a.interval
      b.completion[b.spillover] = a.interval
    else : #==
      c.shape[ c.a_spillover ] = a.interval;  
      a.loop, b.loop = False, False
      c.interval  [c.a_spillover] = a.interval
      c.completion[c.a_spillover] = a.interval
      a.completion[a.spillover] = a.interval
      b.completion[b.spillover] = a.interval
  #!this doesn't set the shape if spillovers are distinct and not contained (=interval) vs distinct and contained (=shape)
  #!which means the block size below is wrong when it uses shape instead of interval
  #fixed:
  else
    if rout[a.spillover] > b.spillover : #a spillover is distinct and not contained in b
      c.shape[c.a_spillover] = a.interval
      a.completion[a.spillover] = a.interval
    if a.spillover < out[b.spillover]  : #b spillover is distinct and not contained in a
      c.shape[c.b_spillover] = b.interval
      b.completion[b.spillover] = b.interval




  #now forward expand the block sizes (sm block)
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
  d.indices = np.extract(notb >= a.spillover, notb)  #$
  d.steps = np.ones(len(d.indices), dtype=int)

  #set steps to spillover if in remaining dims
  if a.spillover in d.indices :
    aspi, = np.where( d.indices == a.spillover )  #I don't want to select first index and tuple every time 
    aspi = list(d.indices).index(a.spillover)  #$
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

 
  #takes indices to gl read order and returns corresponding block (steps reduce the size of the block)
  def blocken( indices, steps ) :
    block = np.zeros(len(indices), dtype=int)
    block[0] = 1
    for i, ii in enumerate(indices[:-1]) :
      block[i+1] = block[i] * int(np.ceil( 1.0 * shape[ii] / steps[i] ))
    return block

  d.block = blocken( d.indices )

  #takes block and splits up number into corresponding index values
  def indexen( s, block ) :
    index = np.zeros(len(block), dtype=int)
    r = s
    for i in reversed(range(len(block))) :
      index[i] = r / block[i] 
      r = r % block[i]
    return index 

  #takes block, indices and values and returns scalar for those. Steps multiply index values. 
  #here the block is supposed to be larger than indices (e.g. full global read block)
  def scalar( block, indices, index, steps ) :
    s = 0
    for i in range(len(indices)) :
      s = s + block[indices[i]] * index[i] * steps[i]
    return s 



  #scalar thread block to scalar global read
  #s to loop over number of global thread blocks d.GN
  def thread_block( s ) :
    index = indexen( s, d.block ) 
    grs = scalar( a.block, d.indices, index, d.steps )

    #checking for partials, return the partial or regular interval
    def partial_check( dim, d )
      i = list(d.indices).index(dim) 
      #if index[i] * (d.steps[i]+1) > shape[dim] : #it's a partial  #!shouldn't it be index[i]+1?
      if (index[i]+1) * d.steps[i] > shape[dim] : #it's a partial  
        return shape[dim]  %  d.steps[i]     
      else 
        return d.steps[i]     

    #!this doesn't look right when a/b same spillover, the max will be d.step, partial completion both in read and write step
    if a.spillover in d.indices :
      completion_interval_a = partial_check( a.spillover, d )
    if out[b.spillover] in d.indices :
      completion_interval_b = partial_check( out[b.spillover], d )

    #recap: we are in global loop at s in range(d.GN)



    #start the sm loop passing down partial loop via completion interval
    def sm_read_outer( completion_interval_a, completion_interval_b, thread ) : 
      #compute the universal outer loop global part of address for global and sm 
      #compute thread specific inner part of address
      #second inner loop adds multiple of read inner block (somewhere need to check vs N and this block size), just store that block size


      #DEAD BLOCK
      #outer loop over b only block. There may be a second loop over a.spillover dim if also in b
      #b and not a
      outy = np.array(out)
      b_only = np.extract(outy[:b.spillover+1] > a.spillover, outy )  #$


      first_outer_completion = list(c.indices).index(a.spillover)  #is also just equal itself, since a has consecutive indices.
      first_outer_interval = first_outer_completion + 1  #note this is wrt c indices order, it's the next index, if any (otherwise gives empty loop, ok)

      #set the completions and intervals
      z = len(c.indices)
      c.interval = np.zeros(z, dtype=int)
      c.completion = np.zeros(z, dtype=int)
      for i in range(z) :
        #!fuck does c shape reflect if a spillover is a regular b dimension. where did I store this info
        c.interval[i]   = c.shape[i] #c shape is already truncated to 2nd loop 
        c.completion[i] = c.shape[i]
       


      prec = np.zeros( c.shape_block[-1] * c.shape[-1] ) #outer size no more than that
      completion_prec_index = -1  #which outer array entry has the inner completion interval, if any (there can only be one per thread block)
      precN = 0
      #recursive implies depth first search, I accumulate the total address by passing down the outer block scalar so
      #an array entry is only written at the leaf loop which is at the first outer completion (loop or not)

      #recurse each dimension from highest to lowest (same as cascading up)
      def loopy( i, so )
        if i >= first_outer_completion :
	  j = 0
	  #completion loop
          while j * c.interval[i]  <  c.completion[i] :
	    
            if (j+1) * c.interval[i]  <=  c.completion[i] :
              interval = c.interval[i]
	    else :
	      interval = c.completion[i] - j*c.interval[i]
	      if first_outer_completion  ==  i :  #I don't like dual meaning of i as c index and read order index
	        completion_prec_index = precN 

	    if i == first_outer_completion :  #write at the outer leaf node, which is always the first outer completion (a's spillover)
              prec[precN] = so + j * c.interval[i] * c.block[i] 
	      precN = precN + 1
	    else :
	      for k in range(interval) :
                loopy( i-1, so + (j * c.interval[i] + k) * c.block[i] )
               
	    j = j + 1
      
      loopy( len(c.indices), 0 )
      #recap: prec now has all the outer scalars for the shared memory, precN is the size, and completion prec index has the leaf completion, only if partial
	      
	    

      #loop over standard size (interval) and extended (completion interval) on each dimension, just to unify the cases.
      #The only required order is thread interval to be the innermost loop.
      #The inner read indices are the first indices in the combined read block, and the first completion interval is always 
      #the last of the inner read indices. So it is sufficient to look at the combined block. Two feasible spins: All intervals in indices order,
      #then all completions in any order. Or each interval and completion, in indices order. 
      #The latter option preserves the natural sequence.

      #Don't forget that coming in I have a thread index and block index, so the loop needs to be over t and b. With precomputed,
      #I can index with t directly for inner scalar. The rest of the rw is an actual loop resulting in c/a sm and global parts that are stored in arrays.
      #Partial can come from any partial completion, whether or not completion index is regular or passed down from global.
    
      #But: global completion can completely change the local sm completion, say from 4,19 to 4,6, cutting short some loop indices and interjecting
      #partial T at different points. So the precomputed array is shortened somewhere (hopefully at the end), and additional T is passed.
      #If you have two seperate completion intervals, that would still put one index loop before the other, no matter which spin.
 
      #Since I unroll the dimensions into more or less into sqrt*sqrt, the actual precalculated arrays are 4*64  for threads and sm for both read and write.
      #If I create arrays for the 3 cases (one or two active completion intervals), the precalc is 64*(2+2+6) = 64*10. That is still amortized, but starts
      #encroaching. As far as memory usage, 64*64 = 4096 sm can be upped by 10x, in particular, 4x should be fine e.g. from 128 threads, for better amortization.
      #So at least in principle, the precalc overhead should be tolerable. Worst case I could cut it down to one extra array instead of 3, by manually
      #shortening the outer one.

      #Ok, so split into a including interval, and everything else. But how to split up the int vs completion int. Well I guess by having two
      #start indices, one for intervals and one for the completions. Then I can still do both options.
      #cases spillover for outer loop read:
      #a or b totally distinct: b distinct: loop over b.interval, not dim
      #a contained in b dim: b dim becomes second loop for a
      #a shared with b: if a in b, second loop for a. Otherwise, nothing on outer loop for b, dim belongs to a inner 
      #b contained in a dim: nothing for b loop, dim belongs to a inner
      #sum.: second loop if a in b shared or not shared. b spill dim loop only if distinct 
      #steps are used for global counts. So rather, interval and completion interval. Lets try again:
      #normal: int and cint = shape. distinct b spill: int and cint = step. second loop: a.int = step, a.cint = b.int (which is shape or not)
      #threads don't spillover into cints, because first ints are taken for spin, then cints. Or rather: a.ints > a.cints > b.ints > b.cints
      #each of these 4 loops gives additive addresses, so need to store 4 arrays
      #whenever the a.cint is smaller than a.int (first or final loop), results in partial thread allocation. For b, just partial loop.
      


    #global thread block loop 
    for thread in range(N) :
      sm_read_outer( completion_interval_a, completion_interval_b, thread )




