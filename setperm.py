import numpy as np
import record
import math
import permutate

# Permutate indices of array A.
# A: device array input in C order, Ap: device array for output 
# index_from/to: einsum notation strings
# A != Ap: no inline permutation allowed/possible. 
# A.shape must be correct shape. Ap.shape is ignored, but be careful when using Ap as input to another permutation
def perm( A, index_from, Ap, index_to ) :

  #I don't have time to think through the trivial case failure (probably assuming inner/outer have at least one difference)
  if index_from == index_to :
    return
  #assert( A.gpu_data != Ap.gpu_data )  #can't use this while in cuda simulator

  itemsize = A.dtype.itemsize

  T = 20  #!debug testing default. Make this proper WARP size.
  rN = len(index_from)
  out = parse( index_from, index_to ) 
  shape = A.shape
  
  shapeOut = np.zeros(rN, dtype=int)
  for i in range(rN) :
    shapeOut[i] = shape[ out[i] ]
  
  N = 1
  for i in range(rN) :
    N = N * shape[i]
 
  if N < T :
    T = N

  #order not relevant for flat array, but original order is for shape to be strided properly in permutation
  #note ravel() will copy data if order is not same as stored order, while reshape will simply stride data differently. 

  #can't access _dummy in cuda simulator
  #assert( A._dummy.flags['C_CONTIGUOUS'] ) 

  fA = A.reshape( N, order='C' )  
  fAp = Ap.reshape( N, order='C' )
 
  overlap = np.zeros(rN, dtype=int)
  #that's the specs.
  
  #determine inner cubes according to threads T
  shapeVirt = np.zeros( rN+2, dtype=int )
  mapv = np.zeros( rN + 1, dtype=int ) #with dummy element. mapv not passed to kernel
  mapv[rN] = rN + 2
  mapr = np.zeros( rN+2, dtype=int ) #reverse map from virt to real with split indicators
  
  i = 1; m = 1
  while 1.0 * T / m > 1.0 and i <= rN :
    inrc = m 
    m = m * shape[rN-i] 
    i = i + 1
  inrN = i - 1 #inner read cube size
  assert( inrN <= rN )
  i = 1; m = 1
  while 1.0 * T / m > 1.0 and i <= rN :
    inwc = m 
    m = m * shape[out[rN-i]] 
    i = i + 1
  inwN = i - 1 #inner write cube size

  assert( inwN <= rN )
  #mode overlap: the inner always owns the overlap  
  #!also redundant with other block that uses it. only need virtual overlap
  for i in range(1,inwN+1) :
    if out[rN-i] >= rN-inrN :
      overlap[ out[rN-i] ] = 1
  #split overlap
  if out[rN-inwN] == rN-inrN :
    if inwc > inrc :  #write split has lower fill/happens first
      s1s = T // inwc #first split dimension is how many times T fills dim-1 inner write cube, ie less than read cube
      s2s = T // (s1s*inrc) #second split dim how many times T fills dim-1 inner read cube w 1st split (at least >= 1X)
      s3s =  math.ceil( 1.0 * shape[rN-inrN] / (s2s*s1s) ) #closing virtual buffer dimension to cover whole index 
      #inner write cube: s1s * inwc. inner read cube: s2s*s1s*inrc (either may be the larger still)
      #the final thread count T' is the max of the two, and threads in diff have to be masked out.
      Tf = max( s1s*inwc, s2s*s1s*inrc )
  
      innerReadN = inrN + 1; innerRead = np.zeros( innerReadN, dtype=int )
      innerWriteN = inwN;    innerWrite = np.zeros( innerWriteN, dtype=int )
      innerRead[0]  = rN-inrN + 1 #second piece
      innerRead[1]  = rN-inrN + 2 #third piece
      innerWrite[0] = rN-inwN + 2 #reflects the split ind -> ind, ind+1, ind+2
      readExtra, writeExtra = 1, 0
  
    else : #inwc <= inrc:
      s1s = T // inrc 
      s2s = T // (s1s*inwc) 
      s3s =  math.ceil( 1.0 * shape[rN-inrN] / (s2s*s1s) ) #closing virtual buffer dimension to cover whole index 
      Tf = max( s1s*inrc, s2s*s1s*inwc )
  
      innerReadN = inrN;      innerRead = np.zeros( innerReadN, dtype=int )
      innerWriteN = inwN + 1; innerWrite = np.zeros( innerWriteN, dtype=int )
      innerWrite[0] = rN-inrN + 1
      innerWrite[1] = rN-inrN + 2
      innerRead[0]  = rN-inrN + 2  
      readExtra, writeExtra = 0, 1
  
    for i in range(rN-inrN) : 
      mapv[i] = i
      mapr[i] = i
      shapeVirt[i] = shape[i]
    mapv[rN-inrN] = rN-inrN
    mapr[rN-inrN] = -2
    mapr[rN-inrN +1] = -1
    mapr[rN-inrN +2] = rN-inrN
    shapeVirt[rN-inrN] = s3s
    shapeVirt[rN-inrN +1] = s2s
    shapeVirt[rN-inrN +2] = s1s
    for i in range(rN-inrN+1, rN) :
      mapv[i] = i+2
      mapr[i+2] = i
      shapeVirt[i+2] = shape[i]
  
    for i in range(1,inrN) : #remaining non-split indices
      innerRead[ readExtra + i ] = (rN-inrN)+i + 2 #+2 for virtual index shift after triple split
    for i in range(1,inwN) :
      if  out[ (rN-inwN)+i ]  <  rN-inrN : #before split means no shift
        innerWrite[ writeExtra + i ] = out[ (rN-inwN)+i ]
      else :
        innerWrite[ writeExtra + i ] = out[ (rN-inwN)+i ] + 2
  
  
  else : #separate splits   
    srs = T // inrc 
    scr = math.ceil( 1.0 * shape[rN-inrN] / srs )
    sws = T // inwc 
    scw = math.ceil( 1.0 * shape[out[rN-inwN]] / sws )
    Tf = max( srs*inrc, sws*inwc )
  
    #does one fully contain the other split?
    #!this block is redundant. Below can just populate the inner arrays and set N after fact
    if overlap[ out[rN-inwN] ] : #inner read contains write split
      innerReadN = inrN + 1; 
    else :
      innerReadN = inrN; 
    innerRead = np.zeros( innerReadN, dtype=int )
    if overlap[ rN-inrN ] : #inner write contains read split
      innerWriteN = inwN + 1; 
    else :
      innerWriteN = inwN; 
    innerWrite = np.zeros( innerWriteN, dtype=int )
  
    if  rN-inrN <  out[rN-inwN] :  #which split comes first
      s1, s2 = rN-inrN, out[rN-inwN]
      s1o, s1c,  s2o, s2c = scr, srs,  scw, sws
    else :
      s2, s1 = rN-inrN, out[rN-inwN]
      s2o, s2c,  s1o, s1c = scr, srs,  scw, sws  #split open close
  
    for i in range( s1 ) : #to first split
      mapv[i] = i
      mapr[i] = i
      shapeVirt[i] = shape[i]
    mapv[s1] = s1
    mapr[s1] = -1 
    mapr[s1+1] = s1
    shapeVirt[s1] = s1o 
    shapeVirt[s1+1] = s1c 
    for i in range( s1+1, s2 ) : #to second split
      mapv[i] = i+1
      mapr[i+1] = i
      shapeVirt[i+1] = shape[i]
    mapv[s2] = s2+1
    mapr[s2+1] = -1 
    mapr[s2+2] = s2 
    shapeVirt[s2+1] = s2o 
    shapeVirt[s2+2] = s2c 
  
    for i in range(s2+1, rN) :
      mapv[i] = i+2
      mapr[i+2] = i
      shapeVirt[i+2] = shape[i]
  
    #first index is split, need right half. Others may include other split.
    innerRead[0] = mapv[ rN-inrN ] +1 #second half
    j = 1
    for i in range(1,inrN) : #remaining
      vi = mapv[ (rN-inrN)+i ]
      while vi != mapv[ (rN-inrN)+i +1 ] :
        innerRead[j] = vi
        j = j + 1
        vi = vi + 1
  
    innerWrite[0] = mapv[ out[rN-inwN] ] +1 #second half
    j = 1
    for i in range(1,inwN) :
      vi = mapv[ out[ (rN-inwN)+i ] ]
      while vi != mapv[ out[ (rN-inwN)+i ] +1 ] : 
        innerWrite[j] = vi
        j = j + 1
        vi = vi + 1
  
    #!j has the right length could just set inner N here.
  
  #virtual overlap of inner, and union
  overlapVirt = np.zeros( rN+2, dtype=int )
  unionVirt   = np.zeros( rN+2, dtype=int )
  for i in range( innerWriteN ) :
    if innerWrite[i] >= rN+2 - innerReadN :
      overlapVirt[ innerWrite[i] ] = 1
    unionVirt[ innerWrite[i] ] = 1
  for i in range( innerReadN ) :
    unionVirt[ innerRead[i] ] = 1
  #block: virtual modes not contained in inners
  block = np.zeros( rN+2, dtype=int )
  j = 0
  for i in range( rN+2 ) :
    if not unionVirt[i] :
      block[j] = i
      j = j + 1
  blockN = j
  
  #outers are the complements: outer read is inner write - inner read, outer write is inner read - inner write
  #also outer order of modes doesn't matter. But just in case there is dependency on first mode being a split
  outerRead = np.zeros( rN+2, dtype=int )
  j = 0
  for i in range( innerWriteN ) :
    if not overlapVirt[ innerWrite[i] ] :
      outerRead[j] = innerWrite[i] 
      j = j + 1
  outerReadN = j
  outerWrite = np.zeros( rN+2, dtype=int )
  j = 0
  for i in range( innerReadN ) :
    if not overlapVirt[ innerRead[i] ] :
      outerWrite[j] = innerRead[i] 
      j = j + 1
  outerWriteN = j
  
  #output virtual: scan out along with mapv
  outVirt = np.zeros( rN+2, dtype=int )
  j = 0
  for i in range( rN ) :
    k = mapv[ out[i] ]
    while k != mapv[ out[i] + 1] :
      outVirt[j] = k
      k = k + 1
      j = j + 1
  assert( j == rN+2 )
  
  p = record.parg( innerReadN=innerReadN, outerReadN=outerReadN, innerWriteN=innerWriteN, outerWriteN=outerWriteN
                 , blockN=blockN,  virtualN=rN+2, realN=rN )
  p['shape'] = shape 
  p['shapeVirt'] = shapeVirt 
  p['mapVirt'] = mapr 
  ps = p['shape']
  si = 1
  
  p['strides'][rN-1] = si
  for i in range( rN-2,-1,-1 ) :
    p['strides'][i] = p['shape'][i+1] * p['strides'][i+1] 
  p['T'] = Tf  
  p['out'] = out 
  p['outVirt'] = outVirt
  #inner/outer virt are the real indices that get split
  outerYet = False 
  for i in range( rN+2 ) :
    if mapr[i] < 0 :
      if not outerYet :
        outerVirt, outerYet = mapr[ i-mapr[i] ], True
      else :
        assert( mapr[i] == -1 )
        innerVirt = mapr[i+1]
  p['innerVirt'] = innerVirt #wrt real 
  p['outerVirt'] = outerVirt
  p['innerRead'] = innerRead[:innerReadN]  
  p['outerRead'] = outerRead[:outerReadN]  
  p['innerWrite'] = innerWrite[:innerWriteN]
  p['outerWrite'] = outerWrite[:outerWriteN]
  p['block'] = block[:blockN]  #order arbitrary
  blocks = 1
  for i in range( blockN ) :
    blocks = blocks * shapeVirt[ block[i] ]
  
  from pdb import set_trace; set_trace()
  permutate.perm[ blocks, Tf, 0, Tf*Tf * itemsize ]( fA, fAp, p)  #Tf*Tf is upper bound when no overlap

  #debug
  Ap_ = Ap.copy_to_host()
  A_ = A.copy_to_host()


#assumes no repeated indices and proper permutation
def parse( index_from, index_to ) :
  out = np.zeros( len(index_from), dtype=int )
  map_index = {}
  for i in range(len(index_from)) :
    map_index[ index_from[i] ] = i
  for i in range(len(index_to)) :
    out[i] = map_index[ index_to[i] ]  
  return out

