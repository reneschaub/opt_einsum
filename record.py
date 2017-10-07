import numpy as np
import numba

x = np.zeros((2,2), dtype=[('virtIn', np.int32), ('virtOut', np.int32), ('out', np.int32, (10,))])
x = np.zeros((2,2), dtype=[('virtIn', np.int32), ('virtOut', np.int32), ('out', np.int32, (10,))])

innerReadN, outerReadN, innerWriteN, outerWriteN, blockN = 2, 2, 2, 2, 1
virtualN, realN = 9, 7
maxM = 10  #!max tensor modes


def  parg( 
  innerReadN
, outerReadN
, innerWriteN
, outerWriteN
, blockN
, virtualN
, realN 
):

  a = np.dtype(
  #real mode of virtual modes (last modes in read sets) 
  [ ('innerVirt', np.int32)
  , ('outerVirt', np.int32)
  , ('out', np.int32, (realN,))
  , ('strides', np.int32, (realN,))
  , ('shape', np.int32, (realN,))
  , ('shapeVirt', np.int32, (realN+2,))
  , ('pad', np.int32, (realN+2,))
  , ('padreal', np.float32, (realN+2,))
  , ('mapVirt', np.int32, (realN+2,)) #map to real index
  , ('outVirt', np.int32, (realN+2,)) #output permutation wrt virtual
  #read modes wrt input order
  , ('innerRead', np.int32, (innerReadN,)), ('innerReadN', np.int32 )
  , ('outerRead', np.int32, (outerReadN,)), ('outerReadN', np.int32 )
  #write modes wrt output permutation
  , ('innerWrite', np.int32, (innerWriteN,)), ('innerWriteN', np.int32 )
  , ('outerWrite', np.int32, (outerWriteN,)), ('outerWriteN', np.int32 )
  #block modes wrt input order
  , ('block', np.int32, (blockN,)), ('blockN', np.int32 )
  #the number of elements in each virtual read mode
  , ('virtualN', np.int32 )
  #the number of elements in each real read mode
  , ('realN', np.int32 )
  , ('T', np.int32)
  ])

  p_ = np.zeros(1, dtype=a)  #why does np not have single instance creation
  p = p_[0]

  p['realN'] = realN
  p['virtualN'] = virtualN
  p['blockN'] = blockN
  p['innerReadN'] = innerReadN
  p['outerReadN'] = outerReadN
  p['innerWriteN'] = innerWriteN
  p['outerWriteN'] = outerWriteN
 
  return p

a = parg( innerReadN=2, outerReadN=2, innerWriteN=2, outerWriteN=2, blockN=1,  virtualN=9, realN=7 )


