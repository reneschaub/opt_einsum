import numpy as np
from numba import cuda, float32, int32
import numba

@cuda.jit
def perm(p):
    t = cuda.threadIdx.x
    b = cuda.blockIdx.x

    S = cuda.shared.array (shape=0, dtype=float32)

    vix = cuda.local.array(10, dtype=int32)

    for i in range(5-1,-1,-1) :
      dx = 5
      dxN = p['shapeVirt'][dx]

    if t < 14 :
    #if True :
      for i in range(5-1,-1,-1) :
        dx = 5
        vix[dx] =  5 % dxN
      if b == 2 and t == 1 :
        p['pad'][0] = 112211

    cuda.syncthreads()

    if t == 17 and b == 2 :
      p['pad'][1] = p['pad'][0]


def setperm( ) :

  a = np.dtype(
  [ ('shapeVirt', np.int32, (4+2,))
  , ('pad', np.int32, (2,)) 
  ])

  p_ = np.zeros(1, dtype=a) 
  p = p_[0]


  itemsize, Tf = 4, 20
  p['shapeVirt'][:6] =  [1, 1, 1, 1, 1, 1]
  blocks = 3
  perm[ blocks, Tf, 0, Tf*Tf * itemsize ]( p )

  print( p['pad'] )


setperm( )

print('success')


