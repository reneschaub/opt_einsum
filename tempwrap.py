
import numpy as np
import record
import math
from numba import cuda, float32, int32
import permutate
import setperm

N = 4*5*6*7 
#N = 4*4
#A = np.arange(N, dtype='float32').reshape(4,4) #.reshape(10,10,10,10)
#C = np.ones(N, dtype='float32').reshape(4,4)  #.reshape(10,10,10,10)
#A = np.arange(N, dtype='float32').reshape(4,5,6,7) #.reshape(10,10,10,10)
A = np.arange(N, dtype='float32').reshape(6,4,7,5) #.reshape(10,10,10,10)
#C = np.arange(N, dtype='float32').reshape(4,5,6,7) #.reshape(10,10,10,10)
C = np.zeros(N, dtype='float32').reshape(6,7,4,5)  #.reshape(10,10,10,10)
#C = np.zeros(N, dtype='float32').reshape(6,7,4,5)  #.reshape(10,10,10,10)

Ad = cuda.to_device(A)
Cd = cuda.to_device(C)




#setperm.perm( Ad, 'ij', Cd, 'ji', A.dtype.itemsize ) #, (4,4) )
#setperm.perm( Ad, 'ij', Cd, 'ji' ) 
#setperm.perm( Ad, 'ijkl', Cd, 'klij' ) 
setperm.perm( Ad, 'minj', Cd, 'mnij' ) 
from pdb import set_trace; set_trace()
Co = Cd.copy_to_host()

