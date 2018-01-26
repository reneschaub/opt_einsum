import numpy as np

from numba import cuda #, float32 #, int32
import setperm


N = 4*5*6*7
#a = np.arange(N, dtype='float32').reshape(4,5,6,7,  order='C')
a = np.arange(N, dtype='float32').reshape(4,5,6,7,  order='C')
d_a = cuda.to_device( a )

view_right = d_a 

#default both
#import pdb; pdb.set_trace()
hold = cuda.device_array_like( view_right ) #!cache this for repeated calls
#setperm.perm( view_right, 'ijkl', hold, 'klij' )
setperm.perm( view_right, 'ijkl', hold, 'klij' )

print('success')
