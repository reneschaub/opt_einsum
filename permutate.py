import numpy as np

from numba import cuda, float32, int32
import record


@cuda.jit
def perm(Ad, Cd, p):

    S = cuda.shared.array (shape=0, dtype=float32) 
 
    #!probably could specify realN in jit type string. Also should use shared mem or absorb into scalar 
    vix = cuda.local.array(10, dtype=int32)  #vix is always wrt original virtual order, even in the write loop 
    rix = cuda.local.array(10, dtype=int32)  #real index of course is also always wrt original order 
    ostv = cuda.local.array(10, dtype=int32)  
    rouv = cuda.local.array(10, dtype=int32)  
    sms  = cuda.local.array(10, dtype=int32)  
    ous  = cuda.local.array(10, dtype=int32)  

    #derived 
    vN = p['virtualN']
    rN = p['realN']
    inr, inw, our, ouw = p['innerReadN'], p['innerWriteN'], p['outerReadN'], p['outerWriteN']
    si = 1  #!itemsize not available, but jit figures out item stride so nothing here
    ourbN = 1 #size of outer read block
    for i in range( our ) :  #!move to input record, waste
      ourbN = ourbN * p['shapeVirt'][ p['outerRead'][i] ]
    ouwbN = 1 #size of outer write block
    for i in range( ouw ) :  #!move to input record, waste
      ouwbN = ouwbN * p['shapeVirt'][ p['outerWrite'][i] ]
    inrbN = 1 
    for i in range( inr ) :  
      inrbN = inrbN * p['shapeVirt'][ p['innerRead'][i] ]
    inwbN = 1 
    for i in range( inw ) :  
      inwbN = inwbN * p['shapeVirt'][ p['innerWrite'][i] ]

    p['pad'][0] = inrbN
    p['pad'][1] = inwbN
    p['pad'][2] = p['T'] 


    #virtual output strides, and reverse virtual out (which out position is ith virt mode)
    #!not used
    st = si 
    for i in range( vN-1,-1,-1 ) :
      ostv[i] = st 
      st =  st * p['shapeVirt'][ p['outVirt'][i] ]
      rouv[ p['outVirt'][i] ] = i

    #shared mem strides of shape, which can be in any order of the cube indices, so why not original order
    st = si
    for i in range(inr-1,-1,-1) :
      dx = p['innerRead'][i] 
      sms[dx] = st
      dxN = p['shapeVirt'][dx] 
      st = st * dxN 
    for i in range(our-1,-1,-1) :
      dx = p['outerRead'][i] 
      sms[dx] = st
      dxN = p['shapeVirt'][dx] 
      st = st * dxN 

    #output strides. Note strides wrt to original order indices, but computed according to out order
    st = si
    for i in range(rN-1,-1,-1) :
      dx = p['out'][i] 
      ous[dx] = st
      dxN = p['shape'][dx] 
      st = st * dxN 

    t = cuda.threadIdx.x
    b = cuda.blockIdx.x

    #block indices: must be set for each thread (once)
    blN = p['blockN'] 
    rb = b
    for i in range(blN-1,-1,-1) :
      dx = p['block'][i]  #will include left parts of split dims if not overlapping
      dxN = p['shapeVirt'][dx] 
      vix[dx] =  rb % dxN
      rb =  rb // dxN  

    if t < inrbN : #since the r/w inner block sizes may differ, some threads will skip work in smaller one
      rt = t
      #inner read cube indices
      for i in range(inr-1,-1,-1) :
        dx = p['innerRead'][i]  #this should be just the last indices really (inner cube)
        dxN = p['shapeVirt'][dx] 
        vix[dx] =  rt % dxN
        rt =  rt // dxN  

  
      #only split indices can exceed bounds. 
  
      #outer read loop
      #ouN = p['T']  BUG: only when no overlap
      ourN = p['outerReadN']
      for j in range(ourbN) :
        rj = j 
        for i in range(ourN-1,-1,-1) :
          dx = p['outerRead'][i]  #this should be just the last indices really (inner cube)
          dxN = p['shapeVirt'][dx] 
          vix[dx] = rj % dxN
          rj = rj // dxN
       
        #all indixes are determined now.
        #real indices to read data with proper strides.
        #only left parts of split indices are different, all others map 1-1
        rix[:] = 0
        for i in range(vN) :  
        #!only virtual indices corresponding to outer loop need be redone: redundant
        #(which can contain at most the inner left virt as a special case)
          rx =  p['mapVirt'][i]
          if rx == -1 : 
            riv = p['mapVirt'][i+1]  #the real index 
            rix[ riv ] = rix[ riv ] +  p['shapeVirt'][i+1] * vix[i] 
          elif rx == -2 : 
            riv = p['mapVirt'][i+2]  #the real index 
            rix[ riv ] = rix[ riv ] +  p['shapeVirt'][i+1] * p['shapeVirt'][i+2] * vix[i] 
          else :
            rix[ rx ] = rix[ rx ] + vix[i] 
  
        #proceed if the two split indices are valid 
        if  rix[ p['innerVirt'] ]  <  p['shape'][ p['innerVirt'] ] \
        and rix[ p['outerVirt'] ]  <  p['shape'][ p['outerVirt'] ] :
          #real flat read index
          r = 0
          for i in range( rN ) :  #!overkill, calculate fixed inner and block ind only once 
            r = r + rix[i] * p['strides'][i]
  
          #destination shared mem output flat virtual index. The full cube is (outerRead,innerRead)
          v = 0
          for i in range( inr ) :
            dx = p['innerRead'][i]
            v = v + vix[dx] * sms[dx]
          for i in range( our ) :
            dx = p['outerRead'][i]
            v = v + vix[dx] * sms[dx]
 
          S[v] = Ad[r]
 
    cuda.syncthreads()  #different threads reading shared memory: essential because of skipped read threads

    if t < inwbN : 
      #remap thread indices to write permuted shared memory to global, using the write cubes 
      rt = t
      #inner write cube indices #!I think inner cube indices can be ignored, they just get flattened again anyhow

      for i in range(inw-1,-1,-1) :
        dx = p['innerWrite'][i]  
        dxN = p['shapeVirt'][dx] 
        vix[dx] =  rt % dxN
        rt =  rt // dxN  
  
      #outer write loop
  
      ouwN = p['outerWriteN']
      for j in range(ouwbN) :
        rj = j
        for i in range(ouwN-1,-1,-1) :
          dx = p['outerWrite'][i]  
          dxN = p['shapeVirt'][dx]
          vix[dx] = rj % dxN
          rj = rj // dxN
        #all indixes are determined now.
        #real indices to write data with proper strides.
        #only left parts of split indices are different, all others map 1-1
        rix[:] = 0
        for i in range(vN) :
        #!only virtual indices corresponding to outer loop need be redone 
        #(which can contain at most the inner left virt as a special case)
          rx =  p['mapVirt'][i]
          if rx == -1 : 
            riv = p['mapVirt'][i+1]  #the real index 
            rix[ riv ] = rix[ riv ] +  p['shapeVirt'][i+1] * vix[i] 
          elif rx == -2 : 
            riv = p['mapVirt'][i+2]  #the real index 
            rix[ riv ] = rix[ riv ] +  p['shapeVirt'][i+1] * p['shapeVirt'][i+2] * vix[i] 
          else :
            rix[ rx ] = rix[ rx ] + vix[i]
  
        #proceed if the two split indices are valid            
        if  rix[ p['innerVirt'] ]  <  p['shape'][ p['innerVirt'] ] \
        and rix[ p['outerVirt'] ]  <  p['shape'][ p['outerVirt'] ] :
          #real flat write index
          r = 0
          for i in range( rN ) :  #!overkill, calculate fixed inner and block ind only once 
            r = r + rix[i] * ous[i]
          #from shared mem output flat virtual index
          #v = 0
          #for i in range( vN ) :
          #  v = v + vix[i] * ostv[ rouv[i] ]
  
          v = 0
          for i in range( inr ) :  #the cube stays same, just the vix change, and given strides order doesn't matter
            dx = p['innerRead'][i]
            v = v + vix[dx] * sms[dx]
          for i in range( our ) :
            dx = p['outerRead'][i]
            v = v + vix[dx] * sms[dx]

          Cd[r] = S[v]
   

