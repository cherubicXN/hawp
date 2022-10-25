import numpy as np
cimport numpy as np

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "correspondPixels.h":
    void correspondPixels(double* bmap1,
                      double* bmap2,
                      double* match1,
                      double* match2,
                      double& cost,  
                      double maxDist,
                      double outlierCost, 
                      int height, 
                      int width)

def correspond(np.ndarray[double,ndim=2] bmap1,
               np.ndarray[double,ndim=2] bmap2,
               double maxDist = 0.001,
               double outlierCost = 100):
    
    assert bmap1.shape[0] == bmap2.shape[0] and bmap1.shape[1] == bmap2.shape[1]

    cdef int height = bmap1.shape[0]
    cdef int width = bmap1.shape[1]

    cdef np.ndarray[np.double_t, ndim=2] \
        match1 = np.zeros((height,width), dtype=np.double)

    cdef np.ndarray[np.double_t, ndim=2] \
        match2 = np.zeros((height,width), dtype=np.double)
    cdef double cost = 0
    correspondPixels(<double*> bmap1.data,
                     <double*> bmap2.data,
                     <double*> match1.data,
                     <double*> match2.data,
                     cost,
                     maxDist,
                     outlierCost,
                     height,
                     width)

    return match1, match2, cost


