import numpy as np
cimport numpy as np

assert  sizeof(int) == sizeof(np.int32_t)

cdef extern from "draw.hpp":
    void _draw(const float* lines, double* map, int height, int width, int nlines);


def drawfn(np.ndarray[float, ndim=2] lines,
         int height, int width):

    cdef np.ndarray[np.double_t, ndim=2] linemap = np.zeros((height,width),
                                                            dtype=np.double)
    cdef int nlines = lines.shape[0]
    _draw(<const float*> lines.data, <double*> linemap.data,
          height, width, nlines)


    return linemap