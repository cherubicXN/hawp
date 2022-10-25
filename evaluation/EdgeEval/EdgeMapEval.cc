#include "Matrix.hh"
#include "match.hh"
#include <string.h>
#include <stdio.h>
#include <math.h>
#include "correspondPixels.h"


void correspondPixels(double* bmap1, 
                      double* bmap2, 
                      double* match1,
                      double* match2,
                      double& cost,  
                      double maxDist,
                      double outlierCost,
                      int height,
                      int width)
{
    const int rows = width;
    const int cols = height;

    const double idiag = sqrt(float(rows*rows+cols*cols));
    const double oc = outlierCost*maxDist*idiag;

    Matrix m1,m2;
    cost = matchEdgeMaps(
        Matrix(rows,cols,bmap1), Matrix(rows,cols,bmap2),
        maxDist*idiag, oc, m1, m2);

    memcpy(match1, m1.data(), m1.numel()*sizeof(double));
    memcpy(match2, m2.data(), m2.numel()*sizeof(double));
}                      