void correspondPixels(double* bmap1, /*input*/
                      double* bmap2, /*input*/
                      double* match1, /*output*/
                      double* match2, /*output*/
                      double& cost,   /*output*/
                      double maxDist, /*parameters*/
                      double outlierCost, /*parameters*/
                      int height, /*image size*/
                      int width /*image size*/);