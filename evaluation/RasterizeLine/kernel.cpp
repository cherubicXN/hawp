#include "draw.hpp"
#include "math.h"
void _draw(const float* lines,
          double* map,
          int height, int width,
          int nlines)
{
    for(int k = 0; k < nlines; ++k){
        float x1 = lines[4*k];
        float y1 = lines[4*k+1];
        float x2 = lines[4*k+2];
        float y2 = lines[4*k+3];

        float vn = ceil(sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2)));

        float dx = (x2-x1)/(vn-1.0);
        float dy = (y2-y1)/(vn-1.0);

        for(int j = 0; j < (int) vn; ++j){
            int xx = round(x1+(float)j*dx);
            int yy = round(y1+(float)j*dy);
            if (xx>=0 && xx<=width-1 && yy>=0 && yy<=height-1){
                int index = yy*width + xx;
                map[index] = 1.0;
            }
        }
    }
}