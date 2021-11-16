/* Reference code for scan. */

#include <stdio.h>
#include <math.h>
#include <float.h>

extern "C" void compute_gold(float *, float *, int);

/* Returns scan of idata in the out array */
void compute_gold(float *out, float *in, int len) 
{
    out[0] = 0;
    double sum = 0;
    int i;
  
    for (i = 1; i < len; ++i) {
        sum += in[i - 1];
        out[i] = in[i - 1] + out[i - 1];
    }
    
    /* Here it should be okay to use != because we have integer values
     * in a range where float can be exactly represented. */
    if (sum != out[i-1])
        printf("Warning: exceeding single-precision accuracy. Scan will be inaccurate\n");
}


