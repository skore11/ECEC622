/* Reference code for stream compaction */

#include <stdio.h>
#include <math.h>
#include <float.h>

extern "C" void compute_gold(int *, int *, int *);

/* Compact the input stream in the out array. */
void compute_gold(int *out, int *in, int *len) 
{
    int n = 0; /* Number of elements in the compacted stream */
    int i;

	for (i = 0; i < (*len); i++) {
		if (in[i] > 0) 
			out[n++] = in[i];
	}

	*len = n; /* Update number of elements in output stream */
}


