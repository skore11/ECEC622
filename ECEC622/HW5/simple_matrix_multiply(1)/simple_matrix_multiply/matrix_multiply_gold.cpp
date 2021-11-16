/* Reference implementation of matrix multiplication */
#include <stdlib.h>

extern "C" void compute_gold(float *, float *, float *, int, int, int);

void compute_gold (float *M, float *N, float *P, int hM, int wM, int wN)
{
    int i, j, k;
    double a, b, sum;
	for (i = 0; i < hM; ++i)
		for (j = 0; j < wN; ++j) {
			sum = 0.0;
			for (k = 0; k < wM; ++k) {
				a = M[i * wM + k];
				b = N[k * wN + j];
				sum += a * b;
			}

			P[i * wN + j] = (float)sum;
		}
}
