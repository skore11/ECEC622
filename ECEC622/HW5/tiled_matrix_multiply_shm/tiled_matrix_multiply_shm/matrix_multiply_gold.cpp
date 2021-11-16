/* Reference code for P = M x N
 * Modified by Naga Kandasamy, 2/16/2017
 */

#include <stdlib.h>
#include "matrix.h"

extern "C" void compute_gold_naive(matrix_t, matrix_t, matrix_t);
extern "C" void compute_gold_blocked(matrix_t, matrix_t, matrix_t);

/*  Note: The below implementation is a naive way to code up matrix multiplication 
 * since the code exhibits very bad cache locality along the column elements of 
 * the N matrix.  
 * */
void compute_gold_naive(matrix_t M, matrix_t N, matrix_t P)
{
    int i, j, k;
    float sum, m, n;

	for (i = 0; i < M.height; ++i)
		for (j = 0; j < N.width; ++j) {
			sum = 0;
			for (k = 0; k < M.width; ++k) {
				m = M.elements[i * M.width + k];
				n = N.elements[k * N.width + j];
				sum += m * n;
			}
			
            P.elements[i * N.width + j] = (float)sum;
		}
}

/* Implementation using blocked or tiled multiplication, 
 * see: Hennessy and Patterson, Computer Architecture, Morgan Kaufmann, 
 * 5th Edition, 2011, page 90.
 *
 * This implementation improves the temporal locality by maximizing accesses 
 * to the data loaded into the cache before the data are replaced. 
 * The innermost loop pair (jk) multiplies a 1 X BLOCK_SIZE sliver of M 
 * by a BLOCK_SIZE X BLOCK_SIZE block of N and accumulates into 
 * 1 X BLOCK_SIZE sliver of P. The loop over i steps through the row slivers 
 * of M & P, using the same block.
 * */
void compute_gold_blocked(matrix_t M, matrix_t N, matrix_t P)
{
    int i, j, k;
    int jj, kk;
    double sum;

    for (jj = 0; jj < M.width; jj += BLOCK_SIZE)
        for (kk = 0; kk < M.height; kk += BLOCK_SIZE)
            for (i = 0; i < M.height; i++) 
                for (j = jj; j < (jj + BLOCK_SIZE); j++) {
                    sum = 0.0;
                    for (k = kk; k < (kk + BLOCK_SIZE); k++) 
                        sum += M.elements[i * M.width + k] * N.elements[k * N.width + j];
        
                    P.elements[i * N.width + j] += sum;             
                }
}

