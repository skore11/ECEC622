/* Reference solution for Y = A.X 
 * Author: Naga Kandasamy
 * Date modified: May 4, 2019
 * */
 
#include <stdlib.h>
#include "vec_mat_mult.h"

extern "C" void compute_gold(matrix_t, matrix_t, matrix_t);

void compute_gold(matrix_t A, matrix_t X, matrix_t Y)
{
    int i, j;
    float sum;
    double a, x;

    for (i = 0; i < A.num_rows; i++) {
		sum = 0.0;
		for (j = 0; j < X.num_rows; j++) {
			a = A.elements[i * A.num_columns + j]; /* Pick A[i, j] */
			x = X.elements[j]; /* Pick X[j] */
			sum += a * x;
		}

		Y.elements[i] = sum; /* Store result in Y. */
	}	
    
    return;
}
