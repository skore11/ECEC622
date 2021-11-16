#ifndef _MATRIXMUL_H_
#define _MATRIXMUL_H_

/* Thread block size */
#define MATRIX_SIZE 32

/* Matrix dimensions chosen as multiples of thread block size for simplicity */
#define WM MATRIX_SIZE      /* Matrix M width */
#define HM MATRIX_SIZE      /* Matrix M height */
#define WN MATRIX_SIZE      /* Matrix N width */
#define HN WM               /* Matrix N height */
#define WP WN               /* Matrix P width */
#define HP HM               /* Matrix P height */

/* Matrix Structure declaration */
typedef struct matrix_s {
	unsigned int width;     /* Width of matrix */
    unsigned int height;    /* Height of matrix */
    float *elements;        /* Pointer to the first element of matrix */
} matrix_t;

#endif /* _MATRIXMUL_H_ */

