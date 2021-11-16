#ifndef _MATRIX_H_
#define _MATRIX_H_

/* Matrix Structure declaration */
#define MATRIX_SIZE 1024
typedef struct matrix_s {
    unsigned int width;
    unsigned int height;
    float* elements;
} matrix_t;

/* Define the thread block size on the GPU */
#define TILE_SIZE 32

/* Define the BLOCK_SIZE for the blocked multiplication on the CPU */
#define BLOCK_SIZE 32

#endif /* _MATRIX_H_ */

