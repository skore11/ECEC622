#ifndef _JACOBI_SOLVER_H_
#define _JACOBI_SOLVER_H_

#define THRESHOLD 1e-5      /* Threshold for convergence */
#define MIN_NUMBER 2        /* Min number in the A and b matrices */
#define MAX_NUMBER 10       /* Max number in the A and b matrices */
#include <semaphore.h>

/* Matrix structure declaration */
typedef struct matrix_s {
    unsigned int num_columns;   /* Matrix width */
    unsigned int num_rows;      /* Matrix height */ 
    float *elements;
}  matrix_t;

typedef struct barrier_s {
    sem_t counter_sem;          /* Protects access to the counter */
    sem_t barrier_sem;          /* Signals that barrier is safe to cross */
    int counter;                /* The value itself */
} barrier_t;

barrier_t barrier;

/* Function prototypes */
matrix_t allocate_matrix(int, int, int);
extern void compute_gold(const matrix_t, matrix_t, const matrix_t, int);
extern void display_jacobi_solution(const matrix_t, const matrix_t, const matrix_t);
int check_if_diagonal_dominant(const matrix_t);
matrix_t create_diagonally_dominant_matrix(int, int);
extern void compute_using_pthreads_v1(const matrix_t, matrix_t, matrix_t, const matrix_t, int max_iter, int num_threads);
extern void compute_using_pthreads_v2(const matrix_t, matrix_t, matrix_t, const matrix_t, int max_iter, int num_threads);
void print_matrix(const matrix_t);
float get_random_number(int, int);
void barrier_sync(barrier_t *, int, int);

#endif /* _JACOBI_SOLVER_H_ */

