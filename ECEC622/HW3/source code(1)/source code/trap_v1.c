/* A program that illustrates the use of OpenMp to compute the area under a curve f(x) using the trapezoidal rule. 
 * 
 * Given a function y = f(x), and a < b, we can estimate the area between the graph of f(x) (within the vertical lines x = a and 
 * x = b) and the x-axis by dividing the interval [a, b] into n subintervals and approximating the area over each subinterval by the 
 * area of a trapezoid. 
 *
 * If each subinterval has the same length and if we define h = (b - a)/n, x_i = a + ih, i = 0, 1, ..., n, then the approximation 
 * becomes: h[f(x_0)/2 + f(x-1) + f(x_2) + ... + f(x_{n-1}) + f(x_n/2)
 *
 * This code assumes that f(x) = sqrt((1 + x^2)/(1 + x^4))
 *
 * Compile as follows: gcc -o trap_v1 trap_v1.c -fopenmp -std=c99 -Wall -O3 -lm
 *
 * Author: Naga Kandasamy
 * Date created: April 15, 2011
 * Date modified: April 26, 2020
 * 
 *  */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

/* Function prototypes */
void trap(double, double, int, double *);

int main(int argc, char **argv)
{
    if (argc < 5){
        fprintf(stderr, "Usage: %s upper-limit lower-limit num-subintervals thread-count\n", argv[0]);
        fprintf(stderr, "upper-limit, lower-limit: upper and lower bounds of integration\n");
        fprintf(stderr, "num-subintervals: number of sub-intervals to partition area under the curve\n");
        fprintf(stderr, "thread-count: number of threads to create\n");
        exit(EXIT_FAILURE);
  }
  
    double a = atof(argv[1]);
    double b = atof(argv[2]);
    int n = atoi(argv[3]);
    int thread_count = atoi(argv[4]);	                        
    
    double approximate_area;
  
    if ((n % thread_count) != 0) {
        fprintf(stderr, "Number of sub-intervals must be evenly divisible by the thread_count\n");
        exit(EXIT_FAILURE);
    }

#pragma omp parallel num_threads(thread_count)                  /* OpenMP block */
    {
        trap(a, b, n, &approximate_area);
    }

    fprintf(stderr, "With %d trapeziods, the estimate for the integral between [%f, %f] is %f\n", n, a, b, approximate_area);

    exit(EXIT_SUCCESS);
}

/* Function to integrate */
double f(double x)                                                    
{
    return sqrt((1 + x*x)/(1 + x*x*x*x));
}

void trap(double a, double b, int n, double *approximate_area)      /* Function executed by each thread in the OpenMP block */
{
    int tid = omp_get_thread_num();	                        
    int thread_count = omp_get_num_threads();

    double h = (b - a) / (float)n;	                            /* Length of the subinterval */

    /* We assume that the number of subintervals is evenly divisible by thread_count */
    int chunk_size = n / thread_count;	
    double start_offset = a + h * chunk_size * tid;
    double end_offset = start_offset + h * chunk_size;

    /* Approximate the area under the curve in the interval [start_offset, end_offset] */
    int i;
    double sum = 0.0;
    double x;
    sum = (f(start_offset) + f(end_offset)) / 2.0;
    for (i = 1; i <= (chunk_size - 1); i++) {
        x = start_offset + i * h;
        sum += f(x);
    }
    sum = sum * h;

#pragma omp critical
    *approximate_area += sum;
}
