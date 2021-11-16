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
 * Compile as follows: gcc -o trap_v2 trap_v2.c -fopenmp -std=c99 -Wall -O3 -lm
 *
 * Author: Naga Kandasamy
 * Date created: April 15, 2011
 * Date modified: April 26, 2020
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

/* Function prototype */
double f(double); 

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
    
    double h = (b - a) / n;               /* Number of subintervals */
    double approximate_area = (f(a) + f(b)) / 2.0;

    int i;
#pragma omp parallel for num_threads(thread_count) private(i) reduction(+: approximate_area)
    for (i = 1; i <= (n - 1); i++) {
        approximate_area += f(a + i * h);
    }
    
    approximate_area = h * approximate_area; 

    fprintf(stderr, "With %d trapeziods, the estimate for the integral between [%f, %f] is %f\n", n, a, b, approximate_area);
	
    exit(EXIT_SUCCESS);
}

/* Function to integrate */
double f(double x)                                                    
{
    return sqrt((1 + x*x)/(1 + x*x*x*x));
}

