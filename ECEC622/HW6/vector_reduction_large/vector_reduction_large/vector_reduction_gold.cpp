/* Reference code for vector reduction */

extern "C" double compute_gold(float *, int);

double compute_gold(float* A, int num_elements)
{
    int i;
    double sum = 0.0; 
  
    for (i = 0; i < num_elements; i++) 
        sum += A[i];
  
    return sum;
}







