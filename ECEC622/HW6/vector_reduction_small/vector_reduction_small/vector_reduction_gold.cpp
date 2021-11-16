/* Reference code for vector reduction */

extern "C" double compute_gold(float *, int);

double compute_gold(float *input, int len) 
{
	double sum = 0.0;
    
    int i;    
	for (i = 0; i < len; i++)
		sum += input[i];

	return sum;
}

