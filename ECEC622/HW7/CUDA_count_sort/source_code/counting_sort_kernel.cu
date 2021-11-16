/* Write GPU code to perform the step(s) involved in counting sort. 
 Add additional kernels and device functions as needed. */
#ifndef _COUNTING_SORT_KERNEL_H_
#define _COUNTING_SORT_KERNEL_H_

__global__ void counting_sort_kernel(int *input_data, int *histogram, int *sorted_array,
                                      int num_elements, int histogram_size)
{
   __shared__ unsigned int s[255];
   
   /* Initialize shared memory */ 
    if(threadIdx.x < histogram_size)
        s[threadIdx.x] = 0;
		
	__syncthreads();
	
	int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
	
	while (offset < num_elements) {
        atomicAdd(&s[input_data[offset]], 1);
        offset += stride;
    }
	
	__syncthreads();
	
	    /* Accumulate histogram in shared memory into global memory */
    if (threadIdx.x < histogram_size) 
        atomicAdd(&histogram[threadIdx.x], s[threadIdx.x]);
		
	    /* Dynamically allocated shared memory for storing the scan array */
    extern  __shared__  int temp[];
	
	    /* Indices for the ping-pong buffers */
    int pout = 0;
    int pin = 1;
	
	    /* Load the input array from global memory into shared memory */
    if (threadIdx.x> 0) 
        temp[pout * histogram_size + threadIdx.x] = histogram[threadIdx.x];
    else
        temp[pout * histogram_size + threadIdx.x] = 0;
		
	int offset1;
    for (offset1 = 1; offset1 < histogram_size; offset1 *= 2) {
        pout = 1 - pout;
        pin  = 1 - pout;
        __syncthreads();

        temp[pout * histogram_size + threadIdx.x] = temp[pin * histogram_size + threadIdx.x];

        if (threadIdx.x > offset1)
            temp[pout * histogram_size + threadIdx.x] += temp[pin * histogram_size + threadIdx.x - offset1];
    }
	
	__syncthreads();

    histogram[threadIdx.x] = temp[pout * histogram_size + threadIdx.x];
	
	int i,j;
	int start_idx = 0;
    for (i = 0; i < histogram_size - 1; i++) {
        for (j = start_idx; j <= histogram[i]; j++) {
            sorted_array[j] = i;
        }
        start_idx = histogram[i];
    }
	
	//while (threadIdx.x < num_elements) {
   //    for (j = start_idx; j <= histogram[threadIdx.x]; j++) {
    //        sorted_array[j] = threadIdx.x;
    //    }
    //    start_idx = histogram[threadIdx.x];
   // }
}

#endif /* _COUNTING_SORT_KERNEL_H_ */