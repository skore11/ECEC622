#ifndef _COMPACT_KERNEL_H_
#define _COMPACT_KERNEL_H_

__global__ void scan_kernel(float *out, float *in, int n)
{
    /* Allocated shared memory for storing the scan array */
    __shared__  float temp[2 * NUM_ELEMENTS];

    int tid = threadIdx.x;

    /* Indices for the ping-pong buffers */
    int pout = 0;
    int pin = 1;

    /* Load the in array from global memory into shared memory */
    if (tid > 0) 
        temp[pout * n + tid] = in[tid - 1];
    else
        temp[pout * n + tid] = 0;

    int offset;
    for (offset = 1; offset < n; offset *= 2) {
        pout = 1 - pout;
        pin  = 1 - pout;
        __syncthreads();

        temp[pout * n + tid] = temp[pin * n + tid];

        if (tid >= offset)
            temp[pout * n + tid] += temp[pin * n + tid - offset];
    }

    __syncthreads();

    out[tid] = temp[pout * n + tid];
}


__global__ void compact_kernel(int *out, int *in, int n, 
                               int *new_n, int *scanned_flag)
{
    __shared__ int temp[NUM_ELEMENTS];
    __shared__ int flag[2 * NUM_ELEMENTS];

    int tid = threadIdx.x;
    
    /* Load input elements from global memory into shared memory */
    if (tid < NUM_ELEMENTS)
        temp[tid] = in[tid];
    else
        temp[tid] = 0;

    __syncthreads();

    /* Examine element in the temp array and if > 0 flag it as such */
    int pout = 0;
    int pin = 1;

    if (tid > 0) {
        if (temp[tid - 1] > 0)
            flag[pout * n + tid] = 1;
        else
            flag[pout * n + tid] = 0;
    }
    else
        flag[pout * n + tid] = 0;
              
    __syncthreads();

    /* Scan the flag array to generate addresses for the output elements. 
     * We are performing an exclusive scan. */
    int offset;
    for (offset = 1; offset < n; offset *= 2) {
        pout = 1 - pout;
        pin  = 1 - pout;
        __syncthreads();

        flag[pout * n + tid] = flag[pin * n + tid];

        if (tid >= offset)
            flag[pout * n + tid] += flag[pin * n + tid - offset];
    }
    __syncthreads();
    
    /* Write out the scanned flag values */
    scanned_flag[tid] = flag[pout * n + tid];

    /* Write output elements to their corresponding addresses */
    if (temp[tid] > 0)
        out[flag[pout * n + tid]] = temp[tid];
   
    /* Calculate number of compacted elements */ 
    if (tid == (blockDim.x - 1)) {
        if (temp[tid] > 0)
            *new_n = flag[pout * n + tid] + 1; 
        else
            *new_n = flag[pout * n + tid]; 
    }
}
#endif /* _COMPACT_KERNEL_H_ */
