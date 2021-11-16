#ifndef _SCAN_KERNEL_H_
#define _SCAN_KERNEL_H_

__global__ void scan_kernel(float *out, float *in, int n)
{
    /* Dynamically allocated shared memory for storing the scan array */
    extern  __shared__  float temp[];

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
#endif /* _SCAN_KERNEL_H_ */
