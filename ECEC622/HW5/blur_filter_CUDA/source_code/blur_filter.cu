/* Reference code implementing the box blur filter.

    Build and execute as follows: 
        make clean && make 
        ./blur_filter size

    Author: Naga Kandasamy
    Date created: May 3, 2019
    Date modified: February 15, 2021

    Student name(s): Abishek S Kumar	
    Date modified: 02/19/2021
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

#define DEBUG 

/* Include the kernel code */
#include "blur_filter_kernel.cu"

extern "C" void compute_gold(const image_t, image_t);
void compute_on_device(const image_t, image_t);
image_t allocate_image_on_device(image_t);
void copy_image_to_device(image_t, image_t);
void copy_image_from_device(image_t, image_t);
void free_image_on_device(image_t *);
int check_results(const float *, const float *, int, float);
void check_CUDA_error(const char *);
void print_image(const image_t);

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s size\n", argv[0]);
        fprintf(stderr, "size: Height of the image. The program assumes size x size image.\n");
        exit(EXIT_FAILURE);
    }

    /* Allocate memory for the input and output images */
    int size = atoi(argv[1]);

    fprintf(stderr, "Creating %d x %d images\n", size, size);
    image_t in, out_gold, out_gpu;
    in.size = out_gold.size = out_gpu.size = size;
    in.element = (float *)malloc(sizeof(float) * size * size);
    out_gold.element = (float *)malloc(sizeof(float) * size * size);
    out_gpu.element = (float *)malloc(sizeof(float) * size * size);
    if ((in.element == NULL) || (out_gold.element == NULL) || (out_gpu.element == NULL)) {
        perror("Malloc");
        exit(EXIT_FAILURE);
    }

    /* Poplulate our image with random values between [-0.5 +0.5] */
    srand(time(NULL));
    int i;
    for (i = 0; i < size * size; i++)
        in.element[i] = rand()/(float)RAND_MAX -  0.5;
  
   /* Calculate the blur on the CPU. The result is stored in out_gold. */
    fprintf(stderr, "Calculating blur on the CPU\n"); 
	struct timeval start, stop;	
	gettimeofday(&start, NULL);
    compute_gold(in, out_gold); 
	gettimeofday(&stop, NULL);
	fprintf(stderr, "CPU execution time = %fs\n", (float)(stop.tv_sec - start.tv_sec +\
                                                     (stop.tv_usec - start.tv_usec)/(float)1000000));

//#ifdef DEBUG 
//   print_image(in);
//   print_image(out_gold);
//#endif

   /* FIXME: Calculate the blur on the GPU. The result is stored in out_gpu. */
   fprintf(stderr, "Calculating blur on the GPU\n");
   compute_on_device(in, out_gpu);
//   
//#ifdef DEBUG 
//	print_image(in);
//   print_image(out_gpu);
//#endif

   /* Check CPU and GPU results for correctness */
   fprintf(stderr, "Checking CPU and GPU results\n");
   int num_elements = out_gold.size * out_gold.size;
   float eps = 1e-6;    /* Do not change */
   int check;
   check = check_results(out_gold.element, out_gpu.element, num_elements, eps);
   if (check == 0) 
       fprintf(stderr, "TEST PASSED\n");
   else
       fprintf(stderr, "TEST FAILED\n");
   
   /* Free data structures on the host */
   free((void *)in.element);
   free((void *)out_gold.element);
   free((void *)out_gpu.element);

    exit(EXIT_SUCCESS);
}

/* FIXME: Complete this function to calculate the blur on the GPU */
void compute_on_device(const image_t in, image_t out)
{
    struct timeval start, stop;	
	gettimeofday(&start, NULL);
	fprintf(stderr, "in size = %d\n", in.size);
	fprintf(stderr, "out size = %d\n", out.size);
    /* Allocate memory and copy matrices to the device */
	int size = in.size * in.size *sizeof(float);
	fprintf(stderr, "size = %d\n", size);
	
	
	image_t in_d = allocate_image_on_device(in);
	//cudaMalloc((void**)&in_d.element, size);
	copy_image_to_device(in_d, in);
	//cudaMemcpy(in_d.element, in.element, size, cudaMemcpyHostToDevice);

	image_t out_d = allocate_image_on_device(out);
	//cudaMalloc((void**)&in_d.element, size);
	copy_image_to_device(out_d, out);
	//cudaMemcpy(in_d.element, in.element, size, cudaMemcpyHostToDevice);
	
	gettimeofday(&stop, NULL);
	fprintf(stderr, "Data transfer time = %fs\n", (float)(stop.tv_sec - start.tv_sec +\
                                                  (stop.tv_usec - start.tv_usec)/(float)1000000));
	
	fprintf(stderr, "in_d size = %d\n", in_d.size);
	fprintf(stderr, "out_d size = %d\n", out_d.size);
	fprintf(stderr, "TILE_SIZE = %d\n", TILE_SIZE);
    /* Set up the execution grid */
    dim3 threads(TILE_SIZE, TILE_SIZE); 
	fprintf(stderr, "Setting up a %d x %d grid of thread blocks\n", (out_d.size + TILE_SIZE - 1)/TILE_SIZE,\\
            (out_d.size + TILE_SIZE - 1)/TILE_SIZE);
	dim3 grid((out_d.size + TILE_SIZE - 1)/TILE_SIZE, (out_d.size + TILE_SIZE - 1)/TILE_SIZE);
	
	gettimeofday(&start, NULL);
	
	/* Launch kernel */
	blur_filter_kernel <<< grid, threads >>>(in_d.element, out_d.element, in.size);
	cudaDeviceSynchronize(); /* Kernel execution is asynchronous; force CPU to wait here */
	
	gettimeofday(&stop, NULL);
	fprintf(stderr, "Kernel execution time = %fs\n", (float)(stop.tv_sec - start.tv_sec +\
                                                     (stop.tv_usec - start.tv_usec)/(float)1000000));

    check_CUDA_error("Error in kernel");
	
	copy_image_from_device(out, out_d); 
	//print_image(out);
	free_image_on_device(&in_d);
	free_image_on_device(&out_d);

}

/* Check correctness of results */
int check_results(const float *pix1, const float *pix2, int num_elements, float eps) 
{
    int i;
    for (i = 0; i < num_elements; i++)
        if (fabsf((pix1[i] - pix2[i])/pix1[i]) > eps) 
            return -1;
    
    return 0;
}

void free_image_on_device(image_t  *M)                              
{
	cudaFree(M->element);
	M->element = NULL;
}

/* Print out the image contents */
void print_image(const image_t img)
{
    int i, j;
    float val;
    for (i = 0; i < img.size; i++) {
        for (j = 0; j < img.size; j++) {
            val = img.element[i * img.size + j];
            printf("%0.4f ", val);
        }
        printf("\n");
    }

    printf("\n");
}

/* Allocate memory on device for image */
image_t allocate_image_on_device(image_t M)                        
{
	image_t Mdevice = M;
	int size = M.size * M.size * sizeof(float);
	
    cudaMalloc((void**)&Mdevice.element, size);
    if (Mdevice.element == NULL) {
        fprintf(stderr, "CudaMalloc error\n");
        exit(EXIT_FAILURE);
    }

	return Mdevice;
}

/* Copy image from host memory to device memory */
void copy_image_to_device(image_t Mdevice, image_t Mhost)      
{
	int size = Mhost.size * Mhost.size * sizeof(float);
	cudaMemcpy(Mdevice.element, Mhost.element, size, cudaMemcpyHostToDevice);
}

/* Copy matrix from device memory to host memory */
void copy_image_from_device(image_t Mhost, image_t Mdevice)   
{
	int size = Mdevice.size * Mdevice.size * sizeof(float);
	cudaMemcpy(Mhost.element, Mdevice.element, size, cudaMemcpyDeviceToHost);
}

/* Check for errors during kernel execution */
void check_CUDA_error(const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err) {
		fprintf(stderr, "CUDA ERROR: %s (%s).\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}	
}