/* Program to illustrate creation of multiple threads. 
 *
 * Author: Naga Kandasamy
 * Date created: January 8, 2019
 * Date modified: February 9, 2020
 *
 * Compile as follows: gcc -o multiple_threads multiple_threads.c -std=c99 -O3 -Wall -lpthread
 *
 * Usage: ./multiple_threads num-threads 
*/

#define _REENTRANT /* Make sure the library functions are MT (muti-thread) safe */
#include <stdio.h>
#include <pthread.h>
#include <unistd.h>
#include <stdlib.h>
#include <math.h>

/* Function prototype for the thread routines */
void *worker(void *);

/* Structure used to pass arguments to the worker threads*/
typedef struct args_for_thread_t {
    int tid;                /* Thread ID */
    int arg1;               /* First argument */
    float arg2;             /* Second argument */
    int processing_time;    /* Third argument */
} ARGS_FOR_THREAD; 


int main(int argc, char **argv)
{
    if (argc != 2) {
        printf("Usage: %s num-threads\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    unsigned int num_threads ;
    num_threads = atoi(argv[1]); 
    
    pthread_t main_thread;
    /* Allocate memory to store the IDs of the worker threads */
    pthread_t *worker_thread = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    ARGS_FOR_THREAD *args_for_thread;
	
    int i;
    main_thread = pthread_self();
    printf("Main thread = %lu is creating %d worker threads\n", main_thread, num_threads);
	
    /* Fork point: create worker threads and ask them to execute worker that takes a structure as an argument */
    for (i = 0; i < num_threads; i++) {
        args_for_thread = (ARGS_FOR_THREAD *)malloc(sizeof(ARGS_FOR_THREAD)); /* Memory for structure to pack the arguments */
        args_for_thread->tid = i; /* Fill the structure with some dummy arguments */
        args_for_thread->arg1 = 5; 
        args_for_thread->arg2 = 2.5;
        args_for_thread->processing_time = 10 * (float)rand()/RAND_MAX; 
		
        if ((pthread_create(&worker_thread[i], NULL, worker, (void *)args_for_thread)) != 0) {
            perror("pthread_create");
            exit(EXIT_FAILURE);
        }
    }
		  
    /* Join point: wait for all the worker threads to finish */
    for (i = 0; i < num_threads; i++)
        pthread_join(worker_thread[i], NULL);
		
    /* Free data structures */
    free((void *)worker_thread);
    printf("Main thread exiting\n");
    pthread_exit((void *)main_thread);
}


/* Function that will be executed by all the worker threads */
void *worker(void *this_arg)
{
    ARGS_FOR_THREAD *args_for_me = (ARGS_FOR_THREAD *)this_arg; /* Typecast argument passed to function to appropriate type */
	
    /* Simulate some processing */
    printf("Thread %d is using args %d and %f\n", args_for_me->tid, args_for_me->arg1, args_for_me->arg2);
    int processing_time;
    processing_time = args_for_me->processing_time;
	
    sleep(processing_time);
	
    printf("Thread %d is done\n", args_for_me->tid);
    free((void *)args_for_me); /* Free up the argument structure */
    pthread_exit(NULL);
}







