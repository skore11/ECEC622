/* Linked list functions for multi-threaded program: 
 * 1. Create and delete the linked list.
 * 2. Insert and delete elements from the linked list: insert() and delete()
 * 3. Check if element exists in the linked list: is_present()
 *
 * Note: 
 * 1. The linked list is maintained in sorted fashion at all times. 
 * 2. Duplicate elements are not allowed in the list.
 *
 * The operations have the following distribution: 80% is_present(), 10% insert(), amd 10% delete()
 *
 * The program illustrates locking at very coarse granularity: prior to performing any linked-list 
 * operation, including just reading the list elements, threads must acquire lock to the entire 
 * linked list data structure. 
 *
 * Compile as follows: gcc -o multi_threaded_linked_list_v1 multi_threaded_linked_list_v1.c -std=c99 -Wall -O3 -lpthread -lm
 *
 * Author: Naga Kandasamy
 * Date created: April 19, 2020
 * Date modified: 
*/

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#include <pthread.h>

/* Set DEBUG to 1 to print debug information */
#define DEBUG 0

#define MAX_VALUE 100

/* Define linked list data structure */
typedef struct element_s {
    int value;                      /* Value of the linked-list element */
    struct element_s *next;         /* Pointer to next linked-list element */
} element_t;

typedef struct linked_list_s {
    element_t *head;                /* Head of the linked list */
    element_t *tail;                /* Tail of the linked list */
    pthread_mutex_t lock;          /* Lock variable protecting linked list */
} linked_list_t;

/* Define thread data */
typedef struct thread_data_s {
    int tid;                        /* Thread ID */
    int num_trials;                 /* Number of trials */
    linked_list_t *list;            /* Pointer to linked list */
} thread_data_t;

/* Function prototypes */
linked_list_t *create_list(void);
void destroy_list(linked_list_t *list);
void print_list(linked_list_t *list);
int insert(linked_list_t *list, int value);
int delete(linked_list_t *list, int value);
int is_present(linked_list_t *list, int value);
void run_test(linked_list_t *list, int num_trials, int num_threads);
void *worker(void *arg);

int main(int argc, char **argv) 
{
    linked_list_t *list;

    if (argc < 3) {
        fprintf(stderr, "Usage: %s num-trials num-threads\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    int num_trials = atoi(argv[1]);
    int num_threads = atoi(argv[2]);
    
    /* Create initial linked list with 5000 elements in the range [0, MAX_VALUE] */
    srand(time(NULL));

    list = create_list();
    int i, r;
    for (i = 0; i < 5000; i++) {
        r = rand() % MAX_VALUE;
        insert(list, r);
    }

    struct timeval start, stop;	
	gettimeofday(&start, NULL);

    run_test(list, num_trials, num_threads);

    gettimeofday(&stop, NULL);
	printf("Execution time = %fs\n", (float)(stop.tv_sec - start.tv_sec\
                + (stop.tv_usec - start.tv_usec)/(float)1000000));
    
#if (DEBUG == 1)
    print_list(list);
#endif

    /* Free data structure */
    destroy_list(list);
           
    exit(EXIT_SUCCESS);
}

/* Run trials using multi-threaded version */
void run_test(linked_list_t *list, int num_trials, int num_threads)
{
    pthread_t *tid = (pthread_t *)malloc (num_threads * sizeof(pthread_t)); /* Data structure to store the thread IDs */
		  
    /* Fork point: allocate memory on heap for required data structures and create worker threads */
    int i;
    thread_data_t *thread_data = (thread_data_t *) malloc(sizeof(thread_data_t) * num_threads);	  
    for (i = 0; i < num_threads; i++) {
        thread_data[i].tid = i; 
        thread_data[i].num_trials = num_trials/num_threads; /* Split the number of trials between the threads */
        thread_data[i].list = list; 
    }

    for (i = 0; i < num_threads; i++)
        pthread_create(&tid[i], NULL, worker, (void *)&thread_data[i]);
					 
    /* Join point: wait for the workers to finish */
    for (i = 0; i < num_threads; i++)
        pthread_join(tid[i], NULL);

    return;
}

/* Worker thread */
void *worker(void *arg)
{
    thread_data_t *thread_data = (thread_data_t *)arg;
    linked_list_t *list = thread_data->list;

    int i, r;
    float op;
    int num_inserts = 0;
    int num_deletes = 0;
    int num_queries = 0; 
    for (i = 0; i < thread_data->num_trials; i++) {
        r = rand() % MAX_VALUE;
        
        op = rand()/(float)RAND_MAX;
        if ((op >= 0) && (op < 0.1)) {          /* Insert element in list */
            pthread_mutex_lock(&list->lock);
            insert(list, r);
            pthread_mutex_unlock(&list->lock);
            num_inserts++;
        } 
        else if ((op >= 0.1) && (op < 0.2)) {   /* Delete element from list */
            pthread_mutex_lock(&list->lock);
            delete(list, r);
            pthread_mutex_unlock(&list->lock);
            num_deletes++;
        } 
        else {                                  /* Check if element is present in list */
            pthread_mutex_lock(&list->lock);
            is_present(thread_data->list, r);
            pthread_mutex_unlock(&list->lock);
            num_queries++;
        }
    }
    
    fprintf(stderr, "Number of trials conducted by thread %d: %d\n", thread_data->tid, thread_data->num_trials);
    fprintf(stderr, "Inserts: %d, deletes: %d, queries: %d\n", num_inserts, num_deletes, num_queries);

    pthread_exit(NULL);
}

/* Create an empty list */
linked_list_t *create_list(void)
{
    linked_list_t *list;
    list = (linked_list_t *)malloc(sizeof(linked_list_t));
    if (list == NULL)
        return NULL;
    
    list->head = NULL;
    list->tail = NULL;
    pthread_mutex_init(&list->lock, NULL);   /* Initialize the mutex */
    return list;
}

/* Destroy existing linked list */
void destroy_list(linked_list_t *list)
{
    if (list == NULL)
        return;
    
    element_t *curr = list->head;
    element_t *to_delete;
    while (curr != NULL) {
        to_delete = curr;
        curr = curr->next;
        free((void *)to_delete);
    }

    pthread_mutex_destroy(&list->lock); /* Destroy mutex */
    return;
}

/* Print list elements */
void print_list(linked_list_t *list)
{
    element_t *curr = list->head;
    while (curr != NULL) {
        fprintf(stderr, "%d\n", curr->value);
        curr = curr->next;
    }

    fprintf(stderr, "\n");
    return;
}

/* Insert value, if it doesn't already exist, in sorted fashion in list */
int insert(linked_list_t *list, int value)
{
    element_t *curr, *prev;
   
    if (list->head == NULL) {   /* List is empty */
        element_t *new = (element_t *)malloc(sizeof(element_t)); /* Create linked-list node for the new element */
        new->value = value;
        new->next = NULL;
        
        list->head = new;
        return 0;
    }

    curr = list->head;
    if (value < curr->value) {  /* Insert value as first element */   
        element_t *new = (element_t *)malloc(sizeof(element_t)); /* Create linked-list node for the new element */
        new->value = value;
        new->next = curr;
        list->head = new;
        return 0;
    }

    /* Search for the correct location within list */
    prev = curr;
    curr = curr->next;
    while (curr != NULL) {
        if ((value > prev->value) && (value < curr->value)) {               
            element_t *new = (element_t *)malloc(sizeof(element_t)); /* Create linked-list node for the new element */
            new->value = value;
            /* Insert between prev and curr */
            prev->next = new; 
            new->next = curr;
            return 0;
        }

        prev = curr;
        curr = curr->next;
    }

    /* Reached the end of list */
    if (value > prev->value) {  /* Insert value as last element */
         element_t *new = (element_t *)malloc(sizeof(element_t)); /* Create linked-list node for the new element */
         new->value = value;
         new->next = NULL;

         prev->next = new;
         return 0;
    }

    /* Value already exists in the list */
    return -1;
}

/* Delete element, if it exists from the list */
int delete(linked_list_t *list, int value)
{
    element_t *curr, *prev;

    curr = list->head;
    if (curr == NULL)
        return -1;

    if (curr->value == value) { /* Delete first element */
        list->head = curr->next;
        free((void *)curr);
        return 0;
    }

    prev = curr;
    curr = curr->next;
    while (curr != NULL) {
        if (curr->value == value) {
            prev->next = curr->next;
            free((void *)curr);
            return 0;
        }
        prev = curr;
        curr = curr->next;
    }

    /* Element not in list */
    return -1;
}

/* Check if value is present in the list */
int is_present(linked_list_t *list, int value)
{
    element_t *curr = list->head;
    if (curr == NULL) /* Empty list */ 
        return -1;

    while (curr != NULL) {
        if (curr->value == value)
            return 0;
        curr = curr->next;
    }

    return -1;
}
