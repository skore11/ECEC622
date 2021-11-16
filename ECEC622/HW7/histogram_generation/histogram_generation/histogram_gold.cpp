#include <stdlib.h>
#include <stdio.h>
#include <string.h>

/* Reference implementation. */
extern "C" void compute_gold(int *, int *, int, int);

void compute_gold (int *input_data, int *histogram, int num_elements, int histogram_size)
{
    int i;
    for (i = 0; i < num_elements; i++)
        histogram[input_data[i]]++;
}

void print_histogram(int *bin, int num_bins, int num_elements)
{
    int num_histogram_entries = 0;
    int i;

    for (i = 0; i < num_bins; i++) {
        printf("Bin %d: %d\n", i, bin[i]);
        num_histogram_entries += bin[i];
    }

    printf("Number of elements in the input array = %d \n", num_elements);
    printf("Number of histogram elements = %d \n", num_histogram_entries);

    return;
}