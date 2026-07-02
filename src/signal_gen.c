#include "matrix.h"
#include <stdlib.h>
#include <time.h>

void generate_random_data(double *data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = (double)rand() / RAND_MAX;
    }
}

Matrix** generate_filter_bank(int *dims, int n) {
    Matrix **mats = (Matrix**)malloc(n * sizeof(Matrix*));
    for (int i = 0; i < n; i++) {
        mats[i] = matrix_create(dims[i], dims[i+1]);
        generate_random_data(mats[i]->data, dims[i] * dims[i+1]);
    }
    return mats;
}

Vector* generate_signal_vector(int size) {
    Vector *v = vector_create(size);
    generate_random_data(v->data, size);
    return v;
}
