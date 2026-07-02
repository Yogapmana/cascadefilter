#ifndef MATRIX_H
#define MATRIX_H

#include <stdlib.h>

typedef struct {
    int rows, cols;
    double *data;  // data[i*cols + j]
} Matrix;

typedef struct {
    int size;
    double *data;
} Vector;

Matrix* matrix_create(int rows, int cols);
void matrix_free(Matrix *m);
void matrix_print(Matrix *m, const char *name);

Vector* vector_create(int size);
void vector_free(Vector *v);

// Multiplication function pointer type
typedef Matrix* (*multiply_fn_t)(Matrix*, Matrix*);

#endif // MATRIX_H
