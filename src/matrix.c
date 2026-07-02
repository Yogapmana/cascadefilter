#include "matrix.h"
#include <stdio.h>

Matrix* matrix_create(int rows, int cols) {
    Matrix *m = (Matrix*)malloc(sizeof(Matrix));
    m->rows = rows;
    m->cols = cols;
    m->data = (double*)calloc(rows * cols, sizeof(double));
    return m;
}

void matrix_free(Matrix *m) {
    if (m) {
        free(m->data);
        free(m);
    }
}

void matrix_print(Matrix *m, const char *name) {
    printf("Matrix %s (%d x %d):\n", name, m->rows, m->cols);
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            printf("%6.2f ", m->data[i * m->cols + j]);
        }
        printf("\n");
    }
}

Vector* vector_create(int size) {
    Vector *v = (Vector*)malloc(sizeof(Vector));
    v->size = size;
    v->data = (double*)calloc(size, sizeof(double));
    return v;
}

void vector_free(Vector *v) {
    if (v) {
        free(v->data);
        free(v);
    }
}
