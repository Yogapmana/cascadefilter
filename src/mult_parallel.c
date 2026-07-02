#include "matrix.h"
#include <omp.h>

Matrix* transpose(Matrix *B) {
    Matrix *BT = matrix_create(B->cols, B->rows);
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < B->rows; i++) {
        for (int j = 0; j < B->cols; j++) {
            BT->data[j * BT->cols + i] = B->data[i * B->cols + j];
        }
    }
    return BT;
}

Matrix* multiply_parallel_data(Matrix *A, Matrix *B) {
    Matrix *BT = transpose(B);        // access column B becomes row BT access
    Matrix *C = matrix_create(A->rows, B->cols);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < B->cols; j++) {
            double sum = 0;
            for (int k = 0; k < A->cols; k++) {
                sum += A->data[i * A->cols + k] * BT->data[j * BT->cols + k]; // contiguous access
            }
            C->data[i * C->cols + j] = sum;
        }
    }
    matrix_free(BT);
    return C;
}
