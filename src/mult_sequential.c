#include "matrix.h"

Matrix* multiply_sequential(Matrix *A, Matrix *B) {
    Matrix *C = matrix_create(A->rows, B->cols);
    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < B->cols; j++) {
            double sum = 0;
            for (int k = 0; k < A->cols; k++) {
                sum += A->data[i * A->cols + k] * B->data[k * B->cols + j];
            }
            C->data[i * C->cols + j] = sum;
        }
    }
    return C;
}

// Function to multiply matrices naive left-to-right
Matrix* multiply_naive(Matrix **matrices, int n) {
    if (n == 1) {
        return matrices[0];
    }
    
    Matrix *result = multiply_sequential(matrices[0], matrices[1]);
    for (int i = 2; i < n; i++) {
        Matrix *temp = multiply_sequential(result, matrices[i]);
        if (i - 1 != 0) { // Don't free matrices[1] since it's an input matrix
            matrix_free(result);
        }
        result = temp;
    }
    return result;
}
