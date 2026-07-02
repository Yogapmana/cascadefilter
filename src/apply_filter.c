#include "matrix.h"

// Apply combined matrix to signal vector: Y = M_combined * X
Vector* apply_filter(Matrix *M_combined, Vector *X) {
    if (M_combined->cols != X->size) {
        return NULL; // Dimension mismatch
    }
    
    Vector *Y = vector_create(M_combined->rows);
    for (int i = 0; i < M_combined->rows; i++) {
        double sum = 0;
        for (int k = 0; k < M_combined->cols; k++) {
            sum += M_combined->data[i * M_combined->cols + k] * X->data[k];
        }
        Y->data[i] = sum;
    }
    return Y;
}
