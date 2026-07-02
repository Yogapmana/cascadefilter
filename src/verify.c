#include "verify.h"
#include <math.h>

int verify_equal(Matrix *A, Matrix *B, double epsilon) {
    if (!A || !B) return 0;
    if (A->rows != B->rows || A->cols != B->cols) return 0;
    for (int idx = 0; idx < A->rows * A->cols; idx++) {
        double diff = fabs(A->data[idx] - B->data[idx]);
        double max_val = fmax(fabs(A->data[idx]), fabs(B->data[idx]));
        if (diff > epsilon && diff / (max_val + 1e-9) > epsilon) {
            return 0; // Return false
        }
    }
    return 1; // Return true
}
