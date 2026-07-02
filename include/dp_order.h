#ifndef DP_ORDER_H
#define DP_ORDER_H

#include "matrix.h"

typedef struct {
    double **m; // m[i][j] = min cost of chain i..j (using double because cost can be large)
    int **s;    // s[i][j] = optimal split point
    int n;      // number of matrices
} DPResult;

// Compute DP table for matrix dimensions. dims has length n+1
DPResult compute_dp_order(int *dims, int n);

void dp_result_free(DPResult *res);

// Execute multiplication using the optimal order
Matrix* execute_optimal_order(Matrix **matrices, int **s, int i, int j, multiply_fn_t multiply_fn);

#endif // DP_ORDER_H
