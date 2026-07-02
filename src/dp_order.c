#include "dp_order.h"
#include <stdlib.h>
#include <float.h>

DPResult compute_dp_order(int *dims, int n) {
    DPResult res;
    res.n = n;
    res.m = (double**)malloc((n + 1) * sizeof(double*));
    res.s = (int**)malloc((n + 1) * sizeof(int*));
    for (int i = 0; i <= n; i++) {
        res.m[i] = (double*)malloc((n + 1) * sizeof(double));
        res.s[i] = (int*)malloc((n + 1) * sizeof(int));
        for (int j = 0; j <= n; j++) {
            res.m[i][j] = 0;
            res.s[i][j] = 0;
        }
    }

    for (int L = 2; L <= n; L++) {
        for (int i = 1; i <= n - L + 1; i++) {
            int j = i + L - 1;
            res.m[i][j] = DBL_MAX;
            for (int k = i; k <= j - 1; k++) {
                // Cost = ops for left + ops for right + cost of multiplying left*right
                double q = res.m[i][k] + res.m[k + 1][j] + (double)dims[i - 1] * dims[k] * dims[j];
                if (q < res.m[i][j]) {
                    res.m[i][j] = q;
                    res.s[i][j] = k;
                }
            }
        }
    }
    return res;
}

void dp_result_free(DPResult *res) {
    for (int i = 0; i <= res->n; i++) {
        free(res->m[i]);
        free(res->s[i]);
    }
    free(res->m);
    free(res->s);
}

Matrix* execute_optimal_order(Matrix **matrices, int **s, int i, int j, multiply_fn_t multiply_fn) {
    if (i == j) {
        return matrices[i - 1]; // matrices is 0-indexed, dp table is 1-indexed
    }
    
    Matrix *left = execute_optimal_order(matrices, s, i, s[i][j], multiply_fn);
    Matrix *right = execute_optimal_order(matrices, s, s[i][j] + 1, j, multiply_fn);
    
    Matrix *result = multiply_fn(left, right);
    
    // Free intermediate results, but DO NOT free the original matrices
    if (i != s[i][j]) {
        matrix_free(left);
    }
    if ((s[i][j] + 1) != j) {
        matrix_free(right);
    }
    
    return result;
}
