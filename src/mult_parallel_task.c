#include "matrix.h"
#include "dp_order.h"
#include <omp.h>

// Forward declaration from mult_parallel.c
Matrix* multiply_parallel_data(Matrix *A, Matrix *B);

// Task-parallel tree execution
Matrix* execute_tree_parallel(Matrix **mats, int **s, int i, int j) {
    if (i == j) return mats[i-1];
    
    Matrix *left, *right;
    
    #pragma omp task shared(left)
    left = execute_tree_parallel(mats, s, i, s[i][j]);
    
    #pragma omp task shared(right)
    right = execute_tree_parallel(mats, s, s[i][j]+1, j);
    
    #pragma omp taskwait
    
    Matrix *result = multiply_parallel_data(left, right);
    
    // Free intermediate results
    if (i != s[i][j]) {
        matrix_free(left);
    }
    if ((s[i][j] + 1) != j) {
        matrix_free(right);
    }
    
    return result;
}

// Wrapper to launch the tree parallel execution
Matrix* execute_tree_parallel_wrapper(Matrix **mats, int **s, int i, int j) {
    Matrix *result;
    #pragma omp parallel
    {
        #pragma omp single
        {
            result = execute_tree_parallel(mats, s, i, j);
        }
    }
    return result;
}
