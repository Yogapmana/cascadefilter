#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include "matrix.h"
#include "dp_order.h"

// Forward declarations
Matrix* multiply_naive(Matrix **matrices, int n);
Matrix* multiply_sequential(Matrix *A, Matrix *B);
Matrix* multiply_parallel_data(Matrix *A, Matrix *B);
Vector* apply_filter(Matrix *M_combined, Vector *X);

Matrix* create_lowpass_matrix(int size, double intensity) {
    Matrix *m = matrix_create(size, size);
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if (i == j) {
                m->data[i * size + j] = 1.0 - intensity;
            } else if (abs(i - j) == 1) {
                m->data[i * size + j] = intensity / 2.0;
            } else if (abs(i - j) == 2) {
                m->data[i * size + j] = intensity / 4.0;
            } else {
                m->data[i * size + j] = 0.0;
            }
        }
    }
    return m;
}

int main(int argc, char *argv[]) {
    if (argc != 6) {
        fprintf(stderr, "Usage: %s <method> <chunk_size> <num_matrices> <input_file> <output_file>\n", argv[0]);
        fprintf(stderr, "Methods: baseline, seq_dp, par_dp\n");
        return 1;
    }

    char *method = argv[1];
    int chunk_size = atoi(argv[2]);
    int num_matrices = atoi(argv[3]);
    char *input_file = argv[4];
    char *output_file = argv[5];

    // Read input vector
    FILE *fin = fopen(input_file, "rb");
    if (!fin) {
        fprintf(stderr, "Error opening input file\n");
        return 1;
    }
    
    // Find file size
    fseek(fin, 0, SEEK_END);
    long fsize = ftell(fin);
    fseek(fin, 0, SEEK_SET);
    
    int total_samples = fsize / sizeof(double);
    double *input_data = (double*)malloc(total_samples * sizeof(double));
    fread(input_data, sizeof(double), total_samples, fin);
    fclose(fin);

    // Generate Filter Matrices
    int *dims = (int*)malloc((num_matrices + 1) * sizeof(int));
    for (int i = 0; i <= num_matrices; i++) dims[i] = chunk_size;
    
    Matrix **matrices = (Matrix**)malloc(num_matrices * sizeof(Matrix*));
    for (int i = 0; i < num_matrices; i++) {
        matrices[i] = create_lowpass_matrix(chunk_size, 0.5); 
    }

    // DP Order
    DPResult dp_res = compute_dp_order(dims, num_matrices);

    double start_time = omp_get_wtime();
    Matrix *M_combined = NULL;

    if (strcmp(method, "baseline") == 0) {
        M_combined = multiply_naive(matrices, num_matrices);
    } else if (strcmp(method, "seq_dp") == 0) {
        M_combined = execute_optimal_order(matrices, dp_res.s, 1, num_matrices, multiply_sequential);
    } else if (strcmp(method, "par_dp") == 0) {
        M_combined = execute_optimal_order(matrices, dp_res.s, 1, num_matrices, multiply_parallel_data);
    } else {
        fprintf(stderr, "Unknown method\n");
        return 1;
    }

    double engine_time = omp_get_wtime() - start_time;

    // Apply Filter to chunks
    double *output_data = (double*)malloc(total_samples * sizeof(double));
    Vector *chunk_in = vector_create(chunk_size);
    
    int num_chunks = total_samples / chunk_size;
    for (int c = 0; c < num_chunks; c++) {
        for (int i = 0; i < chunk_size; i++) {
            chunk_in->data[i] = input_data[c * chunk_size + i];
        }
        
        Vector *chunk_out = vector_create(chunk_size);
        for(int r=0; r<M_combined->rows; r++){
            double sum = 0;
            for(int k=0; k<M_combined->cols; k++){
                sum += M_combined->data[r * M_combined->cols + k] * chunk_in->data[k];
            }
            chunk_out->data[r] = sum;
        }

        for (int i = 0; i < chunk_size; i++) {
            output_data[c * chunk_size + i] = chunk_out->data[i];
        }
        vector_free(chunk_out);
    }

    // Handle remaining samples
    for(int i = num_chunks * chunk_size; i < total_samples; i++){
        output_data[i] = input_data[i];
    }

    // Write output
    FILE *fout = fopen(output_file, "wb");
    fwrite(output_data, sizeof(double), total_samples, fout);
    fclose(fout);

    // Print timing to stdout so Python can parse it
    printf("%f\n", engine_time);

    // Cleanup
    free(input_data);
    free(output_data);
    vector_free(chunk_in);
    for (int i = 0; i < num_matrices; i++) {
        if(M_combined != matrices[i]) matrix_free(matrices[i]);
    }
    matrix_free(M_combined);
    free(matrices);
    free(dims);
    dp_result_free(&dp_res);

    return 0;
}
