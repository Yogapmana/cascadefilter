#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include "matrix.h"
#include "dp_order.h"
#include "verify.h"

// External function declarations
Matrix** generate_filter_bank(int *dims, int n);
Vector* generate_signal_vector(int size);
Matrix* multiply_naive(Matrix **matrices, int n);
Matrix* multiply_sequential(Matrix *A, Matrix *B);
Matrix* multiply_parallel_data(Matrix *A, Matrix *B);
Matrix* execute_tree_parallel_wrapper(Matrix **mats, int **s, int i, int j);
Vector* apply_filter(Matrix *M_combined, Vector *X);

double get_time() {
    return omp_get_wtime();
}

int main(int argc, char *argv[]) {
    srand(time(NULL));

    // Default configuration: Functional Verification (Small Scenario)
    int n = 4;
    int *dims = NULL;

    // If dimensions are provided via command line arguments
    if (argc > 2) {
        n = atoi(argv[1]);
        if (argc != n + 3) {
            printf("Usage: %s <n> <dim0> <dim1> ... <dimN>\n", argv[0]);
            return 1;
        }
        dims = (int*)malloc((n + 1) * sizeof(int));
        for (int i = 0; i <= n; i++) {
            dims[i] = atoi(argv[i + 2]);
        }
    } else {
        dims = (int*)malloc(5 * sizeof(int));
        dims[0] = 10; dims[1] = 150; dims[2] = 20; dims[3] = 80; dims[4] = 10;
    }

    printf("==================================================\n");
    printf("               BENCHMARK CASCADEFILTER            \n");
    printf("==================================================\n");
    printf("Jumlah Matriks Filter        : %d\n", n);
    printf("Dimensi Rantai Matriks       : ");
    for (int i = 0; i <= n; i++) {
        printf("%d", dims[i]);
        if (i < n) printf(" x ");
    }
    printf("\n");

    // 1. Generate Input
    Matrix **matrices = generate_filter_bank(dims, n);
    Vector *X = generate_signal_vector(dims[n]);

    // 2. Compute DP
    DPResult dp_res = compute_dp_order(dims, n);
    
    // Compute ops for naive (left-to-right)
    double ops_naive = 0;
    int cur_rows = dims[0];
    int cur_cols = dims[1];
    for (int i = 2; i <= n; i++) {
        ops_naive += (double)cur_rows * cur_cols * dims[i];
        cur_cols = dims[i];
    }
    double ops_dp = dp_res.m[1][n];
    double dp_reduction = (ops_naive - ops_dp) / ops_naive * 100.0;
    
    printf("Pengurangan Operasi Perkalian: %.2f%% (Memangkas beban matematika oleh DP)\n\n", dp_reduction);

    // 3. Baseline: Naive Sequential (left-to-right)
    double start = get_time();
    Matrix *M_baseline = multiply_naive(matrices, n);
    double end = get_time();
    double t_baseline = end - start;

    // 4. Sequential + DP
    start = get_time();
    Matrix *M_seq_dp = execute_optimal_order(matrices, dp_res.s, 1, n, multiply_sequential);
    end = get_time();
    double t_seq_dp = end - start;

    // 5. Parallel Data + DP
    start = get_time();
    Matrix *M_par_dp = execute_optimal_order(matrices, dp_res.s, 1, n, multiply_parallel_data);
    end = get_time();
    double t_par_dp = end - start;

    // 6. Parallel Task + DP
    start = get_time();
    Matrix *M_task_dp = execute_tree_parallel_wrapper(matrices, dp_res.s, 1, n);
    end = get_time();
    double t_task_dp = end - start;

    // 7. Verify correctness
    double epsilon = 1e-6;
    if (!verify_equal(M_baseline, M_seq_dp, epsilon)) {
        printf("ERROR: Sequential DP output does not match Baseline!\n");
        return 1;
    }
    if (!verify_equal(M_baseline, M_par_dp, epsilon)) {
        printf("ERROR: Parallel Data DP output does not match Baseline!\n");
        return 1;
    }
    if (!verify_equal(M_baseline, M_task_dp, epsilon)) {
        printf("ERROR: Parallel Task DP output does not match Baseline!\n");
        return 1;
    }
    printf("Hasil Verifikasi Keakuratan  : SUKSES (Semua metode menghasilkan matriks yang identik, epsilon = %g)\n\n", epsilon);

    // 8. Output Benchmark Times
    printf("Waktu Eksekusi Perkalian Matriks (Pre-komputasi):\n");
    printf("  1. Naif Sekuensial (Tanpa Optimasi)     : %f detik\n", t_baseline);
    printf("  2. Sekuensial + DP (Optimasi Algoritma)  : %f detik\n", t_seq_dp);
    printf("  3. Paralel Data + DP (Optimasi Hardware) : %f detik\n", t_par_dp);
    printf("  4. Paralel Tugas + DP (Multi-threading)  : %f detik\n\n", t_task_dp);

    // Speedup Metrics
    double speedup_dp_only = t_baseline / t_seq_dp;
    double speedup_par_only = t_seq_dp / t_par_dp;
    double speedup_total = t_baseline / t_par_dp;
    
    int num_threads = omp_get_max_threads();
    double efficiency = speedup_par_only / num_threads;

    printf("Metrik Peningkatan Kecepatan (Speedup):\n");
    printf("  - Kontribusi Optimasi DP Saja (Algoritma)       : %.2f x lebih cepat\n", speedup_dp_only);
    printf("  - Kontribusi Paralelisasi Saja (Hardware)       : %.2f x lebih cepat\n", speedup_par_only);
    printf("  - Total Peningkatan Kecepatan (Naif vs Paralel) : %.2f x lebih cepat\n", speedup_total);
    printf("  - Efisiensi Paralel (pada %d Threads)           : %.2f%% (tingkat utilisasi core)\n\n", num_threads, efficiency * 100.0);

    // 9. Apply Filter to Signal
    printf("Simulasi Penerapan Filter pada Sinyal Suara:\n");
    start = get_time();
    Vector *Y1 = vector_create(dims[0]);
    // Apply one-by-one: X is dims[n]
    Vector *current_X = vector_create(X->size);
    for (int i=0; i<X->size; i++) current_X->data[i] = X->data[i];
    
    for (int i = n - 1; i >= 0; i--) { // matrices[0] is dim0 x dim1 ... matrices[n-1] is dimN-1 x dimN
        Vector *next_X = vector_create(matrices[i]->rows);
        for(int r=0; r<matrices[i]->rows; r++){
            double sum = 0;
            for(int c=0; c<matrices[i]->cols; c++){
                sum += matrices[i]->data[r * matrices[i]->cols + c] * current_X->data[c];
            }
            next_X->data[r] = sum;
        }
        vector_free(current_X);
        current_X = next_X;
    }
    Y1 = current_X;
    end = get_time();
    double t_apply_sequential = end - start;
    
    start = get_time();
    Vector *Y2 = apply_filter(M_par_dp, X);
    end = get_time();
    double t_apply_combined = end - start;

    printf("  - Metode Pemrosesan Satu-per-satu (Tanpa Penggabungan) : %f detik\n", t_apply_sequential);
    printf("  - Metode Matriks Gabungan (Hasil Pre-komputasi)        : %f detik\n", t_apply_combined);
    
    // Check if applied signals match
    int signal_match = 1;
    for(int i=0; i<Y1->size; i++) {
        double diff = fabs(Y1->data[i] - Y2->data[i]);
        double max_val = fmax(fabs(Y1->data[i]), fabs(Y2->data[i]));
        if (diff > epsilon && diff / (max_val + 1e-9) > epsilon) signal_match = 0;
    }
    if (signal_match) {
        printf("  Hasil Sinyal Akhir: SUKSES (Kedua metode menghasilkan suara yang identik)\n");
    } else {
        printf("  Hasil Sinyal Akhir: GAGAL (Ada perbedaan nilai sinyal antara kedua metode)\n");
    }

    // Cleanup
    for (int i = 0; i < n; i++) matrix_free(matrices[i]);
    free(matrices);
    dp_result_free(&dp_res);
    matrix_free(M_baseline);
    matrix_free(M_seq_dp);
    matrix_free(M_par_dp);
    matrix_free(M_task_dp);
    vector_free(X);
    vector_free(Y1);
    vector_free(Y2);
    free(dims);

    return 0;
}
