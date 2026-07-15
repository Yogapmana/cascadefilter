#!/bin/bash
set -e

# Compile the project
make clean
make

echo "==========================================="
echo "  CascadeFilter Benchmark Script"
echo "==========================================="

echo ""
echo "--- Functional Verification (Small Scenario) ---"
# Default is the small scenario: 4 matrices, dims: 10 150 20 80 10
./cascade_filter

echo ""
echo "--- Medium Scenario (DP Contribution) ---"
# 5 matrices, dims: 256 512 512 128 128 64 (per plan)
./cascade_filter 5 256 512 512 128 128 64

echo ""
echo "--- Large Scenario (Parallel Scalability) ---"
# 10 matrices, scaled up dimensions (e.g. x4 roughly)
for threads in 1 2 4 8; do
    echo ">> Running with OMP_NUM_THREADS=$threads"
    export OMP_NUM_THREADS=$threads
    ./cascade_filter 10 2048 4096 4096 1024 1024 512 512 256 256 128 128
done
