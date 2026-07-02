#ifndef VERIFY_H
#define VERIFY_H

#include "matrix.h"

// Verifies if two matrices are equal within epsilon tolerance
int verify_equal(Matrix *A, Matrix *B, double epsilon);

#endif // VERIFY_H
