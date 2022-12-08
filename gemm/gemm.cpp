
// gemm -- general double precision dense matrix-matrix multiplication.
//
// implement: C = alpha * A x B + beta * C, for matrices A, B, C
// Matrix C is M x N  (M rows, N columns)
// Matrix A is M x K
// Matrix B is K x N
//
// Your implementation should make no assumptions about the values contained in any input parameters.
#define BLOCK_SIZE 256

void gemm(int m, int n, int k, double *A, double *B, double *C, double alpha, double beta)
{

    // apply initial beta so in blocked implementation we can simply add
    for (int i = 0; i < m*m; i++) {
        C[i] *= beta;
    }

    int i, ii, j, jj, kk;
    double sum;
    for (kk = 0; kk < m; kk += BLOCK_SIZE) {
        for (jj = 0; jj < n; jj += BLOCK_SIZE) {
            for (i = 0; i < k; i++) {
                for (j = jj; j < jj + BLOCK_SIZE; j++) {
                    sum = C[i*n + j];
                    for (ii = kk; ii < kk + BLOCK_SIZE; ii++) {
                        sum += alpha * A[i*n + ii] * B[ii*n + j];
                    }
                    C[i*n + j] = sum;
                }
            }
        }
    }
}


