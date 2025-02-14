#define NUM_TASKS 16  // i7-7700K machine has 4 cores, 8 HW threads

task void ispc_transpose_task(uniform double src[], uniform double dst[], uniform int n, uniform double beta)
{
    uniform int num_rows = n / taskCount;
    uniform int row_start = num_rows * taskIndex;
    uniform int row_end = row_start + num_rows;

    uniform int c, r;
    for (r = row_start; r < row_end; r++) {
        for(c = 0; c < n; c += programCount) {
            #pragma ignore warning(perf)
            dst[(c + programIndex)*n + r] = beta * src[r*n + c + programIndex];
        }
    }
}

export void ispc_transpose(uniform int n, uniform double src[], uniform double dst[], uniform double beta)
{
    launch[NUM_TASKS] ispc_transpose_task(src, dst, n, beta);
}

#define BLOCK_SIZE 256

task void gemm_ispc_task(
    uniform int n,
    uniform double A[],
    uniform double B[],
    uniform double C[],
    uniform double alpha)
{
    uniform int num_rows = n / taskCount;
    uniform int row_start = taskIndex * num_rows;
    uniform int row_end = num_rows + row_start;

    uniform unsigned int r, j, d, c_start, r_start;
    double sum;
    for (r_start = 0; r_start < n; r_start += BLOCK_SIZE) {
        for (c_start = 0; c_start < n; c_start += BLOCK_SIZE) {
            for (j = r_start; j < r_start + BLOCK_SIZE; j++) {
                for (r = row_start; r < row_end; r++) {
                    #pragma ignore warning(perf)
                    sum = 0;
                    for (d = c_start; d < c_start + BLOCK_SIZE; d += 8 * programCount) {
                        #pragma ignore warning(perf)
                        sum += A[r*n + d + programIndex] * B[j*n + d + programIndex];
                        sum += A[r*n + d + programIndex + programCount] * B[j*n + d + programIndex + programCount];
                        sum += A[r*n + d + programIndex + 2 * programCount] * B[j*n + d + programIndex + 2 * programCount];
                        sum += A[r*n + d + programIndex + 3 * programCount] * B[j*n + d + programIndex + 3 * programCount];
                        sum += A[r*n + d + programIndex + 4 * programCount] * B[j*n + d + programIndex + 4 * programCount];
                        sum += A[r*n + d + programIndex + 5 * programCount] * B[j*n + d + programIndex + 5 * programCount];
                        sum += A[r*n + d + programIndex + 6 * programCount] * B[j*n + d + programIndex + 6 * programCount];
                        sum += A[r*n + d + programIndex + 7 * programCount] * B[j*n + d + programIndex + 7 * programCount];
                    }
                    #pragma ignore warning(perf)
                    C[r*n + j] += alpha * reduce_add(sum);
                }
            }
        }
    }
}

export void gemm_ispc(
    uniform int m,
    uniform int n,
    uniform int k,
    uniform double A[],
    uniform double B[],
    uniform double C[],
    uniform double alpha)
{
    launch[NUM_TASKS] gemm_ispc_task(m, A, B, C, alpha);
}
