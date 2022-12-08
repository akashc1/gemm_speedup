#define NUM_TASKS 8  // i7-7700K machine has 4 cores, 8 HW threads
#define SIMD_WIDTH 4 // 256-bit wide instructions / 8-byte double precision values

task void ispc_transpose_task(uniform double src[], uniform double dst[], uniform int n)
{
    uniform int num_rows = n / taskCount;
    uniform int row_start = num_rows * taskIndex;
    uniform int row_end = row_start + num_rows;

    uniform int c, r;
    for (r = row_start; r < row_end; r++) {
        for(c = 0; c < n; c += programCount) {
            #pragma ignore warning(perf)
            dst[(c + programIndex)*n + r] = src[r*n + c + programIndex];
        }
    }
}

export void ispc_transpose(uniform int n, uniform double src[], uniform double dst[])
{
    launch[NUM_TASKS] ispc_transpose_task(src, dst, n);
}

task void gemm_ispc_task(
    uniform int n,
    uniform double A[],
    uniform double B[],
    uniform double C[],
    uniform double alpha,
    uniform double beta)
{
    uniform int num_rows = n / taskCount;
    uniform int row_start = taskIndex * num_rows;
    uniform int row_end = num_rows + row_start;

    foreach (r = row_start ... row_end) {
        uniform unsigned int j, d;
        varying double sum;
        for (j = 0; j < n; j++) {
            #pragma ignore warning(perf)
            sum = beta * C[r*n + j];
            for (d = 0; d < n; d++) {
                #pragma ignore warning(perf)
                sum += alpha * A[r*n + d] * B[j*n + d];
            }
            #pragma ignore warning(perf)
            C[r*n + j] = sum;
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
    uniform double alpha,
    uniform double beta)
{
    launch[NUM_TASKS] gemm_ispc_task(m, A, B, C, alpha, beta);
}
