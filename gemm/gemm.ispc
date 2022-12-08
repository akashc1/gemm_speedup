#define NUM_TASKS 8  // i7-7700K machine has 4 cores, 8 HW threads
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
                sum += alpha * A[r*n + d] * B[d*n + j];
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
