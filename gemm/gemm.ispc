export void gemm_ispc(
    uniform int m,
    uniform int n,
    uniform int k,
    uniform double A[],
    uniform double B[],
    uniform double C[],
    uniform double alpha,
    uniform double beta
) {
    uniform unsigned int j, d;
    varying double sum;
    foreach (r = 0 ... m) {
        for (j = 0; j < m; j++) {
            #pragma ignore warning(perf)
            sum = beta * C[r*m + j];
            for (d = 0; d < m; d++) {
                #pragma ignore warning(perf)
                sum += alpha * A[r*m + d] * B[d*m + j];
            }
            #pragma ignore warning(perf)
            C[r*m + j] = sum;
        }
    }
}
