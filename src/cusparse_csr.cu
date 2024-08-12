#include "../include/cusparse_csr.cuh"
#include "../include/utilities.cuh"

void cusparse_CSR(std::string fileName) {
    int rows, cols, nnz;
    int *csrRowPointers, *csrColumnIndices;
    dtype *csrValues;

    parseCsvToCsr(rows, cols, nnz, csrRowPointers, csrColumnIndices, csrValues, fileName);

    // Host memory management
    int* cscRowIndices = (int*)malloc(nnz * sizeof(int));
    int* cscColumnPointers = (int*)malloc((cols + 1) * sizeof(int));
    dtype* cscValues = (dtype*)malloc(nnz * sizeof(dtype));

    // Device memory management
    int *d_csrRowPointers, *d_csrColumnIndices;
    dtype *d_csrValues;

    int *d_cscRowIndices, *d_cscColumnPointers;
    dtype *d_cscValues;

    CHECK(cudaMalloc(&d_csrRowPointers, (rows + 1) * sizeof(int)));
    CHECK(cudaMalloc(&d_csrColumnIndices, nnz * sizeof(int)));
    CHECK(cudaMalloc(&d_csrValues, nnz * sizeof(dtype)));
    CHECK(cudaMalloc(&d_cscRowIndices, nnz * sizeof(int)));
    CHECK(cudaMalloc(&d_cscColumnPointers, (cols + 1) * sizeof(int)));
    CHECK(cudaMalloc(&d_cscValues, nnz * sizeof(dtype)));

    CHECK(cudaMemcpy(d_csrRowPointers, csrRowPointers, (rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_csrColumnIndices, csrColumnIndices, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_csrValues, csrValues, nnz * sizeof(dtype), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    float milliseconds = 0.0;

    // cuSparse APIs
    cusparseHandle_t handle;
    void *dBuffer = NULL;
    size_t bufferSize = 0;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    // Reserve buffer
    CHECK_CUSPARSE(
        cusparseCsr2cscEx2_bufferSize(
            handle,
            rows,
            cols,
            nnz,
            d_csrValues,
            d_csrRowPointers,
            d_csrColumnIndices,
            d_cscValues,
            d_cscColumnPointers,
            d_cscRowIndices,
            CUDA_R_64F, // maybe change to CUDA_R_64F
            CUSPARSE_ACTION_NUMERIC,
            CUSPARSE_INDEX_BASE_ZERO,
            CUSPARSE_CSR2CSC_ALG1,
            &bufferSize
        )
    );

    CHECK(cudaMalloc(&dBuffer, bufferSize));

    // Convert CSR to CSC
    CHECK(cudaEventRecord(start));
    CHECK_CUSPARSE(
        cusparseCsr2cscEx2(
            handle,
            rows,
            cols,
            nnz,
            d_csrValues,
            d_csrRowPointers,
            d_csrColumnIndices,
            d_cscValues,
            d_cscColumnPointers,
            d_cscRowIndices,
            CUDA_R_64F, // maybe change to CUDA_R_64F
            CUSPARSE_ACTION_NUMERIC,
            CUSPARSE_INDEX_BASE_ZERO,
            CUSPARSE_CSR2CSC_ALG1,
            dBuffer
        )
    );
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    // Copy results back to host
    CHECK(cudaMemcpy(cscRowIndices, d_cscRowIndices, nnz * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(cscColumnPointers, d_cscColumnPointers, (cols + 1) * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(cscValues, d_cscValues, nnz * sizeof(dtype), cudaMemcpyDeviceToHost));

    // Print results
    // printf("cscRowIndices: ");
    // for (int i = 0; i < nnz; i++) {
    //     printf("%d ", cscRowIndices[i]);
    // }
    // printf("\n");

    // printf("cscColumnPointers: ");
    // for (int i = 0; i < cols + 1; i++) {
    //     printf("%d ", cscColumnPointers[i]);
    // }
    // printf("\n");

    // printf("cscValues: ");
    // for (int i = 0; i < nnz; i++) {
    //     printf("%f ", cscValues[i]);
    // }
    // printf("\n");

    dtype* groundTruth = generateGroundTruthFromMTX(fileName);

    printf("Performed CSR to CSC conversion on matrix %s\n", fileName.c_str());
    if (checkResultCSR(groundTruth, cscColumnPointers, cscRowIndices, cscValues, rows, cols)) {
        printf("Status: ");
        // green color
        printf("\033[1;32m");
        printf("Correct\n");
        printf("\033[0m");
        printf("--------------------------------\n");
    } else {
        printf("Status: ");
        // red color
        printf("\033[1;31m");
        printf("Incorrect\n");
        printf("\033[0m");
        printf("--------------------------------\n");
    }


    // Free device memory
    CHECK_CUSPARSE(cusparseDestroy(handle));
    CHECK(cudaFree(d_csrRowPointers));
    CHECK(cudaFree(d_csrColumnIndices));
    CHECK(cudaFree(d_csrValues));
    CHECK(cudaFree(d_cscRowIndices));
    CHECK(cudaFree(d_cscColumnPointers));
    CHECK(cudaFree(d_cscValues));

    // Free host memory
    free(csrRowPointers);
    free(csrColumnIndices);
    free(csrValues);
    free(cscRowIndices);
    free(cscColumnPointers);
    free(cscValues);

}