#include "../include/cusparse_coo.cuh"
#include "../include/utilities.cuh"

void cusparse_COO(std::string fileName) {
    int rows, cols, nnz;
    int* row, *col;
    dtype* val;

    parseCsvToCoo(rows, cols, nnz, row, col, val, fileName);
    dtype* groundTruth = generateCOOGroundTruth(rows, cols, nnz, row, col, val);

    int size = rows * cols;

    dtype* h_identityMatrix = (dtype*)malloc(size * sizeof(dtype));
    dtype* h_result = (dtype*)malloc(size * sizeof(dtype));

    for (int i = 0; i < size; i++) {
        h_result[i] = 0.0;
    }

    // Fill the identity matrix
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (i == j) {
                h_identityMatrix[j * rows + i] = 1.0;
            } else {
                h_identityMatrix[j * rows + i] = 0.0;
            }
        }
    }

    float alpha = 1.0;
    float beta = 0.0;

    // Device memory management
    int *d_row, *d_col;
    dtype *d_val, *d_identityMatrix, *d_result;

    CHECK(cudaMalloc(&d_row, nnz * sizeof(int)));
    CHECK(cudaMalloc(&d_col, nnz * sizeof(int)));
    CHECK(cudaMalloc(&d_val, nnz * sizeof(dtype)));
    CHECK(cudaMalloc(&d_identityMatrix, size * sizeof(dtype)));
    CHECK(cudaMalloc(&d_result, size * sizeof(dtype)));

    CHECK(cudaMemcpy(d_row, row, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_col, col, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_val, val, nnz * sizeof(dtype), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_identityMatrix, h_identityMatrix, size * sizeof(dtype), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_result, h_result, size * sizeof(dtype), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    float milliseconds = 0.0;

    // cuSparse APIs
    cusparseHandle_t handle;
    cusparseSpMatDescr_t sparseMatrix;
    cusparseDnMatDescr_t identityMatrix, resultMatrix;
    void *dBuffer = NULL;
    size_t bufferSize = 0;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    // Create sparse matrix
    CHECK_CUSPARSE(
        cusparseCreateCoo(
            &sparseMatrix, rows, cols, nnz,
            d_row, d_col, d_val,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F
        )
    );

    // Create identity matrix
    CHECK_CUSPARSE(
        cusparseCreateDnMat(
            &identityMatrix, rows, cols, rows,
            d_identityMatrix, CUDA_R_32F, CUSPARSE_ORDER_COL
        )
    );

    // Create result matrix
    CHECK_CUSPARSE(
        cusparseCreateDnMat(
            &resultMatrix, rows, cols, rows,
            d_result, CUDA_R_32F, CUSPARSE_ORDER_COL
        )
    );

    // Allocate buffer
    /**
     * This is only needed when the CUSPARSE_ALG requires additional memory
     * Indeed, only the CUSPARSE_SPMM_COO_ALG2 requires additional memory
     */
    CHECK_CUSPARSE(
        cusparseSpMM_bufferSize(
            handle, 
            CUSPARSE_OPERATION_NON_TRANSPOSE, 
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, sparseMatrix, identityMatrix, &beta, resultMatrix,
            CUDA_R_32F, CUSPARSE_SPMM_COO_ALG1, &bufferSize
        )
    );

    CHECK(cudaMalloc(&dBuffer, bufferSize));

    warm_up_gpu();

    // Execute SpMM
    CHECK(cudaEventRecord(start));
    for (int i = 0; i < NUM_REPS; i++) {
        CHECK_CUSPARSE(
            cusparseSpMM(
                handle, 
                CUSPARSE_OPERATION_TRANSPOSE,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha, sparseMatrix, identityMatrix, &beta, resultMatrix,
                CUDA_R_32F, CUSPARSE_SPMM_COO_ALG1, dBuffer
            )
        );
    }
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    CHECK_CUSPARSE(cusparseDestroySpMat(sparseMatrix));
    CHECK_CUSPARSE(cusparseDestroyDnMat(identityMatrix));
    CHECK_CUSPARSE(cusparseDestroyDnMat(resultMatrix));
    CHECK_CUSPARSE(cusparseDestroy(handle));

    // Copy the result back to host
    CHECK(cudaMemcpy(h_result, d_result, size * sizeof(dtype), cudaMemcpyDeviceToHost));

    // Compare the results
    bool isCorrect = true;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (h_result[i + j * rows] != groundTruth[j + i * rows]) {
                isCorrect = false;
                break;
            }
        }
    }

    // Print the result
    // printf("Result:\n");
    // for (int i = 0; i < rows; i++) {
    //     for (int j = 0; j < cols; j++) {
    //         printf("%f ", h_result[i + j * rows]);
    //     }
    //     printf("\n");
    // }

    // printf("Ground Truth:\n");
    // for (int i = 0; i < rows; i++) {
    //     for (int j = 0; j < cols; j++) {
    //         printf("%f ", groundTruth[j + i * rows]);
    //     }
    //     printf("\n");
    // }
    dtype totalData = (rows * cols) * sizeof(dtype);
    totalData *= 3;

    printf("Performed cuSparse transposition on matrix %s\n", fileName.c_str());
    if (isCorrect) {
        printf("Bandwidth: %f GB/s\n", totalData * 1e-6 * NUM_REPS / milliseconds);
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


    free(row);
    free(col);
    free(val);
    free(h_identityMatrix);
    free(h_result);
}