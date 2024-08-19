#include "../include/transpose_csr.cuh"
#include "../include/utilities.cuh"

__global__ void count_v3(int nnz, int* csrColumnIndices, int* cscColPtr) {
    __shared__ int shared[BLOCK_SIZE];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    shared[threadIdx.x] = csrColumnIndices[tid];

    __syncthreads();
    if (tid < nnz) {
        int col = shared[threadIdx.x];
        atomicAdd(&cscColPtr[col + 1], 1);
    }
}

__global__ void scan_v2(int n, int *d_cscColPtr) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid == 0) {
        d_cscColPtr[0] = 0;
        for (int i = 1; i < n; i++) {
            d_cscColPtr[i] += d_cscColPtr[i - 1]; 
        }
    }
}

__global__ void fillCSC(int num_rows, int* col_offsets, const int* row_offsets, const int* col_indices, const dtype* values, dtype* csc_values, int* csc_row_indices) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows) {
        for (int j = row_offsets[row]; j < row_offsets[row + 1]; j++) {
            int col = col_indices[j];
            int index = atomicAdd(&col_offsets[col], 1);
            csc_values[index] = values[j];
            csc_row_indices[index] = row;
        }
    }
}

void transpose_CSR(std::string fileName) {
    int rows, cols, nnz;
    int* csrRowPointers, *csrColumnIndices;
    dtype* csrValues;

    parseCsvToCsr(rows, cols, nnz, csrRowPointers, csrColumnIndices, csrValues, fileName);

    // Declare CSC matrix variables
    int* cscColPtr = (int*)malloc((cols + 1) * sizeof(int));
    int* cscRowIdx = (int*)malloc(nnz * sizeof(int));
    dtype* cscVal = (dtype*)malloc(nnz * sizeof(dtype));

    int* cscColPtrCollector = (int*)malloc((cols + 1) * sizeof(int)); // used to collect the results from the device

    // Initialize cscColPtr with zeros
    for (int i = 0; i < cols + 1; i++) {
        cscColPtr[i] = 0;
        cscColPtrCollector[i] = 0;
    }

    // -- Device memory allocation --
    int* d_csrRowPointers;
    int* d_csrColumnIndices;
    dtype* d_csrValues;
    int* d_cscColPtr;
    int* d_cscRowIdx;
    dtype* d_cscVal;

    // Allocate device memory
    CHECK(cudaMalloc(&d_csrRowPointers, (rows + 1) * sizeof(int)));
    CHECK(cudaMalloc(&d_csrColumnIndices, nnz * sizeof(int)));
    CHECK(cudaMalloc(&d_csrValues, nnz * sizeof(dtype)));
    CHECK(cudaMalloc(&d_cscColPtr, (cols + 1) * sizeof(int)));
    CHECK(cudaMalloc(&d_cscRowIdx, nnz * sizeof(int)));
    CHECK(cudaMalloc(&d_cscVal, nnz * sizeof(dtype)));

    // Copy data to device memory
    CHECK(cudaMemcpy(d_csrRowPointers, csrRowPointers, (rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_csrColumnIndices, csrColumnIndices, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_csrValues, csrValues, nnz * sizeof(dtype), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_cscColPtr, cscColPtr, (cols + 1) * sizeof(int), cudaMemcpyHostToDevice));

    // Create cuda events to measure the time
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    float step1_ms = 0.0; // Count
    float step2_ms = 0.0; // Scan
    float step3_ms = 0.0; // Fill CSC

    // Warm up the GPU
    warm_up_gpu();

    /**
     * Step 1: Count the number of non-zero elements in each column
     * First I execute once the kernel
     * Then I measure the time of 101 executions
     * This is done because the execution of the kernel alters the data in the device memory
     */
    int gridSize = (nnz + BLOCK_SIZE - 1) / BLOCK_SIZE;

    count_v3<<<gridSize, BLOCK_SIZE>>>(nnz, d_csrColumnIndices, d_cscColPtr);

    // Copy the result to the host
    CHECK(cudaMemcpy(cscColPtrCollector, d_cscColPtr, (cols + 1) * sizeof(int), cudaMemcpyDeviceToHost));

    // Measure the time of 101 executions
    CHECK(cudaEventRecord(start, 0));
    for (int i = 0; i < NUM_REPS; i++) {
        count_v3<<<gridSize, BLOCK_SIZE>>>(nnz, d_csrColumnIndices, d_cscColPtr);
    }
    CHECK(cudaEventRecord(stop, 0);) 
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&step1_ms, start, stop));

    CHECK(cudaDeviceSynchronize());

    /**
     * Step 2: Perform exclusive scan
     * Same as before, first I execute once the kernel
     * Then I measure the time of 101 executions
     */
    cudaMemcpy(d_cscColPtr, cscColPtrCollector, (cols + 1) * sizeof(int), cudaMemcpyHostToDevice); // this should "reset" the data in the device memory
    scan_v2<<<1, 1>>>(cols + 1, d_cscColPtr);

    CHECK(cudaMemcpy(cscColPtrCollector, d_cscColPtr, (cols + 1) * sizeof(int), cudaMemcpyDeviceToHost));

    CHECK(cudaEventRecord(start, 0));
    for (int i = 0; i < NUM_REPS; i++) {
        scan_v2<<<1, 1>>>(cols + 1, d_cscColPtr);
    }
    CHECK(cudaEventRecord(stop, 0));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&step2_ms, start, stop));

    CHECK(cudaDeviceSynchronize());

    /**
     * Step 3: Compute the cscRowIdx and cscVal
     */
    CHECK(cudaMemcpy(d_cscColPtr, cscColPtrCollector, (cols + 1) * sizeof(int), cudaMemcpyHostToDevice));
    gridSize = (rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
    fillCSC<<<gridSize, BLOCK_SIZE>>>(
        rows, d_cscColPtr, d_csrRowPointers, d_csrColumnIndices, d_csrValues, d_cscVal, d_cscRowIdx
    );

    CHECK(cudaMemcpy(cscRowIdx, d_cscRowIdx, nnz * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(cscVal, d_cscVal, nnz * sizeof(dtype), cudaMemcpyDeviceToHost));

    CHECK(cudaEventRecord(start, 0));
    for (int i = 0; i < NUM_REPS; i++) {
        fillCSC<<<gridSize, BLOCK_SIZE>>>(
            rows, d_cscColPtr, d_csrRowPointers, d_csrColumnIndices, d_csrValues, d_cscVal, d_cscRowIdx
        );
    }
    CHECK(cudaEventRecord(stop, 0));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&step3_ms, start, stop));

    CHECK(cudaDeviceSynchronize());

    // Check if the result is correct
    float milliseconds = step1_ms + step2_ms + step3_ms;
    dtype* groundTruth = generateGroundTruthFromMTX(fileName);

    // Calculate the total data accessed
    // Step 1: Count
    int copy_step_1 = 2 * nnz * sizeof(int); // R/W csrColumnIndices in shared memory
    int count_step_1 = 3 * nnz * sizeof(int); // R nnz elements from shared memory, atomicAdd performs R/W on cscColPtr nnz times

    // Step 2: Scan
    int scan_step_2 = 2 * (cols + 1) * sizeof(int); // R/W cscColPtr

    // Step 3: Fill CSC
    int index_step_3 = 2 * nnz * sizeof(int); // R/W cscColPtr
    int fill_step_3 = 2 * nnz * sizeof(int) + 2 * nnz * sizeof(dtype); // R/W row_offsets + R/W values

    double total_data = copy_step_1 + count_step_1 + scan_step_2 + index_step_3 + fill_step_3;


    printf("Performed CSR transposition on matrix %s\n", fileName.c_str());
    if (checkResultCSR(groundTruth, cscColPtrCollector, cscRowIdx, cscVal, rows, cols)) {
        printf("Bandwidth: %f GB/s\n", total_data * 1e-6 * NUM_REPS / milliseconds);
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

    // Free memory
    CHECK(cudaFree(d_csrRowPointers));
    CHECK(cudaFree(d_csrColumnIndices));
    CHECK(cudaFree(d_csrValues));
    CHECK(cudaFree(d_cscColPtr));
    CHECK(cudaFree(d_cscRowIdx));
    CHECK(cudaFree(d_cscVal));

    free(csrRowPointers);
    free(csrColumnIndices);
    free(csrValues);
    free(cscColPtr);
    free(cscRowIdx);
    free(cscVal);
    free(cscColPtrCollector);
    free(groundTruth);

}