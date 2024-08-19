#include "../include/transpose_csr_v2.cuh"
#include "../include/utilities.cuh"

__global__ void count_nnz(int nnz, int* csrColumnIndices, int* cscColPtr) {
    __shared__ int shared[BLOCK_SIZE];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    shared[threadIdx.x] = csrColumnIndices[tid];

    __syncthreads();
    if (tid < nnz) {
        int col = shared[threadIdx.x];
        atomicAdd(&cscColPtr[col + 1], 1);
    }
}

__global__ void scanLA(int* d_cscColPtr, int* d_auxBlockSums, int cols) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    __shared__ int block[BLOCK_SIZE];
    int i = bid * BLOCK_SIZE + tid;
    if (i < cols) {
        block[tid] = d_cscColPtr[i];
    } else {
        block[tid] = 0;
    }
    __syncthreads();

    for (int stride = 1; stride < BLOCK_SIZE; stride *= 2) {
        int index = (tid + 1) * stride * 2 - 1;
        if (index < BLOCK_SIZE) {
            block[index] += block[index - stride];
        }
        __syncthreads();
    }

    for (int stride = BLOCK_SIZE / 4; stride > 0; stride /= 2) {
        int index = (tid + 1) * stride * 2 - 1;
        if (index + stride < BLOCK_SIZE) {
            block[index + stride] += block[index];
        }
        __syncthreads();
    }

    if (i < cols) {
        d_cscColPtr[i] = block[tid];
    }
    if (tid == BLOCK_SIZE - 1) {
        d_auxBlockSums[bid] = block[tid];
    }

}

__global__ void uniformUpdate(int* d_cscColPtr, int* d_auxBlockSums, int cols) {
    int bid = blockIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bid > 0) {
        d_cscColPtr[idx] += d_auxBlockSums[bid - 1];
    }
}

__global__ void fillCSC_ds(int num_rows, int* col_offsets, const int* row_offsets, const int* col_indices, const dtype* values, dtype* csc_values, int* csc_row_indices) {
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

void transpose_CSR_v2(std::string fileName) {
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
    float step3_ms = 0.0; // Uniform update
    float step4_ms = 0.0; // Fill CSC

    // Warm up the GPU
    warm_up_gpu();

    /**
     * Step 1: Count the number of non-zero elements in each column
     * First I execute once the kernel
     * Then I measure the time of 101 executions
     * This is done because the execution of the kernel alters the data in the device memory
     */
    int gridSize = (nnz + BLOCK_SIZE - 1) / BLOCK_SIZE;

    count_nnz<<<gridSize, BLOCK_SIZE>>>(nnz, d_csrColumnIndices, d_cscColPtr);
    
    // Copy the result to the host
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(cscColPtrCollector, d_cscColPtr, (cols + 1) * sizeof(int), cudaMemcpyDeviceToHost));

    // Measure the time of 100 executions
    CHECK(cudaEventRecord(start, 0));
    for (int i = 0; i < NUM_REPS; i++) {
        count_nnz<<<gridSize, BLOCK_SIZE>>>(nnz, d_csrColumnIndices, d_cscColPtr);
    }
    CHECK(cudaEventRecord(stop, 0));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&step1_ms, start, stop));
    CHECK(cudaDeviceSynchronize());
    
    /**
     * Step 2: Perform exclusive scan
     * Same as before, first I execute once the kernel
     * Then I measure the time of 100 executions
     */
    gridSize = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;

    int* aux = (int*)malloc(gridSize * sizeof(int));
    int* d_auxBlockSums;
    CHECK(cudaMalloc(&d_auxBlockSums, gridSize * sizeof(int)));
    CHECK(cudaMemcpy(d_cscColPtr, cscColPtrCollector, (cols + 1) * sizeof(int), cudaMemcpyHostToDevice)); // this should "reset" the data in the device memory
    scanLA<<<gridSize, BLOCK_SIZE>>>(d_cscColPtr, d_auxBlockSums, cols + 1);

    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(aux, d_auxBlockSums, gridSize * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(cscColPtrCollector, d_cscColPtr, (cols + 1) * sizeof(int), cudaMemcpyDeviceToHost));

    // Measure the time of 100 executions
    CHECK(cudaEventRecord(start, 0));
    for (int i = 0; i < NUM_REPS; i++) {
        scanLA<<<gridSize, BLOCK_SIZE>>>(d_cscColPtr, d_auxBlockSums, cols + 1);
    }
    CHECK(cudaEventRecord(stop, 0));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&step2_ms, start, stop));
    CHECK(cudaDeviceSynchronize());

    // Compute the scan of the aux array and copy it to the device
    for (int i = 1; i < gridSize; i++) {
        aux[i] += aux[i - 1];
    }

    CHECK(cudaMemcpy(d_auxBlockSums, aux, gridSize * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_cscColPtr, cscColPtrCollector, (cols + 1) * sizeof(int), cudaMemcpyHostToDevice));

    uniformUpdate<<<gridSize, BLOCK_SIZE>>>(d_cscColPtr, d_auxBlockSums, cols + 1);

    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(cscColPtrCollector, d_cscColPtr, (cols + 1) * sizeof(int), cudaMemcpyDeviceToHost));

    // Measure the time of 100 executions
    CHECK(cudaEventRecord(start, 0));
    for (int i = 0; i < NUM_REPS; i++) {
        uniformUpdate<<<gridSize, BLOCK_SIZE>>>(d_cscColPtr, d_auxBlockSums, cols + 1);
    }
    CHECK(cudaEventRecord(stop, 0));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&step3_ms, start, stop));
    CHECK(cudaDeviceSynchronize());

    /**
     * Step 3: Compute the cscRowIdx and cscVal
     */
    CHECK(cudaMemcpy(d_cscColPtr, cscColPtrCollector, (cols + 1) * sizeof(int), cudaMemcpyHostToDevice));
    gridSize = (rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
    fillCSC_ds<<<gridSize, BLOCK_SIZE>>>(
        rows, d_cscColPtr, d_csrRowPointers, d_csrColumnIndices, d_csrValues, d_cscVal, d_cscRowIdx
    );

    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(cscRowIdx, d_cscRowIdx, nnz * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(cscVal, d_cscVal, nnz * sizeof(dtype), cudaMemcpyDeviceToHost));

    // Measure the time of 100 executions
    CHECK(cudaEventRecord(start, 0));
    for (int i = 0; i < NUM_REPS; i++) {
        fillCSC_ds<<<gridSize, BLOCK_SIZE>>>(
            rows, d_cscColPtr, d_csrRowPointers, d_csrColumnIndices, d_csrValues, d_cscVal, d_cscRowIdx
        );
    }
    CHECK(cudaEventRecord(stop, 0));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&step4_ms, start, stop));
    CHECK(cudaDeviceSynchronize());


    // Check if the result is correct
    float milliseconds = step1_ms + step2_ms + step3_ms + step4_ms;
    dtype* groundTruth = generateGroundTruthFromMTX(fileName);

    // Calculate the total data accessed
    // Step 1: Count
    int copy_step_1 = 2 * nnz * sizeof(int); // R/W csrColumnIndices in shared memory
    int count_step_1 = 3 * nnz * sizeof(int); // R nnz elements from shared memory, atomicAdd performs R/W on cscColPtr nnz times

    // Step 2: Scan
    int scan_step_2 = 2 * (cols + 1) * sizeof(int); // R/W cscColPtr
    int write_step_2 = gridSize * sizeof(int); // W gridSize times on auxBlockSums (every last threadof each block writes to it)

    // Step 3: Uniform update
    int read_aux_step_3 = gridSize * sizeof(int); // R auxBlockSums
    int uniform_update_step_3 = 2 * (cols + 1) * sizeof(int); // R/W cscColPtr

    // Step 4: Fill CSC
    int index_step_4 = 2 * nnz * sizeof(int); // R/W cscColPtr
    int fill_step_4 = 2 * nnz * sizeof(int) + 2 * nnz * sizeof(dtype); // R/W row_offsets + R/W values

    double total_data = copy_step_1 + count_step_1 + scan_step_2 + write_step_2 + read_aux_step_3 + uniform_update_step_3 + index_step_4 + fill_step_4;


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
    CHECK(cudaFree(d_auxBlockSums));

    free(csrRowPointers);
    free(csrColumnIndices);
    free(csrValues);
    free(cscColPtr);
    free(cscRowIdx);
    free(cscVal);
    free(cscColPtrCollector);
    free(groundTruth);
    free(aux);
}
