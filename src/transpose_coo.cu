#include "../include/transpose_coo.cuh"
#include "../include/utilities.cuh"

/**
 * Function to transpose a matrix in COO format
 * @param rows: array containing the row indices of the non-zero elements
 * @param cols: array containing the column indices of the non-zero elements
 * @param nnz: number of non-zero elements in the matrix
 */
__global__ void simple_COO_transpose(int* rows, int* cols, int nnz) {
    __shared__ int sharedRows[BLOCK_SIZE];
    __shared__ int sharedCols[BLOCK_SIZE];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // load data into shared memory and transpose
    if (tid < nnz) {
        sharedRows[threadIdx.x] = rows[tid];
        sharedCols[threadIdx.x] = cols[tid];
    
        __syncthreads();

        int temp = sharedRows[threadIdx.x];
        sharedRows[threadIdx.x] = sharedCols[threadIdx.x];
        sharedCols[threadIdx.x] = temp;

        __syncthreads();

        // store the result back to the global memory
        rows[tid] = sharedRows[threadIdx.x];
        cols[tid] = sharedCols[threadIdx.x];
    }

}


void transpose_COO(std::string fileName) {
  int rows, cols, nnz;
  int* row, *col;
  dtype* val;

  parseCsvToCoo(rows, cols, nnz, row, col, val, fileName);
  dtype* groundTruth = generateCOOGroundTruth(rows, cols, nnz, row, col, val);


  // device data structures
  int* d_row, *d_col;
  
  // allocate memory on the device
  CHECK(cudaMalloc(&d_row, nnz * sizeof(int)));
  CHECK(cudaMalloc(&d_col, nnz * sizeof(int)));

  // copy the matrix to the device
  CHECK(cudaMemcpy(d_row, row, nnz * sizeof(int), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_col, col, nnz * sizeof(int), cudaMemcpyHostToDevice));

  // create the event to measure the time
  cudaEvent_t start, stop;
  CHECK(cudaEventCreate(&start));
  CHECK(cudaEventCreate(&stop));
  float milliseconds;

  // define blocks number
  int grid_size = (nnz + BLOCK_SIZE - 1) / BLOCK_SIZE;

  // warm up the GPU
  warm_up_gpu();

  // transpose the matrix
  cudaEventRecord(start, 0);
  for (int i = 0; i < NUM_REPS; i++)
    simple_COO_transpose<<<grid_size, BLOCK_SIZE>>>(d_row, d_col, nnz);
  
  cudaEventRecord(stop, 0);
  CHECK(cudaEventSynchronize(stop));
  CHECK(cudaEventElapsedTime(&milliseconds, start, stop));


  CHECK(cudaDeviceSynchronize());

  // copy the result back to the host
  CHECK(cudaMemcpy(row, d_row, nnz * sizeof(int), cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(col, d_col, nnz * sizeof(int), cudaMemcpyDeviceToHost));


  // check if the result is correct
  if (checkResult(groundTruth, row, col, val, nnz, cols)){
    printf("Performed COO transposition on matrix %s\n", fileName.c_str());
    printf("Bandwidth: %f GB/s\n", 4 * nnz * sizeof(int) * 1e-6 * NUM_REPS / milliseconds);
    printf("Status: Correct\n");
    printf("--------------------------------\n");
  } else {
    printf("The result is incorrect\n");
    printf("--------------------------------\n");
  }

  // free memory
  cudaFree(d_row);
  cudaFree(d_col);
  free(row);
  free(col);
  free(val);
}