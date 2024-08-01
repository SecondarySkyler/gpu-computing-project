#include "../include/utilities.hpp"
#include <array>

#define NUM_REPS 101
#define BLOCK_SIZE 256
#define dtype double
#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
    }                                                                          \
}
#define CHECK_CUSPARSE(call)                                                   \
{                                                                              \
    cusparseStatus_t err;                                                      \
    if ((err = (call)) != CUSPARSE_STATUS_SUCCESS)                             \
    {                                                                          \
        fprintf(stderr, "Got error %d at %s:%d\n", err, __FILE__, __LINE__);   \
        cudaError_t cuda_err = cudaGetLastError();                             \
        if (cuda_err != cudaSuccess)                                           \
        {                                                                      \
            fprintf(stderr, "  CUDA error \"%s\" also detected\n",             \
                    cudaGetErrorString(cuda_err));                             \
        }                                                                      \
        exit(1);                                                               \
    }                                                                          \
}

// Matrix names
std::array<std::string, 10> matrixNames = {
    "arc130",
    "fd15",
    "bips07_1998",
    "bayer10",
    "piston",
    "cage10",
    "Ill_Stokes",
    "msc10848",
    "appu",
    "TSOPF_RS_b300_c2"
};

/**
 * Function to warm up the GPU
 */
__global__ void warm_up_gpu(){
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float ia, ib;
  ia = ib = 0.0f;
  ib += ia + tid; 
}

/**
 * Kernel to transpose a matrix in COO format
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

/**
* Function to transpose a matrix in COO format
* @param fileName: name of the file containing the matrix
*/
void transpose_COO(std::string fileName) {
    // parse the matrix
    int rows, cols, nnz;
    int* row, *col;
    dtype* val;
    parseCsvToCoo(rows, cols, nnz, row, col, val, fileName);

    // generate the ground truth to check the correctness of the result
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

    // define the number of blocks
    int grid_size = (nnz + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // warm up the GPU
    warm_up_gpu<<<1, 1>>>();

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

    // free the memory
    cudaFree(d_row);
    cudaFree(d_col);
    free(row);
    free(col);
    free(val);
}

int main(int argc, char const *argv[]) {
    
    if (argc != 2) {
    printf("Usage: ./main <number | all>\n");
    printf("number: 0 - 9, to execute the transposition of a specific matrix\n");
    printf("all: to execute the transposition of all matrices\n");
    exit(1);
    } else {
        std::string arg = argv[1];
        if (arg == "all") {
            for (std::string matrixName : matrixNames) {
                transpose_COO(matrixName);
            }
        } else {
            int matrixNumber = std::stoi(arg);
            if (matrixNumber < 0 || matrixNumber > 10) {
                printf("Invalid matrix number\n");
                exit(1);
            } else {
                std::string matrixName = matrixNames[matrixNumber];
                transpose_COO(matrixName);
            }
        }
    }
    return 0;
}
