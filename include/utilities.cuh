#ifndef _UTILITIES_H_
#define _UTILITIES_H_

#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <string>
#include <vector>

#define NUM_REPS 101
#define dtype double
#define BLOCK_SIZE 256
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

void warm_up_gpu();

/**
 * @brief Parse a CSV file to COO format
 * @param m number of rows
 * @param n number of columns
 * @param nnz number of non-zero elements
 * @param rows row indices
 * @param cols column indices
 * @param values values
 * @param filename name of the CSV file
 */
void parseCsvToCoo(int &m, int &n, int &nnz, int *&rows, int *&cols, dtype *&values, std::string filename);

/**
 * @brief Parse a CSV file to CSR format
 * @param m number of rows
 * @param n number of columns
 * @param nnz number of non-zero elements
 * @param rows row indices pointer
 * @param cols column indices
 * @param values values
 * @param filename name of the CSV file
 */
void parseCsvToCsr(int &m, int &n, int &nnz, int *&rows, int *&cols, dtype *&values, std::string filename);

/**
 * @brief Generate ground truth for COO format
 * @param m number of rows
 * @param n number of columns
 * @param nnz number of non-zero elements
 * @param rows row indices
 * @param cols column indices
 * @param values values
 * @return ground truth in dense format
 * 
 * @note The generated ground truth is already transposed to ease comparison with the output of the algorithm
 */
dtype* generateCOOGroundTruth(int m, int n, int nnz, int *rows, int *cols, dtype *values);

dtype* generateGroundTruthFromMTX(std::string filename);


bool checkResult(dtype* groundTruth, int* transposedRow, int* transposedCol, dtype* vals, int nnz, int sideLength);
bool checkResultCSR(dtype* groundTruth, int* cscColPtr, int* cscRowIdx, dtype* cscVal, int rows, int cols);

void cscToCoo(int m, int n, int nnz, int *rows, int *cols, float *values, int *&cooRows, int *&cooCols, float *&cooVals);

#endif