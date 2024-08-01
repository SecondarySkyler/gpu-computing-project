#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>

#define dtype double

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
void parseCsvToCsr(int &m, int &n, int &nnz, int *&rows, int *&cols, float *&values, std::string filename);

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

float* generateGroundTruthFromMTX(std::string filename);


bool checkResult(dtype* groundTruth, int* transposedRow, int* transposedCol, dtype* vals, int nnz, int sideLength);
bool checkResultCSR(float* groundTruth, int* transposedRow, int* transposedCol, float* vals, int rows, int cols);

void cscToCoo(int m, int n, int nnz, int *rows, int *cols, float *values, int *&cooRows, int *&cooCols, float *&cooVals);