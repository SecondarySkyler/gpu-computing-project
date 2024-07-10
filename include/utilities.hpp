#pragma once
#include <iostream>
#include <fstream>
#include <vector>

struct COO {
    int totalRows;
    int totalCols;
    int totalNnz;
    int* rows;
    int* cols;
    float* values;
};


/**
 * Read a matrix in COO format from a file
 * @param filename the name of the .mtx file containing the matrix description
 * 
 * @return COO struct containing the matrix in COO format
 */
COO readMatrixCOO(std::string filename);

/**
 * Read a matrix in CSR format from a file
 * @param filename the name of the .mtx file containing the matrix description
 * @param m the number of rows of the matrix
 * @param n the number of columns of the matrix
 * @param nnz the number of non-zero elements of the matrix
 * @param csrRowPtr the row pointer array of the matrix
 * @param csrColIdx the column index array of the matrix
 * @param csrVal the value array of the matrix
 */
void mtxToCsr(
    std::string filename,
    int &m,
    int &n,
    int &nnz,
    int *&csrRowPtr,
    int *&csrColIdx,
    float *&csrVal
);
