#pragma once
#include <iostream>
#include <fstream>

/**
 * Read a matrix in COO format from a file
 * 
 * @param rows the array that will store the row indices
 * @param cols the array that will store the column indices
 * @param nnz the array that will store the non-zero elements
 */
void readMatrixCOO(int* &rows, int* &cols, float* &nnz);
