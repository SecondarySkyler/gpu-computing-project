#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>

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
void parseCsvToCoo(int &m, int &n, int &nnz, int *&rows, int *&cols, float *&values, char* filename);

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
void parseCsvToCsr(int &m, int &n, int &nnz, int *&rows, int *&cols, float *&values, char* filename);