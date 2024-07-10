#include "../include/utilities.hpp"
#include "../include/matio.h"

COO readMatrixCOO(std::string filename) {
    COO result;

    std::string filePath = "./test_matrices/" + filename + ".mtx";
    std::ifstream file(filePath);
    if (!file.is_open()) {
        std::cerr << "Error opening file" << std::endl;
    }

    // ignore comments
    while (file.peek() == '%')
        file.ignore(2048, '\n');

    // read the matrix size and number of non-zero elements
    int numRows, numCols, numNnz;
    file >> numRows >> numCols >> numNnz;
    result.totalRows = numRows;
    result.totalCols = numCols;
    result.totalNnz = numNnz;

    // initialize the matrix
    result.rows = new int[numNnz];
    result.cols = new int[numNnz];
    result.values = new float[numNnz];

    // read the matrix elements
    for (int line = 0; line < numNnz; line++) {
        float value;
        int currentRow, currentCol;
        file >> currentRow >> currentCol >> value;
        result.rows[line] = currentRow;
        result.cols[line] = currentCol;
        result.values[line] = value;
    }

    file.close();
    return result;
}

int findIndex(float* array, int size, int value) {
    for (int i = 0; i < size; i++) {
        if (array[i] == value) {
            return i;
        }
    }
    return -1;
}

void mtxToCsr(std::string filename, int &m, int &n, int &nnz,
    int *&csrRowPtr, int *&csrColIdx, float* &csrVal) {
    
    std::string filePath = "./test_matrices/" + filename + ".mtx";
    std::ifstream file(filePath);
    if (!file.is_open()) {
        std::cerr << "Error opening file" << std::endl;
    }

    // ignore comments
    while (file.peek() == '%')
        file.ignore(2048, '\n');

    // read the matrix size and number of non-zero elements
    file >> m >> n >> nnz;

    // initialize row ptrs, column indices and values
    csrRowPtr = new int[m + 1];
    csrColIdx = new int[nnz];
    csrVal = new float[nnz];
    int* auxRowIdx = new int[nnz];


    // read the matrix elements, for the moment we ignore the row pointers
    for (int line = 0; line < nnz; line++) {
        float value;
        int currentRow, currentCol;
        file >> currentRow >> currentCol >> value;
        auxRowIdx[line] = currentRow;
        csrColIdx[line] = currentCol;
        csrVal[line] = value;
    }

    // create dense matrix
    float* denseMatrix = (float*) calloc(m * n, sizeof(float));

    // fill dense matrix
    for (int i = 0; i < nnz; i++) {
        denseMatrix[auxRowIdx[i] * n + csrColIdx[i]] = csrVal[i];
    }

    // scan dense matrix to fill csrRowPtr
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (denseMatrix[i * n + j] != 0) {
                csrRowPtr[i] = findIndex(csrVal, nnz, denseMatrix[i * n + j]);
                break;
            }
        }
    }
    csrRowPtr[m] = nnz;

    file.close();
}