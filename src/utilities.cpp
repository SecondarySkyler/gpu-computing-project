#include "../include/utilities.hpp"

void readMatrixCOO(int* &rows, int* &cols, float* &nnz) {
    std::ifstream file("./test_matrices/494_bus.mtx");
    if (!file.is_open()) {
        std::cerr << "Error opening file" << std::endl;
        return;
    }

    // ignore comments
    while (file.peek() == '%')
        file.ignore(2048, '\n');

    // read the matrix size and number of non-zero elements
    int numRows, numCols, numNnz;
    file >> numRows >> numCols >> numNnz;

    std::cout << "Matrix size: " << numRows << "x" << numCols << std::endl;
    std::cout << "Number of non-zero elements: " << numNnz << std::endl;

    // initialize the matrix
    rows = new int[numNnz];
    cols = new int[numNnz];
    nnz = new float[numNnz];

    // read the matrix elements
    for (int line = 0; line < numNnz; line++) {
        float value;
        int currentRow, currentCol;
        file >> currentRow >> currentCol >> value;
        rows[line] = currentRow;
        cols[line] = currentCol;
        nnz[line] = value;
    }

    file.close();
}