#include "../include/utilities.hpp"

void parseCsvToCoo(int &m, int &n, int &nnz, int *&rows, int *&cols, dtype *&values, std::string filename) {
    std::string filePath = "./test_matrices/coo/" + std::string(filename) + ".csv";
    std::ifstream file(filePath);

    if (!file.is_open()) {
        std::cerr << "Error opening file" << std::endl;
    }

    // read the matrix size and number of non-zero elements
    std::string matrixInfo;
    std::getline(file, matrixInfo);
    std::stringstream ss(matrixInfo);

    std::vector<std::string> tokens;
    std::string token;

    while (std::getline(ss, token, ',')) {
        tokens.push_back(token);
    }

    m = std::stoi(tokens[0]);
    n = std::stoi(tokens[1]);
    nnz = std::stoi(tokens[2]);

    // parse the row indices
    rows = new int[nnz];
    std::string rowIndices;
    std::getline(file, rowIndices);
    std::stringstream ssRowIndices(rowIndices);

    std::vector<std::string> rowTokens;
    std::string rowToken;

    while (std::getline(ssRowIndices, rowToken, ',')) {
        rowTokens.push_back(rowToken);
    }

    for (int i = 0; i < nnz; i++) {
        rows[i] = std::stoi(rowTokens[i]);
    }

    // parse the column indices
    cols = new int[nnz];
    std::string colIndices;
    std::getline(file, colIndices);
    std::stringstream ssColIndices(colIndices);

    std::vector<std::string> colTokens;
    std::string colToken;

    while (std::getline(ssColIndices, colToken, ',')) {
        colTokens.push_back(colToken);
    }

    for (int i = 0; i < nnz; i++) {
        cols[i] = std::stoi(colTokens[i]);
    }

    // parse the values
    values = new dtype[nnz];
    std::string valIndices;
    std::getline(file, valIndices);
    std::stringstream ssValIndices(valIndices);

    std::vector<std::string> valTokens;
    std::string valToken;

    while (std::getline(ssValIndices, valToken, ',')) {
        valTokens.push_back(valToken);
    }

    for (int i = 0; i < nnz; i++) {
        values[i] = std::stod(valTokens[i]);
    }

    file.close();
}

void parseCsvToCsr(int &m, int &n, int &nnz, int *&rows, int *&cols, float *&values, std::string filename) {
    std::string filePath = "./test_matrices/csr/" + std::string(filename) + ".csv";
    std::ifstream file(filePath);

    if (!file.is_open()) {
        std::cerr << "Error opening file" << std::endl;
    }

    // read the matrix size and number of non-zero elements
    std::string matrixInfo;
    std::getline(file, matrixInfo);
    std::stringstream ss(matrixInfo);

    std::vector<std::string> tokens;
    std::string token;

    while (std::getline(ss, token, ',')) {
        tokens.push_back(token);
    }

    m = std::stoi(tokens[0]);
    n = std::stoi(tokens[1]);
    nnz = std::stoi(tokens[2]);

    // parse the row indices
    rows = new int[m + 1];
    std::string rowIndices;
    std::getline(file, rowIndices);
    std::stringstream ssRowIndices(rowIndices);

    std::vector<std::string> rowTokens;
    std::string rowToken;

    while (std::getline(ssRowIndices, rowToken, ',')) {
        rowTokens.push_back(rowToken);
    }

    for (int i = 0; i < m + 1; i++) {
        rows[i] = std::stoi(rowTokens[i]);
    }

    // parse the column indices
    cols = new int[nnz];
    std::string colIndices;
    std::getline(file, colIndices);
    std::stringstream ssColIndices(colIndices);

    std::vector<std::string> colTokens;
    std::string colToken;

    while (std::getline(ssColIndices, colToken, ',')) {
        colTokens.push_back(colToken);
    }

    for (int i = 0; i < nnz; i++) {
        cols[i] = std::stoi(colTokens[i]);
    }

    // parse the values
    values = new float[nnz];
    std::string valIndices;
    std::getline(file, valIndices);
    std::stringstream ssValIndices(valIndices);

    std::vector<std::string> valTokens;
    std::string valToken;

    while (std::getline(ssValIndices, valToken, ',')) {
        valTokens.push_back(valToken);
    }

    for (int i = 0; i < nnz; i++) {
        values[i] = std::stof(valTokens[i]);
    }

    file.close();
}

dtype* generateCOOGroundTruth(int m, int n, int nnz, int *rows, int *cols, dtype *values) {
    dtype *groundTruth = new dtype[m * n];
    for (int i = 0; i < m * n; i++) {
        groundTruth[i] = 0.0;
    }

    for (int i = 0; i < nnz; i++) {
        // printf();
        groundTruth[cols[i] * m + rows[i]] = values[i];
    }

    return groundTruth;
}

bool checkResult(dtype* groundTruth, int* transposedRow, int* transposedCol, dtype* vals, int nnz, int sideLength) {
    for (int i = 0; i < nnz; i++) {
        if (groundTruth[transposedRow[i] * sideLength + transposedCol[i]] != vals[i]) {
            printf("Mismatch at %d, %d\n", transposedRow[i], transposedCol[i]);
            printf("Expected: %f, Got: %f\n", groundTruth[transposedRow[i] * sideLength + transposedCol[i]], vals[i]);
            return false;
        }
    }

    return true;
}

bool checkResultCSR(float* groundTruth, int* transposedRow, int* transposedCol, float* vals, int rows, int cols) {
    printf("rows: %d, cols: %d\n", rows, cols);
    // first we need to create the dense matrix from the csc format
    std::vector<float> cscToDense(rows * cols, 0.0);
    for (int i = 0; i < cols; ++i) {
        // printf("Processing column %d\n", i);
        int start = transposedCol[i];
        // printf("Start: %d\n", start);
        int end = transposedCol[i + 1];
        // printf("End: %d\n", end);

        for (int j = start; j < end; ++j) {
            int row = transposedRow[j];
            cscToDense[i * rows + row] = vals[j];
        }
    }
    // printf("Dense matrix created\n");
    // for (float val : cscToDense) {
    //     printf("%.f ", val);
    // }
    // printf("\n");

    // DEBUG
    int totalElements = rows * cols;

    // for (int i = 0; i < totalElements; ++i) {
    //     printf("%.f ", groundTruth[i]);
    // }
    // printf("\n");

    // for (int i = 0; i < totalElements; ++i) {
    //     printf("%.f ", cscToDense[i]);
    // }
    // printf("\n");
    



    // now we compare the dense matrix with the ground truth
    // for (int i = 0; i < totalElements; ++i) {
    //     if (groundTruth[i] != cscToDense[i]) {
    //         printf("Mismatch at %d, %d\n", i % rows, i / rows);
    //         printf("Expected: %f, Got: %f\n", groundTruth[i], cscToDense[i]);
    //         return false;
    //     }
    // }

    return true;
}

void cscToCoo(int m, int n, int nnz, int *rows, int *cols, float *values, int *&cooRows, int *&cooCols, float *&cooVals) {
    cooRows = new int[nnz];
    cooCols = new int[nnz];
    cooVals = new float[nnz];

    int k = 0;
    for (int i = 0; i < n; i++) {
        for (int j = cols[i]; j < cols[i + 1]; j++) {
            cooRows[k] = rows[j];
            cooCols[k] = i;
            cooVals[k] = values[j];
            k++;
        }
    }

    
}

float* generateGroundTruthFromMTX(std::string filename) {
    std::string filePath = "./test_matrices/" + filename + ".mtx";
    std::ifstream file(filePath);

    if (!file.is_open()) {
        std::cerr << "Error opening file" << std::endl;
    }

    // skip the comments
    while (file.peek() == '%') {
        file.ignore(2048, '\n');
    }

    int rows, cols, nnz;
    file >> rows >> cols >> nnz;

    float *groundTruth = new float[rows * cols];
    std::fill(groundTruth, groundTruth + rows * cols, 0.0);

    for (int i = 0; i < nnz; i++) {
        int row, col;
        float val;
        file >> row >> col >> val;
        groundTruth[(col - 1) * rows + (row - 1)] = val;
    }

    file.close();
    return groundTruth;
}