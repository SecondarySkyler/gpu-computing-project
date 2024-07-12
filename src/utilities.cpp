#include "../include/utilities.hpp"

void parseCsvToCoo(int &m, int &n, int &nnz, int *&rows, int *&cols, float *&values, std::string filename) {
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

float* generateCOOGroundTruth(int m, int n, int nnz, int *rows, int *cols, float *values) {
    float *groundTruth = new float[m * n];
    for (int i = 0; i < m * n; i++) {
        groundTruth[i] = 0.0;
    }

    for (int i = 0; i < nnz; i++) {
        groundTruth[cols[i] * n + rows[i]] = values[i];
    }

    return groundTruth;
}

bool checkResult(float* groundTruth, int* transposedRow, int* transposedCol, float* vals, int nnz, int sideLength) {
    for (int i = 0; i < nnz; i++) {
        if (groundTruth[transposedRow[i] * sideLength + transposedCol[i]] != vals[i]) {
            return false;
        }
    }

    return true;
}