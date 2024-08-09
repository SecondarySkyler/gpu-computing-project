#include "../include/utilities.cuh"

__global__ void warmup_gpu() {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float ia, ib;
    ia = ib = 0.0;
    ib += ia + tid;
}

void warm_up_gpu() {
    warmup_gpu<<<1, 1>>>();
}

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

void parseCsvToCsr(int &m, int &n, int &nnz, int *&rows, int *&cols, dtype *&values, std::string filename) {
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

bool checkResultCSR(dtype* groundTruth, int* cscColPtr, int* cscRowIdx, dtype* cscVal, int rows, int cols) {
    int totalElements = rows * cols;
    dtype* cscToDense = (dtype*)malloc(totalElements * sizeof(dtype));

    for (int i = 0; i < totalElements; i++) {
        cscToDense[i] = 0.0;
    }

    for (int col = 0; col < cols; ++col) {
        // Get the start and end indices for the current column
        int start = cscColPtr[col];
        int end = cscColPtr[col + 1];

        // Fill the dense matrix with the values from the CSC format
        for (int idx = start; idx < end; ++idx) {
            int row = cscRowIdx[idx];
            cscToDense[col * rows + row] = cscVal[idx];
        }
    }

    // check if the transpose is correct
    bool conversionSuccessful = true;
    for (int i = 0; i < totalElements; i++) {
        if (groundTruth[i] != cscToDense[i]) {
            conversionSuccessful = false;
            break;
        }
    }

    free(cscToDense);
    return conversionSuccessful;
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

dtype* generateGroundTruthFromMTX(std::string filename) {
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

    dtype *groundTruth = new dtype[rows * cols];
    std::fill(groundTruth, groundTruth + rows * cols, 0.0);

    for (int i = 0; i < nnz; i++) {
        int row, col;
        dtype val;
        file >> row >> col >> val;
        groundTruth[(col - 1) * rows + (row - 1)] = val;
    }

    file.close();
    return groundTruth;
}