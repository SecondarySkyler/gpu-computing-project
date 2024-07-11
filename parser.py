import numpy as np
from scipy.sparse import coo_matrix
import sys

def read_mtx_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Skipping the header and comments
    data_lines = []
    for line in lines:
        if line.startswith('%'):
            continue
        else:
            data_lines.append(line)
    
    # Extracting the matrix dimensions
    nrows, ncols, nnz = map(int, data_lines[0].split())
    data_lines = data_lines[1:]

    # Extracting the matrix entries
    row = []
    col = []
    data = []

    for line in data_lines:
        entries = line.split()
        r, c, v = map(float, entries)
        row.append(int(r) - 1) # Converting to 0-based indexing
        col.append(int(c) - 1) # Converting to 0-based indexing
        data.append(v)
    
    # Creating the COO matrix
    m = coo_matrix((data, (row, col)), shape=(nrows, ncols))
    return m


def write_coo_to_csv(matrix, filename):
    filepath = folder + 'coo/' + filename + '.csv'
    file = open(filepath, 'x')
    file.write(f'{matrix.get_shape()[0]},{matrix.get_shape()[1]},{matrix.getnnz()}\n')
    rows = ''
    cols = ''
    data = ''
    for i in range(matrix.getnnz()):
        rows += f'{matrix.row[i]},'
        cols += f'{matrix.col[i]},'
        data += f'{matrix.data[i]},'

    file.write(rows + '\n')
    file.write(cols + '\n')
    file.write(data + '\n')
    file.close()

def write_csr_to_csv(matrix, filename):
    filepath = folder + 'csr/' + filename + '.csv'
    file = open(filepath, 'x')
    file.write(f'{matrix.get_shape()[0]},{matrix.get_shape()[1]},{matrix.getnnz()}\n')
    rows = ''
    cols = ''
    data = ''
    for i in range(matrix.get_shape()[0] + 1):
        rows += f'{matrix.indptr[i]},'
    
    for i in range(matrix.getnnz()):
        cols += f'{matrix.indices[i]},'
        data += f'{matrix.data[i]},'
    
    file.write(rows + '\n')
    file.write(cols + '\n')
    file.write(data + '\n')
    file.close()


# Example usage
# file_path = './test_matrices/nvidia.mtx'
# mat = read_mtx_file(file_path)
# # print(mat)
# # print(mat.get_shape())
# # print(mat.row)
# # print(mat.col)
# # print(mat.data)

# # Write to csv file
# test = open('nvidia.csv', 'x')
# test.write(f'{mat.get_shape()[0]}, {mat.get_shape()[1]}, {mat.getnnz()}\n')
# rows = ''
# cols = ''
# data = ''
# for i in range(mat.getnnz()):
#     rows += f'{mat.row[i]},'
#     cols += f'{mat.col[i]},'
#     data += f'{mat.data[i]},'

# test.write(rows + '\n')
# test.write(cols + '\n')
# test.write(data + '\n')
# test.close()

# mat_csr = mat.tocsr()
# print(mat_csr)
# print(mat_csr.indptr)
# print(mat_csr.indices)
# print(mat_csr.data)

folder = './test_matrices/'


def main():
    filename = sys.argv[1]
    file_path = folder + filename + '.mtx'
    coo_mat = read_mtx_file(file_path)
    write_coo_to_csv(coo_mat, filename)

    # Convert to CSR
    csr_mat = coo_mat.tocsr()
    write_csr_to_csv(csr_mat, filename)





if __name__ == '__main__':
    main()
