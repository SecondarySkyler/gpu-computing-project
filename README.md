# GPU Computing Project
Project for the GPU Computing course held at the University of Trento for the A.Y 2023/2024.

## Goal of the project
This project requires to design an efficient algorithm to transpose a sparse matrix. Specifically the
matrix should be highly sparse, namely the number of zero element is more than 75% of the whole
(n × n) elements. The implementation should emphasize:
- storage format for storing sparse matrices (for example, compressed sparse row);
- the implementation to perform the transposition;
- a comparison against vendors’ library (e.g., cuSPARSE);
- dataset for the benchmark (compare all the implementation presented by selecting at least 10
matrices from suite sparse matrix collection https://sparse.tamu.edu/);

As usual, <u>the metric to consider is the effective bandwidth.</u>

## Installation and Usage
First clone the repository
```
git clone https://github.com/SecondarySkyler/gpu-computing-project.git
```
Change directory
```
cd gpu-computing-project/
```
Compile the executable
```
make all
```
At this point you are ready to run the project.

## Matrices used for testing
The matrices used for benchmarking the project has been collected from the [SuiteSparse Matrix Collection](https://sparse.tamu.edu/).
In particular, this is comprehensive list:
### How to use a different matrix
If you would like to test the project with different matrices, be sure to use the *MatrixMarket* format.
From the SuiteSparse website you can download the chosen matrix using *wget* (I don't know why but clicking on the download button won't start the download). <br />
I assume you downloaded the *tar.gz* in the root folder of the project. <br />
```
chmod 755 tarball_name.tar.gz
tar -xvzf tarball_name.tar.gz
```
At this point you will find the directory containing the *.mtx* file. <br />
Move it inside the `test_matrices/`. 
```
mv matrix_name/matrix_name.mtx ./test_matrices/
```
Now we need to generate the corresponding COO and CSR csv files. To do so I wrote a simple python parser. To use it run (!!! insert the matrix name without the .mtx extension)
```
python3 parser.py matrix_name
```
This command will generate the corresponding files under the `./test_matrices/coo/` and `./test_matrices/csr/` directories. <br />
At this point you should be able to run the project specifying the name (without .mtx) of the matrix.