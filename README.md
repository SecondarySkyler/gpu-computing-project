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
make
```
At this point you are ready to run the project using:
```
./transpose <coo | csr> <0 - 9>
```
Where you can specify which algorithm to use and a number between 0 and 9 to perform the transposition on a specific matrix.
<u>__Example usage:__</u>
```
./transpose coo 2
```
This will execute the COO transposition on matrix *bips07_1998*.

## Matrices used for testing
The matrices used for benchmarking the project has been collected from the [SuiteSparse Matrix Collection](https://sparse.tamu.edu/).
In particular, this is a comprehensive table:
|                                **Source**                                | **Formal Name**  | **Dimensions** | **NNZ**   | **Sparsity** |
|:------------------------------------------------------------------------:|------------------|----------------|-----------|--------------|
| [HB/arc130](https://sparse.tamu.edu/HB/arc130)                           |      arc130      |    130 x 130   |   1,282   |      92%     |
| [Hohn/fd15](https://sparse.tamu.edu/Hohn/fd15)                           |       fd15       |  11532 x 11532 |   44,206  |      99%     |
| [Rommes/bips07_1998](https://sparse.tamu.edu/Rommes/bips07_1998)         |    bips07_1998   |  15066 x 15066 |   62,198  |      99%     |
| [Grund/bayer10](https://sparse.tamu.edu/Grund/bayer10)                   |      bayer10     |  13436 x 13436 |   71,594  |      99%     |
| [Oberwolfach/piston](https://sparse.tamu.edu/Oberwolfach/piston)         |      piston      |   2025 x 2025  |  100,015  |      97%     |
| [vanHeukelum/cage10](https://sparse.tamu.edu/vanHeukelum/cage10)         |      cage10      |  11397 x 11397 |  150,645  |      99%     |
| [Szczerba/Ill_Stokes](https://sparse.tamu.edu/Szczerba/Ill_Stokes)       |    Ill_Stokes    |  20896 x 20896 |  191,368  |      99%     |
| [Boeing/msc10848](https://sparse.tamu.edu/Boeing/msc10848)               |     msc10848     |  10848 x 10848 | 1,229,776 |      98%     |
| [Simon/appu](https://sparse.tamu.edu/Simon/appu)                         |       appu       |  14000 x 14000 | 1,853,104 |      99%     |
| [TSOPF/TSOPF_RS_b300_c2](https://sparse.tamu.edu/TSOPF/TSOPF_RS_b300_c2) | TSOPF_RS_b300_c2 |  28338 x 28338 | 2,943,887 |      99%     |
### How to use a different matrix
If you would like to test the project with different matrices, be sure to use the *MatrixMarket* format.
From the SuiteSparse website you can download the chosen matrix using *wget* (I don't know why but clicking on the download button won't start the download). <br />
I assume you downloaded the *tar.gz* in the root folder of the project. <br />
```
chmod 755 tarball_name.tar.gz
tar -xvzf tarball_name.tar.gz
```
At this point you will find the directory containing the *.mtx* file. <br />
Move the *.mtx* file inside the `test_matrices/`. 
```
mv matrix_name/matrix_name.mtx ./test_matrices/
```
Now we need to generate the corresponding COO and CSR csv files. To do so I wrote a simple python parser. To use it run (!!! insert the matrix name without the .mtx extension)
```
python3 parser.py matrix_name
```
This command will generate the corresponding files under the `./test_matrices/coo/` and `./test_matrices/csr/` directories. <br />
As last step modify the *matrixNames* array in `main.cpp` adding the name of the downloaded matrix. (Remember also to change the size of the std::array)  
Now you can run `make clean` if you have previously built the project, then run `make` to compile.  
At this point you should be able to run the project specifying the index of the matrix.