#include "include/transpose_coo.cuh"
#include "include/transpose_csr.cuh"
#include "include/transpose_csr_v2.cuh"
#include "include/cusparse_coo.cuh"
#include "include/cusparse_csr.cuh"
#include <array>

std::array<std::string, 10> matrixNames = {
    "arc130",
    "fd15",
    "bips07_1998",
    "bayer10",
    "piston",
    "cage10",
    "Ill_Stokes",
    "msc10848",
    "appu",
    "TSOPF_RS_b300_c2"
};


int main(int argc, char const *argv[]) {

    /*
    -------- DEVICE PROPERTIES --------
    */
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("\n");
    printf("Device name: %s\n", prop.name);
    printf("Memory Clock Rate (MHz): %d\n", prop.memoryClockRate / 1024);
    printf("Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
    printf("Peak Memory Bandwidth (GB/s): %.1f\n", 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
    printf("Shared memory per block (Kbytes) %.1f\n",(float)(prop.sharedMemPerBlock)/1024.0);  
    printf("--------------------------------\n");

    if (argc != 3) {
        printf("Usage: ./main <algorithm> <number>\n");
        printf("algorithm: coo | csr\n");
        printf("number: 0 - 9, to execute the transposition of a specific matrix\n");
        exit(1);
    } else {
        std::string algorithm = argv[1];
        int index = std::stoi(argv[2]);

        if (algorithm != "coo" && algorithm != "csr") {
            printf("Invalid algorithm, the algorithm should be either coo or csr\n");
            exit(1);
        }

        if (index >= 0 && index <= matrixNames.size() - 1) {
            if (algorithm == "coo") {
                transpose_COO(matrixNames[index]);
                cusparse_COO(matrixNames[index]);
            } else {
                transpose_CSR(matrixNames[index]);
                transpose_CSR_v2(matrixNames[index]);
                cusparse_CSR(matrixNames[index]);
            }

        } else {
            printf("Invalid matrix number, the number should be in range [0,%ld]\n", matrixNames.size() - 1);
            exit(1);
        }
    }

   
  return 0;
}
