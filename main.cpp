#include "include/transpose_coo.cuh"
#include "include/transpose_csr.cuh"
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
    // TODO: think about moving this to a separate function
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("\n");
    printf("Device name: %s\n", prop.name);
    printf("Memory Clock Rate (MHz): %d\n", prop.memoryClockRate / 1024);
    printf("Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
    printf("Peak Memory Bandwidth (GB/s): %.1f\n", 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
    printf("Shared memory per block (Kbytes) %.1f\n",(float)(prop.sharedMemPerBlock)/1024.0);  
    printf("--------------------------------\n");

    if (argc != 2) {
        printf("Usage: ./main <number | all>\n");
        printf("number: 0 - 9, to execute the transposition of a specific matrix\n");
        printf("all: to execute the transposition of all matrices\n");
        exit(1);
    } else {
        std::string arg = argv[1];
        if (arg == "all") {
            for (std::string matrixName : matrixNames) {
                // transpose_COO(matrixName);
                transpose_CSR(matrixName);
            }
        } else {
            int matrixNumber = std::stoi(arg);
            if (matrixNumber < 0 || matrixNumber > 10) {
                printf("Invalid matrix number\n");
                exit(1);
            } else {
                std::string matrixName = matrixNames[matrixNumber];
                // transpose_COO(matrixName);
                transpose_CSR(matrixName);
            }
        }
    }

   
  return 0;
}
