#ifndef _CUSPARSECSR_H_
#define _CUSPARSECSR_H_

#include <cuda_runtime.h>
#include <cusparse.h>
#include <string>

void cusparse_CSR(std::string fileName);

#endif