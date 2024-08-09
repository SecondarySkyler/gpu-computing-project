# CC = g++
# NVCC = nvcc
# NVCC_FLAGS = --gpu-architecture=sm_80 -m64
# SRC := src
# INC := include

# all: main.o utilities.o main 

# main.o: $(SRC)/main.cu
# 	$(NVCC) $(NVCC_FLAGS) -c $(SRC)/main.cu -o main.o

# utilities.o: $(SRC)/utilities.cpp $(INC)/utilities.hpp
# 	$(CC) $(INC) -c $(SRC)/utilities.cpp -o utilities.o

# main: main.o
# 	$(NVCC) $(NVCC_FLAGS) main.o utilities.o -o main -lcudart -lcusparse

# all: transpose.o utilities.o transpose 

# transpose.o: $(SRC)/transpose.cu
# 	$(NVCC) $(NVCC_FLAGS) -c $(SRC)/transpose.cu -o transpose.o

# utilities.o: $(SRC)/utilities.cpp $(INC)/utilities.hpp
# 	$(CC) $(INC) -c $(SRC)/utilities.cpp -o utilities.o

# transpose: transpose.o
# 	$(NVCC) $(NVCC_FLAGS) transpose.o utilities.o -o transpose -lcudart -lcusparse

###########################################################

## USER SPECIFIC DIRECTORIES ##

# CUDA directory:
CUDA_ROOT_DIR=/usr/local/cuda

##########################################################

## CC COMPILER OPTIONS ##

# CC compiler options:
CC=g++
CC_FLAGS=
CC_LIBS=

##########################################################

## NVCC COMPILER OPTIONS ##

# NVCC compiler options:
NVCC=nvcc
NVCC_FLAGS= --gpu-architecture=sm_80 -m64
NVCC_LIBS=

# CUDA library directory:
CUDA_LIB_DIR= -L$(CUDA_ROOT_DIR)/lib64
# CUDA include directory:
CUDA_INC_DIR= -I$(CUDA_ROOT_DIR)/include
# CUDA linking libraries:
CUDA_LINK_LIBS= -lcudart

##########################################################

## Project file structure ##

# Source file directory:
SRC_DIR = src

# Object file directory:
OBJ_DIR = bin

# Include header file diretory:
INC_DIR = include

##########################################################

## Make variables ##

# Target executable name:
EXE = transpose

# Object files:
OBJS = $(OBJ_DIR)/main.o $(OBJ_DIR)/transpose_coo.o $(OBJ_DIR)/transpose_csr.o $(OBJ_DIR)/utilities.o

##########################################################

## Compile ##

# Link c++ and CUDA compiled object files to target executable:
$(EXE) : $(OBJS)
	$(CC) $(CC_FLAGS) $(OBJS) -o $@ $(CUDA_INC_DIR) $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS)

# Compile main .cpp file to object files:
$(OBJ_DIR)/%.o : %.cpp
	$(CC) $(CC_FLAGS) -c $< -o $@

# Compile C++ source files to object files:
$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cpp include/%.h
	$(CC) $(CC_FLAGS) -c $< -o $@

# Compile CUDA source files to object files:
$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cu $(INC_DIR)/%.cuh
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ $(NVCC_LIBS)

# Clean objects in object directory.
clean:
	$(RM) bin/* *.o $(EXE)