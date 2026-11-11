CXX ?= g++
NVCC ?= nvcc

CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra
NVCCFLAGS ?= -O3 -std=c++17 -Xcompiler -Wall -Xcompiler -Wextra \
             -gencode arch=compute_75,code=sm_75 \
             -gencode arch=compute_80,code=sm_80 \
             -gencode arch=compute_89,code=sm_89

COMMON_SRCS = src/common/matrix_loader.cpp src/common/correctness.cpp src/common/benchmark.cpp
COMMON_HEADERS = src/common/matrix_loader.h src/common/correctness.h src/common/benchmark.h src/common/cuda_common.h

BIN_DIR = bin
TARGETS_CPU = $(BIN_DIR)/cpu_spmv
TARGETS_CUDA = $(BIN_DIR)/scalar_csr $(BIN_DIR)/vector_csr $(BIN_DIR)/ell $(BIN_DIR)/hyb $(BIN_DIR)/cusparse_spmv

all: $(TARGETS_CPU)
	@if command -v nvcc >/dev/null 2>&1; then \
		$(MAKE) cuda; \
	else \
		echo "nvcc not available; built CPU baseline only. Run 'make cuda' on GPU systems."; \
	fi

cpu: $(TARGETS_CPU)

cuda: $(TARGETS_CUDA)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)
	mkdir -p results/raw results/summary results/plots

$(BIN_DIR)/cpu_spmv: src/cpu_spmv.cpp $(COMMON_SRCS) $(COMMON_HEADERS) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) src/cpu_spmv.cpp $(COMMON_SRCS) -o $@

$(BIN_DIR)/scalar_csr: src/scalar_csr.cu $(COMMON_SRCS) $(COMMON_HEADERS) | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) src/scalar_csr.cu $(COMMON_SRCS) -o $@

$(BIN_DIR)/vector_csr: src/vector_csr.cu $(COMMON_SRCS) $(COMMON_HEADERS) | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) src/vector_csr.cu $(COMMON_SRCS) -o $@

$(BIN_DIR)/ell: src/ell.cu $(COMMON_SRCS) $(COMMON_HEADERS) | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) src/ell.cu $(COMMON_SRCS) -o $@

$(BIN_DIR)/hyb: src/hyb.cu $(COMMON_SRCS) $(COMMON_HEADERS) | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) src/hyb.cu $(COMMON_SRCS) -o $@

$(BIN_DIR)/cusparse_spmv: src/cusparse_spmv.cu $(COMMON_SRCS) $(COMMON_HEADERS) | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) src/cusparse_spmv.cu $(COMMON_SRCS) -lcusparse -o $@

clean:
	rm -rf $(BIN_DIR)

.PHONY: all cpu cuda clean
