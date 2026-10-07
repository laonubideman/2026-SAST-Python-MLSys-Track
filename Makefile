PYTHON ?= python3
CXX ?= c++
NVCC ?= nvcc

PYTHON_INCLUDE = $(shell $(PYTHON) -c 'import sysconfig; print(sysconfig.get_path("include"))')
PYBIND11_INCLUDE = $(shell $(PYTHON) -c 'import pybind11; print(pybind11.get_include())')
EXT_SUFFIX = $(shell $(PYTHON) -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')

ifeq ($(shell uname -s),Darwin)
PYTHON_LINK_FLAGS := -undefined dynamic_lookup
endif

.PHONY: default cuda clean

default:
	$(CXX) -O3 -Wall -shared -std=c++11 -fPIC \
		-I"$(PYTHON_INCLUDE)" -I"$(PYBIND11_INCLUDE)" \
		src/simple_ml_ext.cpp $(PYTHON_LINK_FLAGS) \
		-o "src/simple_ml_ext$(EXT_SUFFIX)"

cuda:
	$(NVCC) -O3 -std=c++14 --shared -Xcompiler -fPIC \
		-I"$(PYTHON_INCLUDE)" -I"$(PYBIND11_INCLUDE)" \
		src/simple_ml_cuda.cu -o "src/simple_ml_cuda$(EXT_SUFFIX)"

clean:
	rm -f src/simple_ml_ext*.so src/simple_ml_ext*.dylib src/simple_ml_ext*.pyd
	rm -f src/simple_ml_cuda*.so src/simple_ml_cuda*.dylib src/simple_ml_cuda*.pyd
