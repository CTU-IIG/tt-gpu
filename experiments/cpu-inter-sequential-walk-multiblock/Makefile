.PHONY: all clean target_clean
HOST ?= tx2fk
EXE=sequential-walk
REMOTE_TARGET = nvidia@$(HOST)
REMOTE_WORKING_DIR = ~/cpu-inter-sequential_walk-multiblock
REMOTE_OUTPUT_DIR ?=$(REMOTE_WORKING_DIR)/out
CFLAGS := -Wall -Werror -O3 -lpthread -lrt 

cachemode ?=cg

NVCCFLAGS := -Xptxas -O3 --ptxas-options=-v --compiler-options="$(CFLAGS)" \
	--generate-code arch=compute_62,code=[compute_62,sm_62] \
	--maxrregcount=32 -Xptxas -dlcm=$(cachemode) \

#Export CUDA paths
export LIBRARY_PATH:=/usr/local/cuda/lib64:$(LIBRARY_PATH)
export LD_LIBRARY_PATH:=/usr/local/cuda/lib64:$(LD_LIBRARY_PATH)
export PATH:=/usr/local/cuda/bin:$(PATH)


all: $(EXE)

$(EXE): main.c interference.c $(EXE).cu
	nvcc $(NVCCFLAGS) $^ -o $@

target_build: deploy
	ssh -t $(REMOTE_TARGET) "cd $(REMOTE_WORKING_DIR) && make clean && make all"

target_run: target_build
	ssh -t $(REMOTE_TARGET) "cd $(REMOTE_WORKING_DIR) && export LD_BIND_NOW && python3 run.py"
	rsync -avz ${REMOTE_TARGET}:${REMOTE_OUTPUT_DIR} .

deploy:
		rsync -avz . ${REMOTE_TARGET}:${REMOTE_WORKING_DIR}

clean:
	rm -f $(EXE)

target_clean:
	ssh $(REMOTE_TARGET) "rm -rf $(REMOTE_WORKING_DIR)"
