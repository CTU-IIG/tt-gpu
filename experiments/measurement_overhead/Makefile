.PHONY: all clean
HOST ?= tx2fk
REMOTE_TARGET = nvidia@$(HOST)
EXE = meas_oh
REMOTE_WORKING_DIR = ~/meas_oh
CFLAGS := -Wall -Werror -O3

NVCCFLAGS := -Xptxas -O3 --ptxas-options=-v --compiler-options="$(CFLAGS)" \
	--generate-code arch=compute_62,code=[compute_62,sm_62] \
	--maxrregcount=32 -Xptxas -dlcm=ca \

#Export CUDA paths
export LIBRARY_PATH:=/usr/local/cuda/lib64:$(LIBRARY_PATH)
export LD_LIBRARY_PATH:=/usr/local/cuda/lib64:$(LD_LIBRARY_PATH)
export PATH:=/usr/local/cuda/bin:$(PATH)


all: $(EXE)

$(EXE): clock_test.cu
	nvcc -o $@  $(NVCCFLAGS) $^

target_build: deploy
	ssh -t $(REMOTE_TARGET) "cd $(REMOTE_WORKING_DIR) && make clean && make all"

target_run: target_build
	ssh -t $(REMOTE_TARGET) "cd $(REMOTE_WORKING_DIR) && ./$(EXE) 1"

target_clean: 
	ssh -t $(REMOTE_TARGET) "cd $(REMOTE_WORKING_DIR) && rm -f $(EXE)"

deploy:
		rsync -avz . ${REMOTE_TARGET}:${REMOTE_WORKING_DIR}

clean:
	rm -f $(EXE)
