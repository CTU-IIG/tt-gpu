.PHONY: all clean target_clean
HOST ?= tx2fk
EXE = random-walk
REMOTE_TARGET = nvidia@$(HOST)
REMOTE_WORKING_DIR = ~/random_walk
REMOTE_OUTPUT_DIR ?=~/random_walk/out
CFLAGS := -Wall -Werror -O3

cachemode ?=ca
USE_SHM ?=no

ifeq ($(USE_SHM), no) 
	SHM:=
else
	SHM:=-DUSE_SHARED
endif

NVCCFLAGS := -Xptxas -O3 --ptxas-options=-v --compiler-options="$(CFLAGS)" \
	--generate-code arch=compute_53,code=[compute_53,sm_53] \
	--generate-code arch=compute_62,code=[compute_62,sm_62] \
	-Xptxas -dlcm=$(cachemode) \

#Export CUDA paths
export LIBRARY_PATH:=/usr/local/cuda/lib64:$(LIBRARY_PATH)
export LD_LIBRARY_PATH:=/usr/local/cuda/lib64:$(LD_LIBRARY_PATH)
export PATH:=/usr/local/cuda/bin:$(PATH)


all: $(EXE)

$(EXE): $(EXE).cu
	nvcc $(NVCCFLAGS) -o $@ $(SHM) $^

target_build: deploy
	ssh -t $(REMOTE_TARGET) "cd $(REMOTE_WORKING_DIR) && make clean && make all"

target_run: target_build
	ssh -t $(REMOTE_TARGET) "cd $(REMOTE_WORKING_DIR) && python3 run.py"
	rsync -avz ${REMOTE_TARGET}:${REMOTE_OUTPUT_DIR} .

deploy:
	rsync -avz . ${REMOTE_TARGET}:${REMOTE_WORKING_DIR}

clean:
	rm -f $(EXE)

target_clean:
	ssh $(REMOTE_TARGET) "rm -rf $(REMOTE_WORKING_DIR)"
