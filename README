# Experiments for Predictable Execution of GPU Kernels
**Source code for OSPERT19 submission**

This repository contains all experiments, raw results and the
corresponding paper for the OSPERT19 submission by Flavio Kreiliger,
Joel Matějka, Michal Sojka and Zdeněk Hanzálek.


## Preconditions

The experiments can be run on the Jetson TX2 platform with CUDA installed.

Other tools needed on the TX2:
1) python3
2) numpy
3) Working ssh connection
4) nvidia user

To have reproducible results, it is recommended to set the powermode to MAXN
with the command:

`sudo nvpmodel -m 0`

And to configure all frequencies to the maximum by use of:

`sudo /home/nvidia/jetson_clocks.sh`

## How to run the experiments
Each experiment can be run on the Jetson TX2 with the command:

`make HOST="hostname" target_run`

Then the experiment is run as the nvidia user and the results are
copied back to the host computer in json format.

The plotting scripts inside the experiments folders can be used to
visualize the results.

## Plotting

After the experiment has finished, the resulting data is stored in the `out`
folder on the host. To visualize the results, the plotting scripts can be used
with Python3 and matplotlib. Sometimes it is necessary to copy the content of
the out folder to the existing data folder (see
prem-2dconvolution/scheduling-results) to overwrite our provided raw data.

## Experiments
### PREM experiments

**global-timer-jitter:**

A kernel with 4 CUDA blocks with one thread each is launched. Each thread
stores the globaltimer value multiple times into the shared memory of its CUDA
block. The retrieved timer values are plotted by the provided python scripts.
If nvprof is run once on an arbitrary kernel (for example vecAdd from the CUDA
samples) the globaltimer resolution is increased from 1 us to 160ns.

**global-timer-spin:**

A kernel with 4 CUDA blocks with one thread each is launched. Each thread is
spinning on the globaltimer for a preconfigured number of nanoseconds. The
number of clock cycles needed to perform the spinning is recorded for each
thread by use of the clock() call.

**zc-roundtrip:**

Two kernels launched in two different streams perform a Ping-Pong experiment on
zero-copy memory. The round-trip time is recorded multiple times. This
experiment evaluates the overhead of synchronization between two kernels
through host-pinned zerocopy memory.

**prem-2dconvolution:**

Experiment based on the
[Polybench-ACC](https://github.com/cavazos-lab/PolyBench-ACC) [convolution
benchmark](https://github.com/cavazos-lab/PolyBench-ACC/tree/master/CUDA/stencils/convolution-2d).
The original polybench implementation (in the paper called legacy) runs with 1
or 4 kernels in parallel on the GPU to show the contention between the kernels
and the resulting execution jitter. Further, the legacy implementation has been
tiled to use shared memory. This allows the split the processing of one file
into the PREM phases *prefetch*, *compute* and *writeback*. Different
experiments have been performed to shift and schedule these tile processing
phases by use of the globaltimer. It is important to run nvprof once on an
arbitrary kernel to have accurate results.

**measurement_overhead:**

Small experiment to evaluate call overhead to globaltimer, clock() and
clock64().

**block-schedule:**

Simple experiment to see how many blocks with different thread configurations
are scheduled on one SM at the same time. Each block recording its start time
and then spins for a configured amount of time.

### Based on random and sequential walks

**random-walk:**

One thread performs a random walk on the GPU on arrays of different sizes to
evaluate the cache sizes of the GPU.

**sequential-walk:**

One thread performs a sequential walk on the GPU on arrays of different sizes
to evaluate the cache sizes of the GPU.

**random-walk-multikernel:**

Multiple kernels are launched in different streams to allow parallel execution.
Each thread in the kernels performs a random walk.  No CPU interference takes
place.

**random-walk-multithread:**

One kernel with one block and changing number of threads is launched. Each
thread performs a random walk on a shared data set.  To ensure contention
between the threads, each thread starts at a different element.

**sequential-walk-multikernel:**

Multiple kernels are launched in different streams to allow parallel execution.
Each thread in the kernels performs a sequential walk. No CPU interference
takes place.

**sequential-walk-multithread:**

One kernel with one block and changing number of threads is launched. Each
thread performs a sequential walk on a shared data set. Thread 0 starts with
element 0, thread 1 with element 1 and so on.

**cpu-inter-random-walk:**

One thread performs a random walk on the GPU. During this walk, all CPU cores
perform either no, sequential or random memory accesses to generate memory
interference.

**cpu-inter-sequential-walk:**

One thread performs a sequential walk on the GPU. During this walk, all CPU
cores perform either no, sequential or random memory accesses to generate
memory interference.

**cpu-inter-sequential-walk-multiblock:**

One kernel is launched with different thread and block configuration.  Each
thread performs a sequential walk on an array. The average number of cycles to
access one element is recorded for each running thread.  During the walk the
CPU cores perform no, sequential or random memory accesses generate memory
interference.

**cpu-inter-sequential-walk-multikernel:**

Multiple kernels are launched in different streams to allow parallel execution.
Each thread in the kernels performs a sequential walk.  During the walk the CPU
cores perform no, sequential or random memory accesses generate memory
interference.
