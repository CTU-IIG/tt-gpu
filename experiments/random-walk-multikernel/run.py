#!/usr/bin/python3
from subprocess import Popen, PIPE
import json
import statistics as st
import sys
import pathlib

outpath = "out/"
pathlib.Path(outpath).mkdir(parents=True, exist_ok=True)
executable = "./random-walk"


scenarios = [# Filename                                thread block datasize repetition
             ("128-cg-kernels-1thread-same-elem.json",    1,    1,    128,      100),
             ("128-cg-kernels-32thread-same-elem.json",  32,    1,    128,      100),
             ("128-cg-kernels-128thread-same-elem.json",128,    1,    128,      100),

             ("256-cg-kernels-1thread-same-elem.json",    2,    1,    256,      100),
             ("256-cg-kernels-32thread-same-elem.json",  32,    1,    256,      100),
             ("256-cg-kernels-128thread-same-elem.json",128,    1,    256,      100),

             ("512-cg-kernels-1thread-same-elem.json",    2,    1,    512,      100),
             ("512-cg-kernels-32thread-same-elem.json",  32,    1,    512,      100),
             ("512-cg-kernels-128thread-same-elem.json",128,    1,    512,      100)]

             
for scenario in scenarios:
    filename = scenario[0]
    nof_thread = scenario[1]
    nof_block = scenario[2]
    data_size = scenario[3]
    nof_rep = scenario[4]
    mean = []
    median = []
    stdev = []
    minv = []
    maxv = []
    kernel = []

    for nof_kernel in range(1,17,1):
        print("-------------------------------")
        print("Number of kernels: "+str(nof_kernel))
        print("Nof threads: "+str(nof_thread))
        print("-------------------------------")
        process = Popen([executable, str(nof_thread), str(nof_block),str(nof_kernel), str(nof_rep), str(data_size) , "out.json"], stdout=PIPE)
        output = process.communicate()
        print(output)

        with open('out.json') as f:
            data = json.load(f)
            print("Exp_sum: " + str(data['exp_sum']) + ", real_sum: "+ data['real_sum'])
            times = [float(i) for i in data['times']]
            mean.append(st.mean(times))
            median.append(st.median(times))
            stdev.append(st.stdev(times))
            minv.append(min(times))
            maxv.append(max(times))
            kernel.append(nof_kernel)
            print("mean: "+str(mean[-1])+", min: "+str(minv[-1])+", max: "+str(maxv[-1]))

    # Write output json
    agg_data = {}
    agg_data['mean'] = mean
    agg_data['median'] = median
    agg_data['min'] = minv
    agg_data['max'] = maxv
    agg_data['stdev'] = stdev
    agg_data['nof_kernels'] = kernel
    with open(outpath+filename, 'w') as outfile:
        json.dump(agg_data, outfile)
