#!/usr/bin/python3
from subprocess import Popen, PIPE
import json
import statistics as st
import sys
import pathlib

outpath = "out/"
pathlib.Path(outpath).mkdir(parents=True, exist_ok=True)

executable = "./sequential-walk"

scenarios = [# Filename                         block datasize repetition
             ("1-cg-kernels-different-elem.json",   1,      1,     10),
             ("2-cg-kernels-different-elem.json",   1,      2,     10),
             ("4-cg-kernels-different-elem.json",   1,      4,     10),
             ("12-cg-kernels-different-elem.json",  1,     12,     10),
             ("16-cg-kernels-different-elem.json",  1,     16,     10),
             ("64-cg-kernels-different-elem.json",  1,     64,     10),
             ("128-cg-kernels-different-elem.json", 1,    128,     10),
             ("256-cg-kernels-different-elem.json", 1,    256,     10)]

for scenario in scenarios:
    filename = scenario[0]
    nof_block = scenario[1]
    data_size = scenario[2]
    nof_rep = scenario[3]

    mean = []
    median = []
    stdev = []
    minv = []
    maxv = []
    threads = []

    for nof_thread in range(1,1025,1):
        print("-------------------------------")
        print("Number of threads: "+str(nof_thread))
        print("Size: "+str(data_size))
        print("-------------------------------")

        process = Popen([executable, str(nof_thread), str(nof_block),str(nof_rep), str(data_size) , "out.json"], stdout=PIPE)
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
            threads.append(nof_thread)
            print("mean: {:6.4f}, min: {:6.4f}, max: {:6.4f}, diff: {:6.4f}".format(mean[-1], minv[-1], maxv[-1],maxv[-1]- minv[-1]))

    # Write output json
    agg_data = {}
    agg_data['mean'] = mean
    agg_data['median'] = median
    agg_data['min'] = minv
    agg_data['max'] = maxv
    agg_data['stdev'] = stdev
    agg_data['nof_threads'] = threads
    with open(outpath+filename, 'w') as outfile:
        json.dump(agg_data, outfile)
