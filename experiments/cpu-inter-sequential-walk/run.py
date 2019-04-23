#!/usr/bin/python3
from subprocess import Popen, PIPE
import json
import statistics as st
import sys
import pathlib

outpath = "out/"
pathlib.Path(outpath).mkdir(parents=True, exist_ok=True)
exe = "./sequential-walk"

inter = "0" # 0 = rnd, 1= seq
nof_rep = 10

scenarios = [# Filename                 use_zerocopy  interference method(0=rnd, 1=seq, 2=no)
             ("nozc-no-seqwalk.json",     0,             2),
             ("zc-no-seqwalk.json",       1,             2),
             ("nozc-rnd-seqwalk.json",    0,            0),
             ("zc-rnd-seqwalk.json",      1,            0),
             ("nozc-seq-seqwalk.json",    0,            1),
             ("zc-seq-seqwalk.json",      1,            1)]
             
for scenario in scenarios:
    filename = scenario[0]
    use_zerocopy = scenario[1]
    inter_method = scenario[2]
    mean = []
    median = []
    stdev = []
    minv = []
    maxv = []
    size = []


    for data_size in range(1,1500,1):
        print("-------------------------------")
        print("Dataset: "+str(data_size))
        print("Use Zerocopy: "+str(use_zerocopy))
        print("Interference method: "+str(inter_method))
        print("-------------------------------")
        process = Popen([exe, str(nof_rep), str(data_size),str(inter_method),str(use_zerocopy), "out.json"], stdout=PIPE)
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
            size.append(data_size)
            print("mean: "+str(mean[-1])+", min: "+str(minv[-1])+", max: "+str(maxv[-1]))

    # Write output json
    agg_data = {}
    agg_data['mean'] = mean
    agg_data['median'] = median
    agg_data['min'] = minv
    agg_data['max'] = maxv
    agg_data['stdev'] = stdev
    agg_data['size'] = size
    with open(outpath+filename, 'w') as outfile:
        json.dump(agg_data, outfile)
