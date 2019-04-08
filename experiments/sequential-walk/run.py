#!/usr/bin/python3
from subprocess import Popen, PIPE
import json
import statistics as st
import sys
import pathlib

outpath = "out/"
pathlib.Path(outpath).mkdir(parents=True, exist_ok=True)
executable="./sequential-walk"

nof_rep = 100

scenarios = [# Filename                    L1/L2  USE_SHMmake  cachemode prog
            ("out/cg-noshm-0-L2.json",        "cg",       "no",          0),
            ("out/ca-noshm-0-L1default.json", "ca",       "no",          0),
            ("out/ca-noshm-1-L1smaller.json", "ca",       "no",          1)]
            ("out/ca-noshm-2-L1bigger.json",  "ca",       "no",          2),
            ("out/ca-noshm-3-L1equal.json",   "ca",       "no",          0),
            ("out/ca-shm-walkhasshm-0-L1shared.json",     "ca",      "yes",          0),
            ("out/ca-shm-0-L1shared.json",     "ca",      "yes",          0)]

for scenario in scenarios:
    filename = scenario[0]
    cache_level = scenario[1]
    use_shm = scenario[2]
    cache_mode = scenario[3]

    #compile
    makeprocess = Popen(["make", "clean"], stdout=PIPE)
    output = makeprocess.communicate()
    print(output)

    makeprocess = Popen(["make", "cachemode="+cache_level, "USE_SHM="+use_shm], stdout=PIPE)
    output = makeprocess.communicate()
    print(output)

    mean = []
    median = []
    stdev = []
    minv = []
    maxv = []
    size = []

    for data_size in range(1,1024,1):
        print("-------------------------------")
        print("Datasize: "+str(data_size))
        print("-------------------------------")
        process = Popen([executable, str(nof_rep), str(data_size) , str(cache_mode), "out.json"], stdout=PIPE)
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
    with open(filename, 'w') as outfile:
        json.dump(agg_data, outfile)

