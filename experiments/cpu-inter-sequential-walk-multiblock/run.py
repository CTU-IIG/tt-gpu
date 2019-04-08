#!/usr/bin/python3
from subprocess import Popen, PIPE
import json
import statistics as st
import sys
import pathlib

outpath = "out/"
pathlib.Path(outpath).mkdir(parents=True, exist_ok=True)

inter = "2" # 0 = rnd, 1= seq

scenarios = [# Filename              thread kernel datasize repetition
             ("128kb-1ke-32th.json",    32,     1,     128,      100),
             ("128kb-1ke-128th.json",  128,     1,     128,      100),
             ("128kb-1ke-256th.json",  256,     1,     128,      100),
             ("128kb-1ke-512th.json",  512,     1,     128,      100),
             ("128kb-1ke-1024th.json", 1024,    1,     128,      100),
             ]

             
for scenario in scenarios:
    filename = scenario[0]
    nof_thread = scenario[1]
    nof_kernel = scenario[2]
    data_size = scenario[3]
    nof_rep = scenario[4]
    meanblk = []
    minvblk = []
    maxvblk = []
    meanker = []
    minvker = []
    maxvker = []
    block = []

    for nof_block in range(1,33,1):
        print("-------------------------------")
        print("Dataset: "+str(data_size))
        print("Number of blocks: "+str(nof_block))
        print("Nof threads: "+str(nof_thread))
        print("-------------------------------")
        process = Popen(["./sequential-walk", str(nof_thread), str(nof_block),str(nof_kernel), str(nof_rep), str(data_size) , inter, "out.json"], stdout=PIPE)
        output = process.communicate()
        print(output)

        with open('out.json') as f:
            data = json.load(f)
            print("Exp_sum: " + str(data['exp_sum']) + ", real_sum: "+ data['real_sum'])
            timesblk = [float(i) for i in data['blocktimes']]
            timesker = [float(i)*1000 for i in data['kerneltimes']]
            meanblk.append(st.mean(timesblk))
            minvblk.append(min(timesblk))
            maxvblk.append(max(timesblk))
            meanker.append(st.mean(timesker))
            minvker.append(min(timesker))
            maxvker.append(max(timesker))
            block.append(nof_block)
            print("mean: "+str(meanblk[-1])+", min: "+str(minvblk[-1])+", max: "+str(maxvblk[-1]))
            print("mean: "+str(meanker[-1])+", min: "+str(minvker[-1])+", max: "+str(maxvker[-1]))

    # Write output json
    agg_data = {}
    agg_data['meanblk'] = meanblk
    agg_data['minblk'] = minvblk
    agg_data['maxblk'] = maxvblk
    agg_data['meanker'] = meanker
    agg_data['minker'] = minvker
    agg_data['maxker'] = maxvker
    agg_data['nof_blocks'] = block
    with open(outpath+filename, 'w') as outfile:
        json.dump(agg_data, outfile)
