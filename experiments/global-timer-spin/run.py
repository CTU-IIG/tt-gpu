#!/usr/bin/python3
from subprocess import Popen, PIPE
import json
import statistics as st
import sys
import pathlib
import numpy as np

outpath = "out/"
pathlib.Path(outpath).mkdir(parents=True, exist_ok=True)
executable="./global-timer-spin"

scenarios = [# Filename dummy
        ("jitter.json",0)]

for scenario in scenarios:
    filename = scenario[0]

    times={}
    differences={}
    smid={}
    blocks = []

    print("-------------------------------")
    print("Run Globaltimer jitter benchmark")
    print("-------------------------------")
    process = Popen([executable, "out.json"], stdout=PIPE)
    output = process.communicate()
    print(output)

    with open('out.json') as f:
        data = json.load(f)

    blocks = data['blocks']
    for block in blocks:
        tmp = np.array(data['times_'+block], dtype=int)
        times[block] = tmp.tolist()
        differences[block] = np.diff(tmp).tolist()
        smid[block] = data['smid_'+block]
    print(smid)
    # Write output json
    agg_data = {}
    agg_data['times'] = times
    agg_data['differences'] = differences
    agg_data['smid'] = smid
    agg_data['blocks'] = blocks
    agg_data['clockRatekHz'] = data['clockRatekHz']
    agg_data['stepns'] = data['stepns']
    with open(outpath+filename, 'w') as outfile:
        json.dump(agg_data, outfile)
