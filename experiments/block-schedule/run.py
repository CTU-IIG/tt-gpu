#!/usr/bin/python3
from subprocess import Popen, PIPE
import json
import statistics as st
import sys
import pathlib
import numpy as np

outpath = "out/"
pathlib.Path(outpath).mkdir(parents=True, exist_ok=True)
executable="./block-schedule"

scenarios = [# Filename dummy
        ("jitter.json",0)]

for scenario in scenarios:
    filename = scenario[0]

    times={}
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
    times = data['times']
    for block in blocks:
        smid[block] = data['smid_'+block]

    print(smid)
    # Write output json
    agg_data = {}
    agg_data['times'] = times
    agg_data['smid'] = smid
    agg_data['blocks'] = blocks
    with open(outpath+filename, 'w') as outfile:
        json.dump(agg_data, outfile)
