#!/usr/bin/python3
from subprocess import Popen, PIPE
import json
import statistics as st
import sys
import pathlib
import numpy as np

outpath = "out/"
pathlib.Path(outpath).mkdir(parents=True, exist_ok=True)
executable="./zc-roundtrip"

scenarios = [# Filename dummy
        ("roundtrip.json",0)]

for scenario in scenarios:
    filename = scenario[0]

    times={}

    print("-------------------------------")
    print("Run Globaltimer jitter benchmark")
    print("-------------------------------")
    process = Popen([executable, "out.json"], stdout=PIPE)
    output = process.communicate()
    print(output)

    with open('out.json') as f:
        data = json.load(f)

    times = data['times']
    nofPasses = data['nofpasses']
    nofkernel = data['nofkernel']

    # Write output json
    agg_data = {}
    agg_data['times'] = times
    agg_data['nofpasses'] = nofPasses
    agg_data['nofkernel'] = nofkernel
    with open(outpath+filename, 'w') as outfile:
        json.dump(agg_data, outfile)
