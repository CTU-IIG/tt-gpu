#!/usr/bin/python3
import json
import matplotlib.pyplot as plt
import numpy as np
import itertools

file1='out/jitter.json'

with open(file1) as f1:
    data1 = json.load(f1)

marker = itertools.cycle(('o', 'x', '+', '.')) 

mylegend = []

block_sm0 = []
block_sm1 = []

blocks_sm = {}
blocks_sm['0'] = block_sm0
blocks_sm['1'] = block_sm1

blocks = np.array(data1['blocks']).astype(float)
times = np.array(data1['times']).astype(float)

plt.plot(blocks,times,'o')


for block in data1["blocks"]:
    blocks_sm[data1['smid'][block]].append(block)
print(blocks_sm)
print(min(times))
print(max(times))
print(max(times)-min(times))
plt.show()
