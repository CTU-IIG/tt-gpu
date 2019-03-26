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

clockRateHz = int(data1["clockRatekHz"])*1000
stepns = int(data1["stepns"])

for block in data1["blocks"]:
    mymarker = next(marker)
    mylegend.append("Block: "+block+" on SM "+data1['smid'][block])

    blocks_sm[data1['smid'][block]].append(block)
    x1 = list(range(1,len(data1['times'][block])+1))
    x2 = list(range(1,len(data1['differences'][block])+1))

    plt.subplot(2, 2, 1)

    plt.plot(x1, data1['times'][block], marker=mymarker)
    plt.grid(True, which="both")
    plt.title('Global timer spin jitter TX2\n')
    plt.ylabel('Clock [cyc]')

    plt.subplot(2, 2, 2)
    plt.plot(x2, data1['differences'][block], marker=mymarker)
    plt.grid(True, which="both")
    plt.title('Global timer spin jitter differences between steps TX2\n')
    plt.ylabel('Clock [cyc]')
plt.gca().legend(mylegend)





block0 = blocks_sm['0'][0]
block1 = blocks_sm['0'][1]
diff = [x1 - x2 for (x1, x2) in zip(data1['times'][block0], data1['times'][block1])]
x1 = list(range(1,len(data1['times'][block])+1))

plt.subplot(2, 2, 3)

plt.plot(x1, diff, marker=mymarker)
plt.grid(True, which="both")
plt.title('Diff SM 0 (block' + block0 +'-block'+block1+')')
plt.xlabel('Iteration')
plt.ylabel('Clock [cyc]')

block0 = blocks_sm['1'][0]
block1 = blocks_sm['1'][1]
diff = [x1 - x2 for (x1, x2) in zip(data1['times'][block0], data1['times'][block1])]
x1 = list(range(1,len(data1['times'][block])+1))

plt.subplot(2, 2, 4)

plt.plot(x1, diff, marker=mymarker)
plt.grid(True, which="both")
plt.title('Diff SM 1 (block' + block0 +'-block'+block1+')')
plt.xlabel('Iteration')
plt.ylabel('Clock [cyc]')

print(blocks_sm)
plt.show()
