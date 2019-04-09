#!/usr/bin/python3
import json
import matplotlib.pyplot as plt
import numpy as np
import itertools

file1='out/roundtrip.json'

with open(file1) as f1:
    data1 = json.load(f1)



times = np.array(data1['times']).astype(int)
nofPasses = int(data1['nofpasses'])
nofkernel = int(data1['nofkernel'])

times1 = times[1:nofPasses]
times2 = times[nofPasses+1:]

oneWay1 = times2-times1
oneWay2 = times1[1:]-times2[:-1]
oneWay = np.concatenate((oneWay1, oneWay2))
onewayavg = np.mean(oneWay)
onewaymax =  np.max(oneWay)
onewaymin =  np.min(oneWay)

print("Oneway avg: {:f}, max: {:f} min: {:f}".format(onewayavg, onewaymax, onewaymin))

rt1 = np.diff(times1)
rt2 = np.diff(times2)
roundtrip = np.concatenate((rt1, rt2))
rtavg = np.mean(roundtrip)
rtmax = np.max(roundtrip)
rtmin = np.min(roundtrip)
print("RT avg: {:f}, max: {:f} min: {:f}".format(rtavg, rtmax, rtmin))

y = [onewayavg,rtavg]
minerr = []
maxerr = []

centers = np.arange(0,2,1)
print(rtavg-rtmin)
print(rtmax-rtavg)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.bar(0, rtavg, width=0.5, yerr=[[rtavg-rtmin], [rtmax-rtavg]], alpha =0.5, hatch='/',ecolor='r', capsize=5)
ax.bar(1, onewayavg, width=0.5, yerr=[[onewayavg-onewaymin], [onewaymax-onewayavg]], alpha =0.5, hatch='\\',ecolor='b', capsize=5)
plt.show()
