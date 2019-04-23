#!/usr/bin/python3
import json
import matplotlib.pyplot as plt
import numpy as np
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

filename1 = "out-used-threads/1-cg-kernels-different-elem.json"
filename2 = "out-used-threads/2-cg-kernels-different-elem.json"
filename3 = "out-used-threads/4-cg-kernels-different-elem.json"
filename4 = "out-used-threads/12-cg-kernels-different-elem.json"
filename5 = "out-used-threads/16-cg-kernels-different-elem.json"
filename6 = "out-used-threads/64-cg-kernels-different-elem.json"
filename7 = "out-used-threads/128-cg-kernels-different-elem.json"
filename8 = "out-used-threads/256-cg-kernels-different-elem.json"


with open(filename1) as f1:
    data1 = json.load(f1)
with open(filename2) as f2:
    data2 = json.load(f2)
with open(filename3) as f3:
    data3 = json.load(f3)
with open(filename4) as f4:
    data4 = json.load(f4)
with open(filename5) as f5:
    data5 = json.load(f5)
with open(filename6) as f6:
    data6 = json.load(f6)
with open(filename7) as f7:
    data7 = json.load(f7)
with open(filename8) as f8:
    data8 = json.load(f8)

d1 = np.array(data1['mean'])
d2 = np.array(data2['mean'])
d3 = np.array(data3['mean'])
d4 = np.array(data4['mean'])
d5 = np.array(data5['mean'])
d6 = np.array(data6['mean'])
d7 = np.array(data7['mean'])
d8 = np.array(data8['mean'])

scale = d1[0]
d1 = d1/scale
d2 = d2/scale
d3 = d3/scale
d4 = d4/scale
d5 = d5/scale
d6 = d6/scale
d7 = d7/scale
d8 = d8/scale

min1 = np.array(data1['min'])/scale
min2 = np.array(data2['min'])/scale
min3 = np.array(data3['min'])/scale
min4 = np.array(data4['min'])/scale
min5 = np.array(data5['min'])/scale
min6 = np.array(data6['min'])/scale
min7 = np.array(data7['min'])/scale
min8 = np.array(data8['min'])/scale

max1 = np.array(data1['max'])/scale
max2 = np.array(data2['max'])/scale
max3 = np.array(data3['max'])/scale
max4 = np.array(data4['max'])/scale
max5 = np.array(data5['max'])/scale
max6 = np.array(data6['max'])/scale
max7 = np.array(data7['max'])/scale
max8 = np.array(data8['max'])/scale

# red dashes, blue squares and green triangles
fig = plt.figure(figsize=[6,3])
#plt.plot(data1['nof_threads'], d1,'r', label='1k')
plt.plot(data2['nof_threads'], d2,'g', linewidth=0.75, label='2k')
plt.plot(data3['nof_threads'], d3,'b', linewidth=0.75, label='3k')
plt.plot(data4['nof_threads'], d4,'c', linewidth=0.75, label='12k')
plt.plot(data5['nof_threads'], d5,'m', linewidth=0.75, label='16k')
plt.plot(data6['nof_threads'], d6,'k', linewidth=0.75, label='64k')
plt.plot(data7['nof_threads'], d7,'y', linewidth=0.75, label='128k')
plt.plot(data8['nof_threads'], d8,'k', linewidth=0.75, label='256k')
handles, labels = plt.gca().get_legend_handles_labels()
plt.gca().legend(handles[::-1], labels[::-1])
#plt.fill_between(data1['nof_threads'], min1, max1, facecolor='red', alpha=0.5)
plt.fill_between(data2['nof_threads'], min2, max2, facecolor='green',   alpha=0.4)
plt.fill_between(data3['nof_threads'], min3, max3, facecolor='blue',    alpha=0.4)
plt.fill_between(data4['nof_threads'], min4, max4, facecolor='cyan',    alpha=0.4)
plt.fill_between(data5['nof_threads'], min5, max5, facecolor='magenta', alpha=0.4)
plt.fill_between(data6['nof_threads'], min6, max6, facecolor='black',   alpha=0.4)
plt.fill_between(data7['nof_threads'], min7, max7, facecolor='yellow',  alpha=0.4)
plt.fill_between(data8['nof_threads'], min8, max8, facecolor='black',   alpha=0.4)
plt.xticks(np.arange(0, 1025, step=64), rotation=90)
plt.grid(True, which="both")
plt.title('Random walk - L2')
plt.xlabel('Number of threads')
plt.ylabel('Slow-down')

#plt.figure()
#plt.plot(data1['nof_threads'], max3-min3,'r')
#plt.xticks(np.arange(0, 1025, step=32),[100,25,50,75,100,25,50,75,100,25,50,75,100,25,50,75,100,25,50,75,100,25,50,75,100,25,50,75,100,25,50,75,100])
#plt.grid(True, which="both")
#plt.title('256kB, max-min')
#plt.xlabel('Nof Threads')
plt.tight_layout()
fig.savefig('rand-walk-threads.pdf', format='pdf')
plt.show()
