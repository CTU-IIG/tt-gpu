#!/usr/bin/python3
import json
import matplotlib.pyplot as plt
import numpy as np

filename1 = "out-random/512-cg-kernels-1thread-same-elem.json"
filename2 = "out-random/512-cg-kernels-32thread-same-elem.json"
filename3 = "out-random/512-cg-kernels-128thread-same-elem.json"
filename4 = "out-random/256-cg-kernels-1thread-same-elem.json"
filename5 = "out-random/256-cg-kernels-32thread-same-elem.json"
filename6 = "out-random/256-cg-kernels-128thread-same-elem.json"
filename7 = "out-random/128-cg-kernels-1thread-same-elem.json"
filename8 = "out-random/128-cg-kernels-32thread-same-elem.json"
filename9 = "out-random/128-cg-kernels-128thread-same-elem.json"

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
with open(filename9) as f9:
    data9 = json.load(f9)

ax = plt.subplot(111)

# red dashes, blue squares and green triangles
ax.plot(data1['nof_kernels'], data1['mean'],'r',marker='1')
ax.plot(data2['nof_kernels'], data2['mean'],'g',marker='2')
ax.plot(data3['nof_kernels'], data3['mean'],'b',marker='3')
ax.plot(data4['nof_kernels'], data4['mean'],'c',marker='4')
ax.plot(data5['nof_kernels'], data5['mean'],'m',marker='*')
ax.plot(data6['nof_kernels'], data6['mean'],'k',marker='+')
ax.plot(data7['nof_kernels'], data7['mean'],'y',marker='x')
ax.plot(data8['nof_kernels'], data8['mean'],'pink',marker='|')
ax.plot(data9['nof_kernels'], data9['mean'],'brown',marker='v')

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(('512kB, 1 threads',\
                  '512kB, 32 threads',\
                  '512kB, 128 threads',\
                  '256kB, 1 threads',\
                  '256kB, 32 threads',\
                  '256kB, 128 threads',\
                  '128kB, 1 threads',\
                  '128kB, 32 threadst',\
                  '128kB, 128 threads'),loc='center left',bbox_to_anchor=(1, 0.5))
ax.fill_between(data1['nof_kernels'], data1['min'], data1['max'], facecolor='red', alpha=0.5)
ax.fill_between(data2['nof_kernels'], data2['min'], data2['max'], facecolor='green', alpha=0.5)
ax.fill_between(data3['nof_kernels'], data3['min'], data3['max'], facecolor='blue', alpha=0.5)
ax.fill_between(data4['nof_kernels'], data4['min'], data4['max'], facecolor='cyan', alpha=0.5)
ax.fill_between(data5['nof_kernels'], data5['min'], data5['max'], facecolor='magenta', alpha=0.5)
ax.fill_between(data6['nof_kernels'], data6['min'], data6['max'], facecolor='black', alpha=0.5)
ax.fill_between(data7['nof_kernels'], data7['min'], data7['max'], facecolor='yellow', alpha=0.5)
ax.fill_between(data8['nof_kernels'], data8['min'], data8['max'], facecolor='pink', alpha=0.5)
ax.fill_between(data9['nof_kernels'], data9['min'], data9['max'], facecolor='brown', alpha=0.5)

plt.grid(True, which="both")
plt.title('Sequential walk - multiple kernels, each thread same element')
plt.xlabel('Nof Kernels')
plt.ylabel('Cycles/Elem')
plt.show()
