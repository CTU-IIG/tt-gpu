#!/usr/bin/python3
import json
import matplotlib.pyplot as plt
import numpy as np

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

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
d1 = np.array(data1['mean'])
d2 = np.array(data2['mean'])
d3 = np.array(data3['mean'])
d4 = np.array(data4['mean'])
d5 = np.array(data5['mean'])
d6 = np.array(data6['mean'])
d7 = np.array(data7['mean'])
d8 = np.array(data8['mean'])
d9 = np.array(data9['mean'])

scale = d2[0]
print("Scale: {:f}".format(scale))
d1 = d1/scale
d2 = d2/scale
d3 = d3/scale
d4 = d4/scale
d5 = d5/scale
d6 = d6/scale
d7 = d7/scale
d8 = d8/scale
d9 = d9/scale

min1 = np.array(data1['min'])/scale
min2 = np.array(data2['min'])/scale
min3 = np.array(data3['min'])/scale
min4 = np.array(data4['min'])/scale
min5 = np.array(data5['min'])/scale
min6 = np.array(data6['min'])/scale
min7 = np.array(data7['min'])/scale
min8 = np.array(data8['min'])/scale
min9 = np.array(data9['min'])/scale

max1 = np.array(data1['max'])/scale
max2 = np.array(data2['max'])/scale
max3 = np.array(data3['max'])/scale
max4 = np.array(data4['max'])/scale
max5 = np.array(data5['max'])/scale
max6 = np.array(data6['max'])/scale
max7 = np.array(data7['max'])/scale
max8 = np.array(data8['max'])/scale
max9 = np.array(data9['max'])/scale

fig = plt.figure(figsize=[7,3])

# red dashes, blue squares and green triangles
#plt.plot(data1['nof_kernels'], d1,'r',marker='1')
plt.plot(data2['nof_kernels'], d2,'r',marker='2')
#plt.plot(data3['nof_kernels'], d3,'b',marker='3')
#plt.plot(data4['nof_kernels'], d4,'c',marker='4')
plt.plot(data5['nof_kernels'], d5,'b',marker='*')
#plt.plot(data6['nof_kernels'], d6,'k',marker='+')
#plt.plot(data7['nof_kernels'], d7,'y',marker='x')
plt.plot(data8['nof_kernels'], d8,'g',marker='|')
#plt.plot(data9['nof_kernels'], d9,'brown',marker='v')

plt.gca().legend((
                 # '512kB, 1 threads',\
                  '512kB, 32 threads per kernel',\
                 # '512kB, 128 threads',\
                 # '256kB, 1 threads',\
                  '256kB, 32 threads per kernel',\
                 # '256kB, 128 threads',\
                 # '128kB, 1 threads',\
                  '128kB, 32 threads per kernel',\
                 # '128kB, 128 threads',
                 ), loc='lower right')

#plt.fill_between(data1['nof_kernels'], min1, max1, facecolor='red', alpha=0.5)
plt.fill_between(data2['nof_kernels'], min2, max2, facecolor='r', alpha=0.5)
#plt.fill_between(data3['nof_kernels'], min3, max3, facecolor='blue', alpha=0.5)
#plt.fill_between(data4['nof_kernels'], min4, max4, facecolor='cyan', alpha=0.5)
plt.fill_between(data5['nof_kernels'], min5, max5, facecolor='b', alpha=0.5)
#plt.fill_between(data6['nof_kernels'], min6, max6, facecolor='black', alpha=0.5)
#plt.fill_between(data7['nof_kernels'], min7, max7, facecolor='yellow', alpha=0.5)
plt.fill_between(data8['nof_kernels'], min8, max8, facecolor='g', alpha=0.5)
#plt.fill_between(data9['nof_kernels'], min9, max9, facecolor='brown', alpha=0.5)

plt.grid(True, which="both")
plt.title('Sequential walk with multiple kernels - random CPU interference')
plt.xlabel('Number of parallel Kernels')
plt.xticks(range(0,17))
plt.xlim([0.5,16.5])
plt.ylabel('Slowdown')
fig.savefig('seq-walk-mult-kernel-randint.pdf', format='pdf',bbox_inches='tight')
plt.show()
