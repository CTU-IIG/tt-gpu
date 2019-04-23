#!/usr/bin/python3
import json
import matplotlib.pyplot as plt
import numpy as np

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

clk = 1300500000

filename1 = "out/512-cg-kernels-1thread-same-elem.json"
filename2 = "out/512-cg-kernels-32thread-same-elem.json"
filename3 = "out/512-cg-kernels-128thread-same-elem.json"
filename4 = "out/256-cg-kernels-1thread-same-elem.json"
filename5 = "out/256-cg-kernels-32thread-same-elem.json"
filename6 = "out/256-cg-kernels-128thread-same-elem.json"
filename7 = "out/128-cg-kernels-1thread-same-elem.json"
filename8 = "out/128-cg-kernels-32thread-same-elem.json"
filename9 = "out/128-cg-kernels-128thread-same-elem.json"

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

dk1 = np.array(data1['nof_kernels'])*1*1e-9 
dk2 = np.array(data2['nof_kernels'])*1*1e-9
dk3 = np.array(data3['nof_kernels'])*1*1e-9
dk4 = np.array(data4['nof_kernels'])*1*1e-9
dk5 = np.array(data5['nof_kernels'])*1*1e-9
dk6 = np.array(data6['nof_kernels'])*1*1e-9
dk7 = np.array(data7['nof_kernels'])*1*1e-9
dk8 = np.array(data8['nof_kernels'])*1*1e-9
dk9 = np.array(data9['nof_kernels'])*1*1e-9

scale = clk#d2[0]
print("Scale: {:f}".format(scale))
d1 = (scale/d1)/4*dk1 
d2 = (scale/d2)/4*dk2 
d3 = (scale/d3)/4*dk3
d4 = (scale/d4)/4*dk4
d5 = (scale/d5)/4*dk5
d6 = (scale/d6)/4*dk6
d7 = (scale/d7)/4*dk7
d8 = (scale/d8)/4*dk8
d9 = (scale/d9)/4*dk9

min1 = scale/np.array(data1['min'])/4*dk1 
min2 = scale/np.array(data2['min'])/4*dk2
min3 = scale/np.array(data3['min'])/4*dk3
min4 = scale/np.array(data4['min'])/4*dk4
min5 = scale/np.array(data5['min'])/4*dk5
min6 = scale/np.array(data6['min'])/4*dk6
min7 = scale/np.array(data7['min'])/4*dk7
min8 = scale/np.array(data8['min'])/4*dk8
min9 = scale/np.array(data9['min'])/4*dk9

max1 = scale/np.array(data1['max'])/4*dk1
max2 = scale/np.array(data2['max'])/4*dk2
max3 = scale/np.array(data3['max'])/4*dk3
max4 = scale/np.array(data4['max'])/4*dk4
max5 = scale/np.array(data5['max'])/4*dk5
max6 = scale/np.array(data6['max'])/4*dk6
max7 = scale/np.array(data7['max'])/4*dk7
max8 = scale/np.array(data8['max'])/4*dk8
max9 = scale/np.array(data9['max'])/4*dk9

fig = plt.figure(figsize=[7,3])

# red dashes, blue squares and green triangles
plt.plot(data1['nof_kernels'], d1,'r',marker='1')
plt.plot(data2['nof_kernels'], d2,'r',marker='2')
plt.plot(data3['nof_kernels'], d3,'b',marker='3')
plt.plot(data4['nof_kernels'], d4,'c',marker='4')
plt.plot(data5['nof_kernels'], d5,'b',marker='*')
plt.plot(data6['nof_kernels'], d6,'k',marker='+')
plt.plot(data7['nof_kernels'], d7,'y',marker='x')
plt.plot(data8['nof_kernels'], d8,'g',marker='|')
plt.plot(data9['nof_kernels'], d9,'brown',marker='v')

plt.gca().legend((
                  '512kB, 1 threads',\
                  '512kB, 32 threads per kernel',\
                  '512kB, 128 threads',\
                  '256kB, 1 threads',\
                  '256kB, 32 threads per kernel',\
                  '256kB, 128 threads',\
                  '128kB, 1 threads',\
                  '128kB, 32 threads per kernel',\
                  '128kB, 128 threads',
                 ), loc='lower right')

plt.fill_between(data1['nof_kernels'], min1, max1, facecolor='red', alpha=0.5)
plt.fill_between(data2['nof_kernels'], min2, max2, facecolor='r', alpha=0.5)
plt.fill_between(data3['nof_kernels'], min3, max3, facecolor='blue', alpha=0.5)
plt.fill_between(data4['nof_kernels'], min4, max4, facecolor='cyan', alpha=0.5)
plt.fill_between(data5['nof_kernels'], min5, max5, facecolor='b', alpha=0.5)
plt.fill_between(data6['nof_kernels'], min6, max6, facecolor='black', alpha=0.5)
plt.fill_between(data7['nof_kernels'], min7, max7, facecolor='yellow', alpha=0.5)
plt.fill_between(data8['nof_kernels'], min8, max8, facecolor='g', alpha=0.5)
plt.fill_between(data9['nof_kernels'], min9, max9, facecolor='brown', alpha=0.5)

plt.grid(True, which="both")
plt.title('Sequential walk with multiple kernels - no CPU interference')
plt.xlabel('Number of parallel Kernels')
plt.xticks(range(0,17))
plt.xlim([0.5,16.5])
plt.ylabel('Slowdown')
fig.savefig('seq-walk-mult-kernel-noint.pdf', format='pdf',bbox_inches='tight')
plt.show()
