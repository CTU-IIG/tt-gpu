from showBlocks import drawScenario
import matplotlib.pyplot as plt

title = "PREM mulitple threads: ni,nj = 1026x1022, 512 threads, 2 blocks, 4 kernel"
filename= "out/threads.json"
drawScenario(filename, title)
plt.show()
