#!/usr/bin/python3
from subprocess import Popen, PIPE
import json
import statistics as st
import sys
import pathlib
import shutil

outpath = "out/"
pathlib.Path(outpath).mkdir(parents=True, exist_ok=True)

inter = "2" # 0 = rnd, 1= seq

#./2DConvolution <#threads> <#blocks> <# kernel> <# of intervals> <ni> <nj> <interference method (0rnd or 1seq) <usePREM> <output JSON file name>
scenarios = [# Filename                thread   block   kernel     ni      nj   usePREM  repetition
             ("1024t-1b-1k-1024-legacy.json",  1024,     1,     1,      1026,    1022,   0,       100),
             ("512t-1b-1k-1024-legacy.json",    512,     1,     1,      1026,    1022,   0,       100),
             ("512t-1b-2k-1024-legacy.json",    512,     1,     2,      1026,    1022,   0,       100),
             ("512t-2b-1k-1024-legacy.json",    512,     2,     1,      1026,    1022,   0,       100),
             ("512t-2b-2k-1024-legacy.json",    512,     2,     2,      1026,    1022,   0,       100),
             ("512t-2b-4k-1024-legacy.json",    512,     2,     4,      1026,    1022,   0,       100),

             ("512t-1b-1k-1024-prem.json",    512,     1,     1,      1026,    1022,   1,       100),
             ("512t-1b-2k-1024-prem.json",    512,     1,     2,      1026,    1022,   1,       100),
             ("512t-2b-1k-1024-prem.json",    512,     2,     1,      1026,    1022,   1,       100),
             ("512t-2b-2k-1024-prem.json",    512,     2,     2,      1026,    1022,   1,       100),
             ("512t-2b-4k-1024-prem.json",    512,     2,     4,      1026,    1022,   1,       100),

#             ("512t-1b-1k-512-legacy.json",     512,     1,     1,       512,     512,   0,       100),
#             ("256t-1b-1k-512-legacy.json",     256,     1,     1,       512,     512,   0,       100),
#             ("256t-1b-2k-512-legacy.json",     256,     1,     2,       512,     512,   0,       100),
#             ("256t-2b-1k-512-legacy.json",     256,     2,     1,       512,     512,   0,       100),
#             ("256t-2b-2k-512-legacy.json",     256,     2,     2,       512,     512,   0,       100),
#             ("256t-2b-4k-512-legacy.json",     256,     2,     4,       512,     512,   0,       100),
            # ("512t-2b-4k-4096-legacy.json",    512,     2,     4,      4090,    4096,   100),
             ]

             
for scenario in scenarios:
    filename = scenario[0]
    nof_thread = scenario[1]
    nof_block = scenario[2]
    nof_kernel = scenario[3]
    ni = scenario[4]
    nj = scenario[5]
    usePREM = scenario[6]
    nof_rep = scenario[7]

    print("-------------------------------")
    print("ni/nj: "+str(ni)+"/"+str(nj))
    print("Nof threads: "+str(nof_thread))
    print("Number of blocks: "+str(nof_block))
    print("Number of kernel: "+str(nof_kernel))
    print("-------------------------------")
    process = Popen(["./2DConvolution", str(nof_thread), str(nof_block),str(nof_kernel), str(nof_rep), str(ni) , str(nj), inter, str(usePREM), "out.json"], stdout=PIPE)
    output = process.communicate()
    print(output)

    # Copy the output file
    shutil.move('out.json', outpath+filename) # complete target filename given
