#!/usr/bin/python3
from subprocess import Popen, PIPE
import json
import statistics as st
import sys
import pathlib
import shutil

outpath = "out/"
pathlib.Path(outpath).mkdir(parents=True, exist_ok=True)

#./2DConvolution <#threads> <#blocks> <# kernel> <# of intervals> <ni> <nj> <interference method (0rnd or 1seq) <usePREM> <output JSON file name>
scenarios = [# Filename                                     thread   block   kernel  ni    nj    usePREM  repetition USE_SCHED  KERNELWISE_SYNC  SCHED_ALL_PHASES   PREM_PROF  PF_OFF   WB_OFF
# # Experiment block sched
#    ("1024t-1b-1k-1024-legacy.json",                         1024,    1,      1,      1026, 1022, 0,        100,       'no',      'yes',           'no',           'no',        '0',    '0'),
#
# # Full tile schedule experiment
#    ("512t-2b-1k-1024-legacy.json",                           512,    2,      1,      1026, 1022,  0,       1000,       'no',      'yes',           'no',           'no',       '0',      '0'),
#    ("512t-2b-4k-1024-legacy.json",                           512,    2,      4,      1026, 1022,  0,       1000,       'no',      'yes',           'no',           'no',       '0',      '0'),
#    ("512t-2b-4k-1024-prem-nosched.json",                     512,    2,      4,      1026, 1022,  1,       1000,       'no',      'yes',           'no',           'no',       '0',      '0'),
#    ("512t-2b-4k-1024-prem-nosched-prof.json",                512,    2,      4,      1026, 1022,  1,       1,          'no',      'yes',           'no',           'yes',       '0',      '0'),
#    ("512t-2b-4k-1024-prem-kernelsched-0pfo.json",            512,    2,      4,      1026, 1022,  1,       1000,       'yes',     'yes',           'no',           'no',       '0',      '0'),
#    ("512t-2b-4k-1024-prem-kernelsched-500pfo.json",          512,    2,      4,      1026, 1022,  1,       1000,       'yes',     'yes',           'no',           'no',       '500',    '0'),
#    ("512t-2b-4k-1024-prem-kernelsched-1000pfo.json",         512,    2,      4,      1026, 1022,  1,       1000,       'yes',     'yes',           'no',           'no',       '1000',   '0'),
#    ("512t-2b-4k-1024-prem-kernelsched-1100pfo.json",         512,    2,      4,      1026, 1022,  1,       1000,       'yes',     'yes',           'no',           'no',       '1100',   '0'),
#    ("512t-2b-4k-1024-prem-kernelsched-1200pfo.json",         512,    2,      4,      1026, 1022,  1,       1000,       'yes',     'yes',           'no',           'no',       '1200',   '0'),
#    ("512t-2b-4k-1024-prem-kernelsched-1300pfo.json",         512,    2,      4,      1026, 1022,  1,       1000,       'yes',     'yes',           'no',           'no',       '1300',   '0'),
#    ("512t-2b-4k-1024-prem-kernelsched-1400pfo.json",         512,    2,      4,      1026, 1022,  1,       1000,       'yes',     'yes',           'no',           'no',       '1400',   '0'),
#    ("512t-2b-4k-1024-prem-kernelsched-1500pfo.json",         512,    2,      4,      1026, 1022,  1,       1000,       'yes',     'yes',           'no',           'no',       '1500',   '0'),
#    ("512t-2b-4k-1024-prem-kernelsched-1600pfo.json",         512,    2,      4,      1026, 1022,  1,       1000,       'yes',     'yes',           'no',           'no',       '1600',   '0'),
#    ("512t-2b-4k-1024-prem-kernelsched-1700pfo.json",         512,    2,      4,      1026, 1022,  1,       1000,       'yes',     'yes',           'no',           'no',       '1700',   '0'),
#    ("512t-2b-4k-1024-prem-kernelsched-2500pfo.json",         512,    2,      4,      1026, 1022,  1,       1000,       'yes',     'yes',           'no',           'no',       '2500',   '0'),
#
#    ("512t-2b-4k-1024-prem-tilesched-0pfo.json",              512,    2,      4,      1026, 1022,  1,       100,       'yes',      'no',           'no',            'no',       '0',      '0'),
#    ("512t-2b-4k-1024-prem-tilesched-500pfo.json",            512,    2,      4,      1026, 1022,  1,       100,       'yes',      'no',           'no',            'no',       '500',    '0'),
#    ("512t-2b-4k-1024-prem-tilesched-1000pfo.json",           512,    2,      4,      1026, 1022,  1,       100,       'yes',      'no',           'no',            'no',       '1000',   '0'),
#    ("512t-2b-4k-1024-prem-tilesched-1200pfo.json",           512,    2,      4,      1026, 1022,  1,       100,       'yes',      'no',           'no',            'no',       '1200',   '0'),
#    ("512t-2b-4k-1024-prem-tilesched-1400pfo.json",           512,    2,      4,      1026, 1022,  1,       100,       'yes',      'no',           'no',            'no',       '1400',   '0'),
#    ("512t-2b-4k-1024-prem-tilesched-1600pfo.json",           512,    2,      4,      1026, 1022,  1,       100,       'yes',      'no',           'no',            'no',       '1600',   '0'),
#    ("512t-2b-4k-1024-prem-tilesched-1800pfo.json",           512,    2,      4,      1026, 1022,  1,       100,       'yes',      'no',           'no',            'no',       '1800',   '0'),
#    ("512t-2b-4k-1024-prem-tilesched-2000pfo.json",           512,    2,      4,      1026, 1022,  1,       100,       'yes',      'no',           'no',            'no',       '2000',   '0'),
#    ("512t-2b-4k-1024-prem-tilesched-2200pfo.json",           512,    2,      4,      1026, 1022,  1,       100,       'yes',      'no',           'no',            'no',       '2200',   '0'),
#    ("512t-2b-4k-1024-prem-tilesched-2500pfo.json",           512,    2,      4,      1026, 1022,  1,       100,       'yes',      'no',           'no',            'no',       '2500',   '0'),

# Prefetch schedule experiment
    # make USE_SCHED=yes KERNELWISE_SYNC=yes SCHED_ALL_PHASES=yes PF_PHASEOFFSET=0 WB_PHASEOFFSET=3000
    # make USE_SCHED=yes KERNELWISE_SYNC=yes SCHED_ALL_PHASES=yes PF_PHASEOFFSET=3000 WB_PHASEOFFSET=3000
#    ("512t-2b-4k-1024-prem-kernelsched-0pfo-3000wbo.json",    512,    2,      4,      1026, 1022,  1,       1,       'yes',     'yes',           'yes',           'yes',       '0',  '3000'),
#    ("512t-2b-4k-1024-prem-kernelsched-500pfo-3000wbo.json",  512,    2,      4,      1026, 1022,  1,       1,       'yes',     'yes',           'yes',           'yes',     '500',  '3000'),
#    ("512t-2b-4k-1024-prem-kernelsched-1000pfo-3000wbo.json", 512,    2,      4,      1026, 1022,  1,       1,       'yes',     'yes',           'yes',           'yes',    '1000',  '3000'),
#    ("512t-2b-4k-1024-prem-kernelsched-1500pfo-3000wbo.json", 512,    2,      4,      1026, 1022,  1,       1,       'yes',     'yes',           'yes',           'yes',    '1500',  '3000'),
#    ("512t-2b-4k-1024-prem-kernelsched-2000pfo-3000wbo.json", 512,    2,      4,      1026, 1022,  1,       1,       'yes',     'yes',           'yes',           'yes',    '2000',  '3000'),
#    ("512t-2b-4k-1024-prem-kernelsched-2500pfo-3000wbo.json", 512,    2,      4,      1026, 1022,  1,       1,       'yes',     'yes',           'yes',           'yes',    '2500',  '3000'),
#    ("512t-2b-4k-1024-prem-kernelsched-3000pfo-3000wbo.json", 512,    2,      4,      1026, 1022,  1,       1,       'yes',     'yes',           'yes',           'yes',    '3000',  '3000'),
#
#    ("512t-2b-4k-1024-prem-tilesched-0pfo-3000wbo.json",      512,    2,      4,      1026, 1022,  1,       1,       'yes',      'no',           'yes',           'yes',       '0',  '3000'),
#    ("512t-2b-4k-1024-prem-tilesched-500pfo-3000wbo.json",    512,    2,      4,      1026, 1022,  1,       1,       'yes',      'no',           'yes',           'yes',     '500',  '3000'),
#    ("512t-2b-4k-1024-prem-tilesched-1000pfo-3000wbo.json",   512,    2,      4,      1026, 1022,  1,       1,       'yes',      'no',           'yes',           'yes',    '1000',  '3000'),
#    ("512t-2b-4k-1024-prem-tilesched-1500pfo-3000wbo.json",   512,    2,      4,      1026, 1022,  1,       1,       'yes',      'no',           'yes',           'yes',    '1500',  '3000'),
#    ("512t-2b-4k-1024-prem-tilesched-2000pfo-3000wbo.json",   512,    2,      4,      1026, 1022,  1,       1,       'yes',      'no',           'yes',           'yes',    '2000',  '3000'),
#    ("512t-2b-4k-1024-prem-tilesched-2500pfo-3000wbo.json",   512,    2,      4,      1026, 1022,  1,       1,       'yes',      'no',           'yes',           'yes',    '2500',  '3000'),
#    ("512t-2b-4k-1024-prem-tilesched-3000pfo-3000wbo.json",   512,    2,      4,      1026, 1022,  1,       1,       'yes',      'no',           'yes',           'yes',    '3000',  '3000'),
#
## Writeback schedule experiment
#    # make USE_SCHED=yes KERNELWISE_SYNC=yes SCHED_ALL_PHASES=yes PF_PHASEOFFSET=3000 WB_PHASEOFFSET=0
#    # make USE_SCHED=yes KERNELWISE_SYNC=yes SCHED_ALL_PHASES=yes PF_PHASEOFFSET=3000 WB_PHASEOFFSET=1000
#    ("512t-2b-4k-1024-prem-kernelsched-3000pfo-0wbo.json",    512,    2,      4,      1026, 1022,  1,       1,       'yes',     'yes',           'yes',           'yes',    '3000',    '0'),
#    ("512t-2b-4k-1024-prem-kernelsched-3000pfo-200wbo.json",  512,    2,      4,      1026, 1022,  1,       1,       'yes',     'yes',           'yes',           'yes',    '3000',  '200'),
#    ("512t-2b-4k-1024-prem-kernelsched-3000pfo-400wbo.json",  512,    2,      4,      1026, 1022,  1,       1,       'yes',     'yes',           'yes',           'yes',    '3000',  '400'),
#    ("512t-2b-4k-1024-prem-kernelsched-3000pfo-600wbo.json",  512,    2,      4,      1026, 1022,  1,       1,       'yes',     'yes',           'yes',           'yes',    '3000',  '600'),
#    ("512t-2b-4k-1024-prem-kernelsched-3000pfo-800wbo.json",  512,    2,      4,      1026, 1022,  1,       1,       'yes',     'yes',           'yes',           'yes',    '3000',  '800'),
#    ("512t-2b-4k-1024-prem-kernelsched-3000pfo-1000wbo.json", 512,    2,      4,      1026, 1022,  1,       1,       'yes',     'yes',           'yes',           'yes',    '3000', '1000'),
#
#    ("512t-2b-4k-1024-prem-tilesched-3000pfo-0wbo.json",      512,    2,      4,      1026, 1022,  1,       1,       'yes',      'no',           'yes',           'yes',    '3000',    '0'),
#    ("512t-2b-4k-1024-prem-tilesched-3000pfo-200wbo.json",    512,    2,      4,      1026, 1022,  1,       1,       'yes',      'no',           'yes',           'yes',    '3000',  '200'),
#    ("512t-2b-4k-1024-prem-tilesched-3000pfo-400wbo.json",    512,    2,      4,      1026, 1022,  1,       1,       'yes',      'no',           'yes',           'yes',    '3000',  '400'),
#    ("512t-2b-4k-1024-prem-tilesched-3000pfo-600wbo.json",    512,    2,      4,      1026, 1022,  1,       1,       'yes',      'no',           'yes',           'yes',    '3000',  '600'),
#    ("512t-2b-4k-1024-prem-tilesched-3000pfo-800wbo.json",    512,    2,      4,      1026, 1022,  1,       1,       'yes',      'no',           'yes',           'yes',    '3000',  '800'),
#    ("512t-2b-4k-1024-prem-tilesched-3000pfo-1000wbo.json",   512,    2,      4,      1026, 1022,  1,       1,       'yes',      'no',           'yes',           'yes',    '3000', '1000'),
             ]

             
for scenario in scenarios:
    filename       = scenario[0]
    nof_thread     = scenario[1]
    nof_block      = scenario[2]
    nof_kernel     = scenario[3]
    ni             = scenario[4]
    nj             = scenario[5]
    usePREM        = scenario[6]
    nof_rep        = scenario[7]
    sched          = scenario[8]
    kernelWiseSync = scenario[9]
    schedall       = scenario[10]
    premprof       =scenario[11]
    PFOff          = scenario[12]
    WBOff          = scenario[13]

    #compile
    makeprocess = Popen(["make", "clean"], stdout=PIPE)
    output = makeprocess.communicate()
    print(output)

# make USE_SCHED=yes KERNELWISE_SYNC=yes SCHED_ALL_PHASES=no PF_PHASEOFFSET=3000 WB_PHASEOFFSET=0
    makeprocess = Popen(["make", "USE_SCHED="+sched, "KERNELWISE_SYNC="+kernelWiseSync, "SCHED_ALL_PHASES="+schedall, "PREM_PROF="+premprof, "PF_PHASEOFFSET="+PFOff, "WB_PHASEOFFSET="+WBOff], stdout=PIPE)
    output = makeprocess.communicate()
    print(output)


    print("-------------------------------")
    print("ni/nj: "+str(ni)+"/"+str(nj))
    print("Nof threads: "+str(nof_thread))
    print("Number of blocks: "+str(nof_block))
    print("Number of kernel: "+str(nof_kernel))
    print("-------------------------------")

    process = Popen(["./2DConvolution", str(nof_thread), str(nof_block),str(nof_kernel), str(nof_rep), str(ni) , str(nj), str(usePREM), "out.json"], stdout=PIPE)
    output = process.communicate()
    print(output)

    # Copy the output file
    shutil.move('out.json', outpath+filename) # complete target filename given
