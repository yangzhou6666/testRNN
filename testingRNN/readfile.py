import numpy as np
from scipy import io

lines = open('log_folder/record.txt','r').readlines()
nc = []
cc = []
mc = []
sq_p = []
sq_n = []
for i, line in enumerate(lines):
    parts = line.split(' ')
    if parts[0] == 'cell':
        cc.append(float(parts[2]))
    elif parts[0] == 'neuron':
        nc.append(float(parts[2]))
    elif parts[0] == 'gate':
        mc.append(float(parts[2]))
    elif parts[0] == 'positive':
        sq_p.append(float(parts[3]))
    elif parts[0] == 'negative':
        sq_n.append(float(parts[3]))

nc = np.array(nc)
cc = np.array(cc)
mc = np.array(mc)
sq_p = np.array(sq_p)
sq_n = np.array(sq_n)
io.savemat('log_folder/coverage_count_NC.mat', {'coverage_count_NC': cc})
io.savemat('log_folder/coverage_count_CC.mat', {'coverage_count_CC': cc})
io.savemat('log_folder/coverage_count_MC.mat', {'coverage_count_MC': mc})
io.savemat('log_folder/coverage_count_SQ_P.mat', {'coverage_count_SQ_P': sq_p})
io.savemat('log_folder/coverage_count_SQ_N.mat', {'coverage_count_SQ_N': sq_n})

