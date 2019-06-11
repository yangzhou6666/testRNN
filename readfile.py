import numpy as np
import scipy.io as scio
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Coverage Updating Plot')
parser.add_argument('--output', dest='filename', default='./log_folder/record.txt', help='')
parser.add_argument('--metrcis', dest='metrics', default='all', help='')
args=parser.parse_args()

filename = args.filename
metrics = args.metrics


lines = open(filename,'r').readlines()
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
# io.savemat('log_folder/coverage_count_NC.mat', {'coverage_count_NC': nc})
# io.savemat('log_folder/coverage_count_CC.mat', {'coverage_count_CC': cc})
# io.savemat('log_folder/coverage_count_MC.mat', {'coverage_count_MC': mc})
# io.savemat('log_folder/coverage_count_SQ_P.mat', {'coverage_count_SQ_P': sq_p})
# io.savemat('log_folder/coverage_count_SQ_N.mat', {'coverage_count_SQ_N': sq_n})

if metrics == 'CC' :
    plt.plot(cc)
    plt.xlabel('Test Cases')
    plt.ylabel('Coverage Rate')
    plt.title('Cell Coverage Updating')
    plt.savefig("log_folder/cc.jpg")
elif metrics == 'NC' :
    plt.plot(nc)
    plt.xlabel('Test Cases')
    plt.ylabel('Coverage Rate')
    plt.title('Neuron Coverage Updating')
    plt.savefig("log_folder/nc.jpg")
elif metrics == 'GC' :
    plt.plot(mc)
    plt.xlabel('Test Cases')
    plt.ylabel('Coverage Rate')
    plt.title('Gate Coverage Updating')
    plt.savefig("log_folder/gc.jpg")
elif metrics == 'SQP' :
    plt.plot(sq_p)
    plt.xlabel('Test Cases')
    plt.ylabel('Coverage Rate')
    plt.title('Positive Sequence Coverage Updating')
    plt.savefig("log_folder/sqp.jpg")

elif metrics == 'SQN' :
    plt.plot(sq_n)
    plt.xlabel('Test Cases')
    plt.ylabel('Coverage Rate')
    plt.title('Negative Sequence Coverage Updating')
    plt.savefig("log_folder/sqn.jpg")

elif metrics == 'all' :
    plt.figure()
    plt.plot(cc)
    plt.xlabel('Test Cases')
    plt.ylabel('Coverage Rate')
    plt.title('Cell Coverage Updating')
    plt.savefig("log_folder/cc.jpg")

    plt.figure()
    plt.plot(nc)
    plt.xlabel('Test Cases')
    plt.ylabel('Coverage Rate')
    plt.title('Neuron Coverage Updating')
    plt.savefig("log_folder/nc.jpg")

    plt.figure()
    plt.plot(mc)
    plt.xlabel('Test Cases')
    plt.ylabel('Coverage Rate')
    plt.title('Gate Coverage Updating')
    plt.savefig("log_folder/gc.jpg")

    plt.figure()
    plt.plot(sq_p)
    plt.xlabel('Test Cases')
    plt.ylabel('Coverage Rate')
    plt.title('Positive Sequence Coverage Updating')
    plt.savefig("log_folder/sqp.jpg")

    plt.figure()
    plt.plot(sq_n)
    plt.xlabel('Test Cases')
    plt.ylabel('Coverage Rate')
    plt.title('Negative Sequence Coverage Updating')
    plt.savefig("log_folder/sqn.jpg")

else :
    print("Please specify a metrics to plot {NC, CC, GC, SQP, SQN, all}")

data1 = scio.loadmat('./log_folder/feature_count_CC.mat')
feature_count_CC = data1['feature_count_CC']
plt.figure()
plt.bar(range(len(feature_count_CC[0])),feature_count_CC[0])
plt.xlabel('Test Conditions')
plt.ylabel('Covering Times')
plt.title('Cell Coverage Test Conditions Count')
plt.savefig("log_folder/feature_count_cc.jpg")

data2 = scio.loadmat('./log_folder/feature_count_GC.mat')
feature_count_GC = data2['feature_count_GC']
plt.figure()
plt.bar(range(len(feature_count_GC[0])),feature_count_GC[0])
plt.xlabel('Test Conditions')
plt.ylabel('Covering Times')
plt.title('Gate Coverage Test Conditions Count')
plt.savefig("log_folder/feature_count_gc.jpg")
