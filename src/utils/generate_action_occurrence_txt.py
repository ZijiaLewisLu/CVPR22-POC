import numpy as np
from glob import glob
from .dataset import load_action_mapping

label2index, index2label = load_action_mapping('./dataset/MC2/mapping.txt')
files = './dataset/MC2/groundTruth/*.txt'
files = glob(files)

count = {}

for fname in files:
    with open(fname) as fp:
        labels = fp.read().split('\n')[:-1]

    action_set = list(set(labels))
    for x in action_set:
        if x not in count:
            count[x] = 0
        count[x] = count[x] + 1
        
count = { x: float(i) / len(files) for x, i in count.items() }

freq = np.array([ count[index2label[i]] for i in range(len(index2label)) ])
np.savetxt('./dataset/MC2/action_occurence.txt', freq)


