import numpy
import os
import json

with open('epnet/data/wireframe/test.json','r') as buffer:
    annotations = json.load(buffer)

n_lines = 0

for anno in annotations:
    n_lines += len(anno['lines'])


n_lines/=len(annotations)

print(n_lines)
# import pdb
# pdb.set_trace()