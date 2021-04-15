import numpy as np
import json, sys, os, argparse, io
from sentence_transformers import CrossEncoder
from sentence_transformers.readers import InputExample
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
from torch.utils.data import DataLoader
from torch import nn

sys.path.append(os.path.abspath('lib/'))

model = CrossEncoder('ssce_save/fsce/', num_labels = 1)

with open('datasets/editor_standalone/pairwise_OOD.tsv', 'r') as r:
    OOD_data = r.readlines()

OOD_sentence_pairs = []
OOD_labels = []
for line in OOD_data:
    pair = line.strip('\n').split('\t')
    new_entry = []
    try:
        new_entry.append([pair[0], pair[1]])
    except:
        continue
    try:
        new_entry.append(int(pair[2]))
    except:
        continue
    OOD_sentence_pairs.append(new_entry[0])
    OOD_labels.append(new_entry[1])

OOD_evaluator = CEBinaryClassificationEvaluator(OOD_sentence_pairs, OOD_labels)

OOD_evaluator(model = model,
    output_path = './ssce_save/fsce/')