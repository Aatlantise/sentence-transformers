import numpy as np
import json, sys, os, argparse, io
from sentence_transformers import CrossEncoder
from sentence_transformers.readers import InputExample
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator, CEBinaryAccuracyEvaluator
from torch.utils.data import DataLoader
from torch import nn

sys.path.append(os.path.abspath('lib/'))

model = CrossEncoder('ssce_save/fsce/ko_re_tag', num_labels = 1)

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

OOD_evaluator = CEBinaryAccuracyEvaluator(OOD_sentence_pairs, OOD_labels)

OOD_evaluator(model = model,
    output_path = './ssce_save/fsce/ko_re_tag')

'''
OOD_examples = []

for line in OOD_data:
    pair = line.strip('\n').split('\t')
    try:
        OOD_examples.append(InputExample(texts = [pair[0], pair[1]], label = int(pair[2])))
    except:
        continue

train_dataloader = DataLoader(OOD_examples, shuffle=True, batch_size=16)
label_evaluator = CEBinaryClassificationEvaluator(test_sentence_pairs, test_labels)
'''