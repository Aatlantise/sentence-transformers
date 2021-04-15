import numpy as np
import json, sys, os, argparse, io

sys.path.append(os.path.abspath('lib/'))
from evaluation import ClusterEvaluation
from messager import messager
from clusters import *

# load clustering dataset
# load pytorch model as predictor

path = 'ssce_save/sce/'
roberta = torch.load(path)

with open('fewrel/pairwise_cluster', 'r', encoding = 'utf-8') as r:
    pairwise_cluster = json.load(r)

'''
We need: 
data_to_cluster, gt, SN_pred_X

'''