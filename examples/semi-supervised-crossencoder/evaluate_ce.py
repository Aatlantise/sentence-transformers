import numpy as np
import json, sys, os, argparse, io
from sentence_transformers import CrossEncoder
from sentence_transformers.readers import InputExample
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator, CEBinaryAccuracyEvaluator
from torch.utils.data import DataLoader
from torch import nn

sys.path.append(os.path.abspath('lib/'))


def evaluate_ce(num_labels, dataset_name, evalset_name, evaluator, threshold):
    model = CrossEncoder('ssce_save/fsce/' + dataset_name, num_labels = num_labels)

    with open(evalset_name, 'r') as r:
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

    if evaluator == 'accuracy':
        OOD_evaluator = CEBinaryAccuracyEvaluator(OOD_sentence_pairs, OOD_labels, threshold = threshold)
    elif evaluator == 'classification':
         OOD_evaluator = CEBinaryClassificationEvaluator(OOD_sentence_pairs, OOD_labels)

    OOD_evaluator(model = model,
        output_path = 'ssce_save/fsce/' + dataset_name)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type = str, default = 'augmented_ko_re_tag')
    parser.add_argument("--num_labels", type = int, default = 2)
    parser.add_argument("--evalset_name", type = str, default = 'datasets/editor_standalone/pairwise_OOD.tsv')
    parser.add_argument("--evaluator", type = str, default = 'accuracy')
    parser.add_argument("--threshold", type = float, default = 0.5)

    args = parser.parse_args()

    evaluate_ce(
        num_labels = args.num_labels,
        dataset_name = args.dataset_name,
        evalset_name = args.evalset_name,
        evaluator = args.evaluator,
        threshold = args.threshold
    )