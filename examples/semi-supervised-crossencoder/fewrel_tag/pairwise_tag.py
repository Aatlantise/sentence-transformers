import json
import sys
import numpy as np
import random
np.random.seed(233)

with open('fewrel80_train.json','r') as r:
    train_data = json.load(r)

train_data_sample = random.sample(train_data, 1500)

w = open('pairwise_labeled_train.tsv', 'wb')

print("-----Parsing through labeled data-----")

for i, left_datum in enumerate(train_data_sample):
	if i > 0 and i % 250 == 0:
		count = i * (i - 1)
		print(str(int(count / 2)) + " pairs made")
	for right_datum in train_data_sample[i:]:
		label = left_datum['relid'] == right_datum['relid']
		output_line = '\t'.join([' '.join(right_datum["tag_after_text"]), ' '.join(left_datum["tag_after_text"]), str(int(label))]) + '\n'
		w.write(output_line.encode('utf-8'))

with open('fewrel80_test_train.json','r') as r:
    unlabeled_data = json.load(r)

unlabeled_data_sample = random.sample(unlabeled_data, 1000)

x = open('pairwise_unlabeled_train.tsv', 'wb')

print("-----Parsing through unlabeled data-----")

for i, left_datum in enumerate(unlabeled_data_sample):
	if i > 0 and i % 250 == 0:
		count = i * (i - 1)
		print(str(int(count / 2)) + " pairs made")
	for right_datum in unlabeled_data_sample[i:]:
		label = left_datum['relid'] == right_datum['relid']
		output_line = '\t'.join([' '.join(right_datum["tag_after_text"]), ' '.join(left_datum["tag_after_text"]), str(int(label))]) + '\n'
		x.write(output_line.encode('utf-8'))

with open('fewrel80_test_test.json','r') as r:
    test_data = json.load(r)

test_data_sample = random.sample(test_data, 1000)

y = open('pairwise_test.tsv', 'wb')

print("-----Parsing through test data-----")

for i, left_datum in enumerate(test_data_sample):
	if i > 0 and i % 250 == 0:
		count = i * (i - 1)
		print(str(int(count / 2)) + " pairs made")
	for right_datum in test_data_sample[i:]:
		label = left_datum['relid'] == right_datum['relid']
		output_line = '\t'.join([' '.join(right_datum["tag_after_text"]), ' '.join(left_datum["tag_after_text"]), str(int(label))]) + '\n'
		y.write(output_line.encode('utf-8'))