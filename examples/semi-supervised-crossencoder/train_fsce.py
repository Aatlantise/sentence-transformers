from sentence_transformers import CrossEncoder
from sentence_transformers.readers import InputExample
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
from torch.utils.data import DataLoader
from torch import nn


roberta = CrossEncoder('monologg/kobert', num_labels = 1)
tokens = ["<e1>", "<e2>"]
roberta.tokenizer.add_tokens(tokens, special_tokens=True)
roberta.model.resize_token_embeddings(len(roberta.tokenizer))

with open('datasets/editor_standalone/pairwise_labeled_train.tsv','r') as r:
    labeled_data = r.readlines()

with open('datasets/editor_standalone/pairwise_unlabeled_train.tsv', 'r') as r:
	unlabeled_data = r.readlines()

with open('datasets/editor_standalone/pairwise_test.tsv', 'r') as r:
	test_data = r.readlines()

train_examples = [] 

for line in labeled_data + unlabeled_data:
	pair = line.strip('\n').split('\t')
	try:
	    train_examples.append(InputExample(texts = [pair[0], pair[1]], label = int(pair[2])))
	except:
		continue

test_sentence_pairs = []
test_labels = []
for line in test_data:
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
	test_sentence_pairs.append(new_entry[0])
	test_labels.append(new_entry[1])

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
evaluator = CEBinaryClassificationEvaluator(test_sentence_pairs, test_labels)

roberta.fit(train_dataloader = train_dataloader,
	evaluator = evaluator,
	epochs = 5,
	loss_fct = nn.BCEWithLogitsLoss(),
	output_path = './ssce_save/fsce/'
	)

with open('datasets/editor_standalone/processed_editor_eval_tag.tsv', 'r') as r:
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

OOD_evaluator(model = roberta,
    output_path = './ssce_save/fsce/')