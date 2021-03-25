from sentence_transformers import CrossEncoder
from sentence_transformers.readers import InputExample
from sentence_transformers.cross_encoder.evaluation import CEBinaryAccuracyEvaluator
from torch.utils.data import DataLoader


model = CrossEncoder('stsb-roberta-base', num_labels = 2)
roberta = model._first_module()
tokens = ["<e1>", "<e2>"]
roberta.tokenizer.add_tokens(tokens, special_tokens=True)
roberta.auto_model.resize_token_embeddings(len(roberta.tokenizer))

with open('fewrel_tag/pairwise_labeled_train.tsv','r') as r:
    labeled_data = r.readlines()

with open('fewrel_tag/pairwise_unlabeled_train.tsv', 'r') as r:
	unlabeled_data = r.readlines()

with open('fewrel_tag/pairwise_test.tsv', 'r') as r:
	test_data = r.readlines()

train_examples = [] 

for line in labeled_data:
	pair = line.strip('\n').split('\t')
	train_examples.append(InputExample(texts = [pair[0], pair[1]], label = int(pair[2])))

test_sentence_pairs = []
test_labels = []
for line in test_data:
	pair = line.strip('\n').split('\t')
	test_sentence_pairs.append([pair[0], pair[1]])
	test_labels.append(int(pair[2]))

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
evaluator = CEBinaryAccuracyEvaluator(test_sentence_pairs, test_labels, write_csv = True)

roberta.fit(train_dataloader = train_dataloader,
			evaluator = evaluator,
			epochs = 5,
			output_path = './ssce_save/'
			)