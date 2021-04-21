from sentence_transformers import CrossEncoder
from sentence_transformers.readers import InputExample
from sentence_transformers.cross_encoder.evaluation import CEBinaryAccuracyEvaluator, CEBinaryClassificationEvaluator
from torch.utils.data import DataLoader
from torch import nn
import argparse 

def train_fsce(model_name, num_labels, dataset_name, num_epochs, batch_size):
    kobert = CrossEncoder(model_name, num_labels = num_labels)
    tokens = ["<e1>", "<e2>"]
    kobert.tokenizer.add_tokens(tokens, special_tokens=True)
    kobert.model.resize_token_embeddings(len(kobert.tokenizer))

    with open('datasets/' + dataset_name + '/pairwise_labeled_train.tsv','r') as r:
        labeled_data = r.readlines()

    with open('datasets/' + dataset_name + '/pairwise_unlabeled_train.tsv', 'r') as r:
        unlabeled_data = r.readlines()

    with open('datasets/' + dataset_name + '/pairwise_test.tsv', 'r') as r:
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

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    evaluator = CEBinaryAccuracyEvaluator(test_sentence_pairs, test_labels)

    kobert.fit(train_dataloader = train_dataloader,
        evaluator = evaluator,
        epochs = 5,
        #loss_fct = nn.BCEWithLogitsLoss(),
        output_path = './ssce_save/fsce/' + dataset_name
        )


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type = str, default = 'monologg/kobert')
    parser.add_argument("--dataset_name", type = str, default = 'augmented_ko_re_tag')
    parser.add_argument("--num_labels", type = int, default = 1)
    parser.add_argument("--num_epochs", type = int, default = 5)
    parser.add_argument("--batch_size", type = int, default = 128)

    args = parser.parse_args()

    train_fsce(
        model_name = args.model,
        num_labels = args.num_labels,
        dataset_name = args.dataset_name,
        num_epochs = args.num_epochs,
        batch_size = args.batch_size
    )