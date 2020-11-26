"""
Run evaluation with saved models.
"""
import random
import argparse
from tqdm import tqdm
import torch
import numpy as np

from data.loader import DataLoader
from model.trainer import GCNTrainer
from utils import torch_utils, scorer, constant, helper
from utils.vocab import Vocab

parser = argparse.ArgumentParser()
parser.add_argument('model_dir', type=str, help='Directory of the model.')
parser.add_argument('--model', type=str, default='best_model.pt', help='Name of the model file.')
parser.add_argument('--data_dir', type=str, default='/dataset/cause_effect')
parser.add_argument('--dataset', type=str, default='MedCaus_test', help="Evaluate on dev or test.")

parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--cpu', action='store_true')
args = parser.parse_args()

torch.manual_seed(args.seed)
random.seed(1234)
if args.cpu:
    args.cuda = False
elif args.cuda:
    torch.cuda.manual_seed(args.seed)

# load opt
model_file = args.model_dir + '/' + args.model
print("Loading model from {}".format(model_file))
opt = torch_utils.load_config(model_file)
trainer = GCNTrainer(opt)
trainer.load(model_file)



# load vocab
vocab_file = args.model_dir + '/vocab.pkl'
vocab = Vocab(vocab_file, load=True)
assert opt['vocab_size'] == vocab.size, "Vocab size must match that in the saved model."

# load data
data_file = opt['data_dir'] + '/{}.txt'.format(args.dataset)
print("Loading data from {} with batch size {}...".format(data_file, opt['batch_size']))
batch = DataLoader(data_file, opt['batch_size'], opt, vocab, evaluation=True)

helper.print_config(opt)
label2id = constant.LABEL_TO_ID
id2label = dict([(v,k) for k,v in label2id.items()])

predictions = []
all_probs = []
batch_iter = tqdm(batch)
metrics = np.asarray([0, 0, 0, 0, 0, 0], dtype=int)
for i, b in enumerate(batch_iter):
    metrics += trainer.predict(b)


p, total_predict, total_entity = metrics[0], metrics[1], metrics[2]
test_p = p * 1.0 / total_predict * 100 if total_predict != 0 else 0
test_r = p * 1.0 / total_entity * 100 if total_entity != 0 else 0
test_f1 = 2.0 * test_p * test_r / (test_p + test_r) if test_p != 0 or test_r != 0 else 0

p_p, total_predict_p, total_entity_p = metrics[3], metrics[4], metrics[5]
test_p_p = p_p * 1.0 / total_predict_p * 100 if total_predict_p != 0 else 0
test_r_p = p_p * 1.0 / total_entity_p * 100 if total_entity_p != 0 else 0
test_f1_p = 2.0 * test_p_p * test_r_p / (test_p_p + test_r_p) if test_p_p != 0 or test_r_p != 0 else 0

print("Precision: %.2f, Recall: %.2f, F1: %.2f, P_partial: %.2f, Recall: %.2f, F1: %.2f" % (test_p, test_r, test_f1, test_p_p, test_r_p, test_f1_p), flush=True)

#predictions = [id2label[p] for p in predictions]
#p, r, f1 = scorer.score(batch.gold(), predictions, verbose=True)
#print("{} set evaluate result: {:.2f}\t{:.2f}\t{:.2f}".format(args.dataset,p,r,f1))

print("Evaluation ended.")

