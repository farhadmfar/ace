"""
A trainer class.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from model.gcn import GCNClassifier
from model import adversarial
from model import ace
from utils import constant, torch_utils
from utils.torch_utils import make_cuda
from utils.crf_utils import evaluate_batch_insts


class Trainer(object):
    def __init__(self, opt, emb_matrix=None):
        raise NotImplementedError

    def update(self, batch):
        raise NotImplementedError

    def predict(self, batch):
        raise NotImplementedError

    def update_lr(self, new_lr):
        torch_utils.change_lr(self.optimizer, new_lr)

    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.model.load_state_dict(checkpoint['model'])
        self.opt = checkpoint['config']

    def save(self, filename, epoch):
        params = {
                'model': self.model.state_dict(),
                'config': self.opt,
                }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")


def unpack_batch(batch, cuda):
    if cuda:
        inputs = [Variable(b.cuda()) for b in batch[:7]]
        labels = Variable(batch[7].cuda())
    else:
        inputs = [Variable(b) for b in batch[:7]]
        labels = Variable(batch[7])
    tokens = batch[0]
    head = batch[5]
    lens = batch[1].eq(0).long().sum(1).squeeze()
    return inputs, labels, tokens, head, lens

class GCNTrainer(Trainer):
    def __init__(self, opt, emb_matrix=None):
        self.opt = opt
        self.emb_matrix = emb_matrix
        self.model = GCNClassifier(opt, emb_matrix=emb_matrix)
        self.adversarial_net = adversarial.AdversarialNetwork( 2 * self.opt['hidden_dimension'], \
                                                               2 * self.opt['hidden_dimension'])
        self.criterion = nn.CrossEntropyLoss()	
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        if opt['cuda']:
            self.model.cuda()
            self.criterion.cuda()
            self.adversarial_net.cuda()
        self.optimizer = torch_utils.get_optimizer(opt['optim'], self.parameters, opt['lr'])

    def update(self, batch, i, diff):
        inputs, labels, tokens, head, lens = unpack_batch(batch, self.opt['cuda'])

        # step forward
        self.model.train()
        self.optimizer.zero_grad()
        
        if self.opt['adapation']:
            if i < diff: loss, feature = self.model(inputs)
            else: loss = 0
            loss += ace.ACE(feature, self.adversarial_net, self.opt)
        else:
            loss, feature = self.model(inputs)

        loss_val = loss.item()
        #backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opt['max_grad_norm'])
        self.optimizer.step()
        return loss_val

    def predict(self, batch, unsort=True):
        inputs, labels, tokens, head, lens = unpack_batch(batch, self.opt['cuda'])
        l = (inputs[1].data.cpu().numpy() == 0).astype(np.int64).sum(1)
        l = make_cuda(torch.LongTensor(l))
        # forward
        self.model.eval()
        batch_max_scores, batch_max_ids = self.model.decode(inputs)
        
        tag2id = constant.TAG_TO_ID
        id2tag = dict([(v,k) for k,v in tag2id.items()])
        metric = evaluate_batch_insts( batch_max_ids, inputs[-1], l, id2tag)

        return metric
        
    def get_feature(self, batch, unsort=True):
        inputs, labels, tokens, head, lens = unpack_batch(batch, self.opt['cuda'])
        l = (inputs[1].data.cpu().numpy() == 0).astype(np.int64).sum(1)
        l = make_cuda(torch.LongTensor(l))
        # forward
        self.model.eval()
        _, feature = self.model(inputs)
        return feature
