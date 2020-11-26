import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F
import pdb


def ACE(features, ad_net, opt):
    ad_out = ad_net(features)
    batch_size = features.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * (features.size(0) - batch_size))).float()
    if opt['cuda']: dc_targets = dc_targets.cuda()
    return nn.BCELoss()(ad_out, dc_target)
