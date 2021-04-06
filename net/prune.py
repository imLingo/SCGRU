##################################################################################
'''
Description     :   Pruning methods.
Author          :   Xiangzheng Ling.
institution     :   ECJTU, Data Science & Deep Learning Lab
Date            :   2020/10/27 15:12
'''
##################################################################################

import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
import random
# from torch._jit_internal import weak_module, weak_script_method


class RandomPruningModule(Module):
    r" random sparse "
    def prune_by_random(self,connectivity):
        for name, module in self.named_modules():
            if name in ['gate', 'output']:
                print(module)
                row = module.weight.data.shape[0]
                column = module.weight.data.shape[1]
                print(f'row={row}, column={column}')
                weight_mask = torch.tensor(self.generate_weight_mask((row, column), connectivity)).float()
                weight_data = nn.init.orthogonal_(module.weight.data)
                weight_data = weight_data.cuda() * weight_mask[0].cuda()
                module.weight.data = weight_data

    def generate_weight_mask(self, shape, connection=1.):
        sub_shape = (shape[0], shape[1])
        w = []
        w.append(self.generate_mask_matrix(sub_shape, connection=connection))
        return w

    def generate_mask_matrix(self, shape, connection=1.):
        r" Generate mask matrix"
        random.seed(0)
        total = shape[0] * shape[1]
        threshold = int(total * (1 - connection))
        s = [0] * threshold
        s.extend([1] * (total - threshold))
        random.shuffle(s)
        s = np.reshape(s, [shape[0], shape[1]])
        return s


class PruningModule(Module):
    def prune_by_std(self, s, k):
        for name, module in self.named_modules():
            if name in ['gate','output']:
                threshold = np.std(module.weight.data.abs().cpu().numpy()) * s
                print(f'Pruning with threshold:{threshold} for layer {name}')
                while not module.prune(threshold, k) : threshold *= 0.99


# @weak_module
class MaskedLinear(Module):
    r"""Applies a masked linear transformation to the incoming data: :math:`y = (A * M)x + b`"""
    def __init__(self, in_features, out_features, bias = True):
        super(MaskedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.mask = Parameter(torch.ones([out_features,in_features]),requires_grad= False)
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv,stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    # @weak_script_method
    def forward(self, input):
        return F.linear(input, self.weight, self.bias)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) \
            + ', bias= ' + str(self.bias is not None) + ')'

    def prune(self, threhold, k):
        weight_dev = self.weight.device
        mask_dev = self.mask.device

        # Convert Tensors to numpy and calculate
        tensor = self.weight.data.cpu().numpy()
        mask = self.mask.data.cpu().numpy()
        new_mask = np.where(abs(tensor) < threhold, 0, mask)

        # count non-zeros
        nz_count = np.count_nonzero(new_mask)
        if k <= nz_count/(self.in_features * self.out_features):
            # Apply new weight and mask
            self.weight.data = torch.from_numpy(tensor * new_mask).to(weight_dev)
            self.mask.data = torch.from_numpy(new_mask).to(mask_dev)
            return True
        else:
            return False
