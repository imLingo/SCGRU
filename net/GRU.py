##################################################################################
'''
Description     :   GRU model library.
Author          :   Xiangzheng Ling.
institution     :   ECJTU, Data Science & Deep Learning Lab
Date            :   2020/10/25 19:10
'''
##################################################################################

import torch
import torch.nn as nn
import numpy as np
from net.prune import PruningModule,MaskedLinear
import random

# Select Device
use_cuda =  torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else 'cpu')

# seed
np.random.seed(0)


class GRUcell(PruningModule):
    r" GRU cell "
    def __init__(self, input_size, hidden_size, mask=False):
        super(GRUcell, self).__init__()
        linear = MaskedLinear if mask else nn.Linear
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gate = linear(input_size + hidden_size, hidden_size)
        self.output = linear(hidden_size, input_size)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

    def print_weight(self):
        print(self.gate.weight.data)
        return self.gate.weight.data

    def forward(self, input):
        max_time = input.size(1)
        batch_size = input.size(0)
        hidden = self.initHidden(batch_size)
        for time in range(max_time):
            input_slice = input[:, max_time - 1 - time].reshape([batch_size, 1])
            combined = torch.cat((hidden.float(), input_slice.float()), 1)
            reset_gate = self.gate(combined)
            update_gate = self.gate(combined)
            reset_gate = self.sigmoid(reset_gate)
            update_gate = self.sigmoid(update_gate)
            combined_2 = torch.cat((torch.mul(reset_gate, hidden).float() , input_slice.float()), 1)    # [128,351]
            hidden_helper = self.tanh(self.gate(combined_2))
            hidden = torch.add(torch.mul(update_gate, hidden_helper),
                               torch.mul((1- update_gate), hidden))
        hidden = self.output(hidden)
        # softmax_hidden = self.softmax(hidden)
        return hidden

    def initHidden(self,batch_size):
        return torch.zeros(batch_size, self.hidden_size).to(device)


class random_GRU(PruningModule):
    '''
    random function make a mask matrix used to sparse matrix W.
    '''
    def __init__(self, input_size, hidden_size,RC=False, connectivity=1.0, mask=False):
        super(random_GRU, self).__init__()
        linear = MaskedLinear if mask else nn.Linear
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.connectivity = connectivity
        self.gate = linear(input_size + hidden_size, hidden_size)
        self.output = linear(hidden_size, input_size)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        if RC:
            self.reset_parameters()

    def reset_parameters(self):
        '''
        Initialize parameters following the way proposed in the paper.
        '''
        row = self.hidden_size
        column = self.input_size +self.hidden_size
        weight_mask = torch.tensor(self.generate_weight_mask((row,column),self.connectivity)).float()
        weight_data = nn.init.orthogonal_(self.gate.weight.data)
        weight_data = self.gate.weight.data
        weight_data = weight_data.cuda() * weight_mask[0].cuda()
        self.gate.weight.data = weight_data

        column2 = self.input_size
        output_weight_mask = torch.tensor(generate_weight_mask((row, column2), self.connectivity)).float()
        output_weight_data = nn.init.orthogonal_(self.output.weight.data)
        output_weight_data = self.output.weight.data
        output_weight_data = output_weight_data.cuda() * output_weight_mask[:, :, 0].cuda()
        self.output.weight.data = output_weight_data

    def generate_weight_mask(self, shape, connection=1.):
        sub_shape = (shape[0], shape[1])
        w = []
        w.append(self.generate_mask_matrix(sub_shape, connection=connection))
        return w

    def generate_mask_matrix(self, shape, connection=1.):
        random.seed(0)
        total = shape[0] * shape[1]
        threshold = int(total * (1 - connection))
        s = [0] * threshold
        s.extend([1] * (total - threshold))
        random.shuffle(s)
        s = np.reshape(s, [shape[0], shape[1]])
        return s

    def forward(self, input):
        max_time = input.size(1)
        batch_size = input.size(0)
        hidden = self.initHidden(batch_size)
        for time in range(max_time):
            input_slice = input[:, max_time - 1 - time].reshape([batch_size, 1])  # [32]，最后一个[24]
            combined = torch.cat((hidden.float(), input_slice.float()), 1)
            reset_gate = self.gate(combined)
            update_gate = self.gate(combined)
            reset_gate = self.sigmoid(reset_gate)  # reset gate
            update_gate = self.sigmoid(update_gate)  # updata gate
            combined_2 = torch.cat((torch.mul(reset_gate, hidden).float() , input_slice.float()), 1)
            hidden_helper = self.tanh(self.gate(combined_2))    # h~(t+1)
            hidden = torch.add(torch.mul(update_gate, hidden_helper),
                               torch.mul((1- update_gate), hidden))
        hidden = self.output(hidden)
        # softmax_hidden = self.softmax(hidden)
        return hidden

    def initHidden(self,batch_size):
        return torch.zeros(batch_size, self.hidden_size).cuda()


def generate_mask_matrix(shape,connection=1.):
    random.seed(0)
    s = np.random.uniform(size=shape)
    s_flat = s.flatten()
    s_flat.sort()
    threshold = s_flat[int(shape[0]* shape[1]* (1-connection))]
    super_threshold_indices = s>= threshold
    lower_threshold_indices = s< threshold
    s[super_threshold_indices] = 1.
    s[lower_threshold_indices] = 0.
    return s


def generate_weight_mask(shape, connection = 1.):
    sub_shape = (shape[0],shape[1])
    w = []
    w.append(generate_mask_matrix(sub_shape,connection=connection))
    return w