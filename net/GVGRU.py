##################################################################################
'''
Description     :   Gated Variant GRU model library.
Paper name      :   Gated-Variants of Gated Recurrent Unit (GRU) Neural Networks.
Paper source    :   https://arxiv.org/ftp/arxiv/papers/1701/1701.05923.pdf
Code reproducer :   Xiangzheng Ling.
institution     :   ECJTU, Data Science & Deep Learning Lab
Date            :   2020/11/26 11:10
'''
##################################################################################

import torch
import torch.nn as nn
import random

# Select Device
use_cuda =  torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else 'cpu')

# seed
random.seed(0)


class GVGRU1(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GVGRU1,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.gate_after = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, input_size)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
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
            combined = hidden
            reset_gate = self.gate_after(combined)
            update_gate = self.gate_after(combined)
            reset_gate = self.sigmoid(reset_gate)
            update_gate = self.sigmoid(update_gate)
            combined_2 = torch.cat((torch.mul(reset_gate, hidden).float() , input_slice.float()), 1)
            hidden_helper = self.relu(self.gate(combined_2))
            hidden = torch.add(torch.mul(update_gate, hidden_helper),
                               torch.mul((1- update_gate), hidden))

        hidden = self.output(hidden)
        return hidden

    def initHidden(self,batch_size):
        return torch.zeros(batch_size, self.hidden_size).to(device)


class GVGRU2(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GVGRU2, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.gate_after = nn.Linear(hidden_size, input_size,bias=False)
        self.output = nn.Linear(hidden_size, input_size)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
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
            combined = hidden
            reset_gate = self.gate_after(combined)
            update_gate = self.gate_after(combined)
            reset_gate = self.sigmoid(reset_gate)
            update_gate = self.sigmoid(update_gate)
            combined_2 = torch.cat((torch.mul(reset_gate, hidden).float() , input_slice.float()), 1)
            hidden_helper = self.relu(self.gate(combined_2))
            hidden = torch.add(torch.mul(update_gate, hidden_helper),
                               torch.mul((1- update_gate), hidden))
        hidden = self.output(hidden)
        return hidden

    def initHidden(self,batch_size):
        return torch.zeros(batch_size, self.hidden_size).to(device)


class GVGRU3(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GVGRU3, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, input_size)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
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
            combined = self.output.bias
            reset_gate = combined
            update_gate = combined
            reset_gate = self.sigmoid(reset_gate)
            update_gate = self.sigmoid(update_gate)
            combined_2 = torch.cat((torch.mul(reset_gate, hidden).float(), input_slice.float()), 1)
            hidden_helper = self.relu(self.gate(combined_2))
            hidden = torch.add(torch.mul(update_gate, hidden_helper),
                               torch.mul((1 - update_gate), hidden))
        hidden = self.output(hidden)
        return hidden

    def initHidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size).to(device)

