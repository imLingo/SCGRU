import argparse
import torch
import torch.utils.data as Data
from torch import nn
import torch.optim as optim
import numpy as np
import pandas as pd
from net.GVGRU import GVGRU1, GVGRU2, GVGRU3
import util
from tqdm import tqdm
from time import time
import matplotlib.pyplot as plt

# Parameter setting
parser = argparse.ArgumentParser(description='PyTorch GEANT Gated Variants GRU')
parser.add_argument('--data', default='./data/traffic-1-2.npy', help='path to dataset')
parser.add_argument('--model', default='GVGRU2', choices=['GVGRU1', 'GVGRU2', 'GVGRU3'], help='the model to use')
parser.add_argument('--connectivity', type=float, default=1.0, help='the neural connectivity')
parser.add_argument('--save', default='./model', help='The path to save model files')
parser.add_argument('--hidden_size', type=int, default=350, help='The number of hidden units')
parser.add_argument('--batch_size', type=int, default=32, help='The size of each batch')
parser.add_argument('--input_size', type=int, default=1, help='The size of input data')
parser.add_argument('--max_iter', type=int, default=1, help='The maximum iteration count')
parser.add_argument('--gpu', default=True, action='store_true', help='The value specifying whether to use GPU')
parser.add_argument('--time_window', type=int, default=100, help='The length of time window')
parser.add_argument('--dropout', type=float, default=1., help='Dropout')
parser.add_argument('--num_layers', type=int, default=1, help='The number of RNN layers')
parser.add_argument('--seed', type=int, default=42, metavar='S', help='random seed')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',help='learning rate')
parser.add_argument('--epochs', type=int, default=10, metavar='N',help='number of epochs to train')
args = parser.parse_args()

# Control Seed
torch.manual_seed(args.seed)

# Select Device
use_cuda = args.gpu and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else 'cpu')
if use_cuda:
    print("Using CUDA!")
else:
    print('Not using CUDA!!!')


# load data
data = np.load(args.data)
# take the logarithm of the original data
new_data = []
for x in data:
    if x > 0: new_data.append(np.log10(x))
    else: new_data.append(0.001)
new_data = np.array(new_data)
# handle abnormal data
new_data = new_data[new_data > 2.5]
data = new_data[new_data < 6]
# min-max normalization
max_data = np.max(data)
min_data = np.min(data)
data = (data - min_data) / (max_data - min_data)
df = pd.DataFrame({'temp': data})

for i in range(args.time_window):
    df['Lag' + str(i + 1)] = df.temp.shift(i + 1)
df = df.dropna()
# create X and y
y = df.temp.values
X = df.iloc[:, 1:].values
train_idx = int(len(df) * .9)
# create train and test data
train_X, train_Y, test_X, test_Y = X[:train_idx], y[:train_idx], X[train_idx:], y[train_idx:]

# Loader to contain
kwargs = {'num_workers': 5, 'pin_memory': True} if use_cuda else {}
torch_train_dataset = Data.TensorDataset(torch.tensor(train_X),torch.tensor(train_Y))
torch_test_dataset = Data.TensorDataset(torch.tensor(test_X),torch.tensor(test_Y))
train_loader = Data.DataLoader(
    dataset = torch_train_dataset,
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = Data.DataLoader(
    dataset = torch_test_dataset,
    batch_size=args.batch_size, shuffle=False, **kwargs)


# Define which model to use
model = GVGRU2(input_size= args.input_size, hidden_size= args.hidden_size).to(device)
util.print_model_parameters(model)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.00009)
initial_optimizer_state_dict = optimizer.state_dict()
MSE = nn.MSELoss()


vali_loss = []
vali_acc = []
def validation(mod):
    mod.eval()
    correct = 0
    vali_losss = 0
    done = 0
    with torch.no_grad():
        for data, target in train_loader:
            data,target = data.to(device), target.to(device)
            output = mod(data)
            target = target.reshape([len(target), 1]).float()
            loss = torch.sqrt(nn.MSELoss()(input=output, target=target))
            vali_losss += loss.item()
        vali_losss /= len(train_loader)
    return vali_losss

epochs_train_loss=[]
epochs_test_loss=[]

all_epochs_loss=[]
def train(epochs):
    model.train()
    for epoch in range(epochs):
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        train_loss = 0
        done = 0
        for batch_idx, (data, target) in pbar:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            target = target.reshape([len(target), 1]).float()
            loss = torch.sqrt(MSE(input=output, target=target))
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            done += len(target)
            percentage = 100. * (batch_idx + 1) / len(train_loader)
            pbar.set_description(f'Train Epoch: {epoch+1}/{args.epochs} [{done:5}/{len(train_loader.dataset)} ({percentage:3.0f}%)]  Loss: {loss.item():.6f}')
        train_loss /= len(train_loader)
        epochs_train_loss.append(train_loss)
        print(f'train set: Average loss: {train_loss:.4f}')
        all_epochs_loss.append(train_loss)


def Test():
    model.eval()
    test_loss = 0
    correct = 0
    # torch.no_grad()
    with torch.no_grad():
        done = 0
        pbar = tqdm(enumerate(test_loader), total=len(test_loader))
        for batch_idx,(data, target) in pbar:
            data, target = data.to(device),target.to(device)
            output = model(data)
            target = target.reshape([len(target), 1]).float()
            loss = torch.sqrt(nn.MSELoss()(input=output, target=target.float()))
            test_loss += loss.item()
            done += len(target)
            percentage = 100. * (batch_idx + 1) / len(test_loader)
            pbar.set_description(f'Test Epoch: [{done:5}/{len(test_loader.dataset)} ({percentage:3.0f}%)]  Loss: {loss.item():.6f}')
        test_loss /= len(test_loader)
        print(f'Test set: Average loss: {test_loss}')
    return test_loss

def show_plot(x):
    fig ,ax = plt.subplots()
    colors = ['red','blue']
    for i,data in enumerate(x):
        ax.plot(range(len(data)),data,c=colors[i])
    plt.show()

# start training
print(f"--- {args.model}_{args.hidden_size} training ---")
start_time = time()
train(args.epochs)

# print training time
train_time = time() - start_time;
print(f"training times：{train_time:.4f} s = {train_time/60} min")

# test and print test time
Test()
test_time = time()-start_time-train_time;
print(f"test times ：{test_time:.4f} s = {test_time/60} min")

# save model
util.print_nonzeros(model=model)
torch.save(model.state_dict(), f'./saves/{args.model}_{args.hidden_size}_ep{args.epochs}.pt')
