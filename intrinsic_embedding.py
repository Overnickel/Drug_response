import pandas as pd
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from sklearn.model_selection import train_test_split

class dts(Dataset):

    def __init__(self, expr_df, ess_df):
        self.expr_df = expr_df
        self.ess_df = ess_df
        self.num = self.expr_df.shape[0]

    def __len__(self):
        return self.num

    def __getitem__(self, i):
        sample = self.expr_df.iloc[i, :].values, self.ess_df.iloc[i, :].values
        sample = torch.from_numpy(sample[0]).type(torch.FloatTensor), torch.from_numpy(sample[1]).type(torch.FloatTensor)


# embedding
class embedding(nn.Module):
    # change so that number of layer could be also a parameter [64, 32, 32]
    def __init__(self, input_size, embedding_size, num_nodes):
        super(embedding, self).__init__()
        layer_nodes = [input_size] + num_nodes + [embedding_size]
        self.num_layers = len(layer_nodes)-1
        self.fc_layers = []
        self.batchnorm_layers = []
        for i in xrange(1, len(layer_nodes)):
            self.fc_layers.append(nn.Linear(layer_nodes[i-1], layer_nodes[i]))
        for i in xrange(1, len(layer_nodes)-1):
            self.batchnorm_layers.append(nn.BatchNorm1d(layer_nodes[i]))

    def forward(self, x):
        out = x
        for i in xrange(self.num_layers-1):
            out = self.fc_layers[i](out)
            out = self.batchnorm_layers[i](out)
            out = F.relu(out)
            out = F.dropout(out, p=0.5, training=self.training)
        out = F.fc_layers[-1](out)
        return out

class output_layer(nn.Module):
    def __init__(self, embedding_size):
        super(output_layer, self).__init__()
        self.fc1 = nn.Linear(embedding_size, input_size)

    def forward(self, x):
        out = self.fc1(x)
        return out

def train(args, embedded, output_layer, device, train_loader, optimizer, epoch):
    embedded.train()
    output_layer.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = output_layer(embedded(data))
        loss = nn.MSELoss()(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, embedded, output_layer, device, test_loader): 
    embedded.eval()
    output_layer.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = output_layer(embedded(data))
            test_loss += nn.MSELoss(reduction='sum')(output, target).item() # sum up batch loss
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))

# Device configuration
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
class Args: pass
args = Args()
args.input_size = 17234
args.hidden_size = 500
args.num_epochs = 5
args.batch_size = 100
args.lr = 0.001
args.split_ratio = 0.8
args.seed = 1
args.num_workers = 1

##
# TP53(essentiality ) ~ embedding + output (exp)
exp_mat = pd.read_csv('Gene_expr.tsv', sep='\t')
exp_mat.columns = [s.split()[0] for s in exp_mat.columns]
ess_mat = pd.read_csv('Crispr_ess.tsv', sep='\t')
ess_mat.columns = [s.split()[0] for s in exp_mat.columns]

tp53_ess = ess_mat.loc[:, 'TP53']
cellline = ess_mat.loc[cel1, :]



embedded = embedding(input_size, embedding_size, num_nodes).to(device)
output_layer = output_layer(embedding_size).to(device)
optimizer = optim.Adam(list(embedded.parameters())+list(output_layer.parameters()), lr=args.lr)

expr_train, expr_test, ess_train, ess_test = train_test_split(exp_mat, ess_mat, train_size=args.split_ratio, random_state=args.seed)
trainset = dts(expr_train, ess_train)
testset = dts(expr_test, ess_test)
kwargs = {'pin_memory': True} if use_cuda else {}
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, num_workers=args.num_workers, timeout=1, **kwargs)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, num_workers=args.num_workers, timeout=1, **kwargs)

for epoch in range(1, args.epochs + 1):
    train(args, embedded, output_layer, device, train_loader, optimizer, epoch)
    test(args, embedded, output_layer, device, test_loader)

