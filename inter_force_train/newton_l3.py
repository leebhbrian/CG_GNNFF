import torch
import torch.nn.functional as F 
from torch.nn import Linear, BatchNorm1d, ModuleList
from torch_geometric.nn import TransformerConv, TopKPooling , GATConv, GCNConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch_geometric
from torch_geometric.data import Dataset, Data, Batch, InMemoryDataset

from torch import Tensor
from torch_sparse import SparseTensor
from torch_geometric.nn.conv import MessagePassing2
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_geometric.utils import softmax
from torch_scatter import gather_csr, scatter,scatter_sum, segment_csr

import math
import sys
from typing import Optional, Tuple, Union
import numpy as np 
import os
from torch_geometric.loader import DataLoader
from transconv3 import TransformerConv3
from dataclass import CGFFDataset
#from util import plot_parity
fstem='nb'        
prefixer1=sys.argv[3]
prefixer2=sys.argv[4]
prefixer=prefixer1+prefixer2
datain=CGFFDataset(root='data/',file_stem='nball',prefixer=prefixer,test=False)
datatest=CGFFDataset(root='data/',file_stem='nball',prefixer=prefixer,test=True)

loader = DataLoader(datain, batch_size=32)

class GNN_nb(torch.nn.Module):
    def __init__(self, feature_size, model_params):
        super(GNN_nb, self).__init__()
        embedding_size = model_params["model_embedding_size"]
        n_heads = model_params["model_attention_heads"]
        edge_dim = model_params["model_edge_dim"]

        self.conv0 = TransformerConv3(feature_size,embedding_size,heads=n_heads,dropout=0.0,edge_dim=edge_dim,beta=False,concat=False,root_weight=False) 
        self.conv1 = TransformerConv3(embedding_size,embedding_size,heads=n_heads,dropout=0.0,edge_dim=edge_dim,beta=False,concat=False,root_weight=False) 
        self.conv2 = TransformerConv3(embedding_size,embedding_size,heads=n_heads,dropout=0.0,edge_dim=edge_dim,beta=False,concat=False,root_weight=False) 
        self.conv3 = TransformerConv3(embedding_size,1,heads=n_heads,dropout=0.0,edge_dim=edge_dim,beta=False,concat=False,root_weight=False)         
    def forward(self, x, edge_attr, edge_index,uvec):
        # First Conv layer
        hidden = self.conv0(x, edge_index, edge_attr)
        hidden=torch.tanh(hidden)
        hidden = self.conv1(hidden, edge_index, edge_attr)
        hidden=torch.tanh(hidden)
        hidden = self.conv2(hidden, edge_index, edge_attr)
        hidden=torch.tanh(hidden)
        out,outx,outy,outz = self.conv3(hidden, edge_index,edge_attr,uvec)
        out1=torch.hstack((outx,outy))
        outf=torch.hstack((out1,outz))
        return outf
        
stem=prefixer+'_'+str(sys.argv[1])+'_'+str(sys.argv[2])+'_3'
model_params={}
model_params["model_embedding_size"]=int(sys.argv[1])
model_params["model_attention_heads"]=int(sys.argv[2])
if prefixer1=='r' or prefixer1=='f':
    model_params["model_edge_dim"]=1
elif prefixer1=='g':
    model_params["model_edge_dim"]=4
model=GNN_nb(feature_size=datain[0].x.shape[1], model_params=model_params)
print(model)


# Use GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model = model.to(device)

# Initialize Optimizer
learning_rate = 0.005
decay = 5e-4
#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=decay)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.7, nesterov=True)
criterion = torch.nn.MSELoss()
losses=[]

nepoch=2000
prevloss=5000000
for epoch in range(nepoch):
    totloss=0
    model.train()
    for batch in loader:
        optimizer.zero_grad()
        data=batch.to(device)
        out=model(data.x.float(), data.edge_attr.float(),data.edge_index,data.uvec)
        #print(np.shape(out),np.shape(data.y),np.shape(data.x))
        loss=criterion(out,data.y)
        loss.backward()
        optimizer.step()
        #print(loss.item,data.num_graphs)
        totloss+=loss.item()*data.num_graphs
    totloss/=len(loader.dataset)
    losses.append(totloss)
    if epoch % 10 == 0:
        print('Epoch:'+str(epoch)+', Loss: '+str(totloss))
        if totloss<prevloss:
            torch.save(model.state_dict(), 'newton_'+stem+'.pt')
            prevloss=totloss
            print("model_saved",epoch)
        if math.isnan(totloss):
            break
