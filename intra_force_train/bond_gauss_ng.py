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
from transconv2 import TransformerConv2
from dataclass import CGFFDataset
from util import plot_parity
prefixer='ng'
datain=CGFFDataset(root='data/',file_stem='gg',prefixer=prefixer,test=False)
datatest=CGFFDataset(root='data/',file_stem='gg',prefixer=prefixer,test=True)

loader = DataLoader(datain, batch_size=32)

class GNN_bond(torch.nn.Module):
    def __init__(self, feature_size, model_params):
        super(GNN_bond, self).__init__()
        embedding_size = model_params["model_embedding_size"]
        n_heads = model_params["model_attention_heads"]
        edge_dim1 = model_params["model_edge_dim1"]
        edge_dim2 = model_params["model_edge_dim2"]

        self.conv10 = TransformerConv2(feature_size,embedding_size,heads=n_heads,dropout=0.0,edge_dim=edge_dim1,beta=False,concat=False,root_weight=False) 
        self.conv11 = TransformerConv2(embedding_size,embedding_size,heads=n_heads,dropout=0.0,edge_dim=edge_dim1,beta=False,concat=False,root_weight=False) 
        self.conv12 = TransformerConv2(embedding_size,embedding_size,heads=n_heads,dropout=0.0,edge_dim=edge_dim1,beta=False,concat=False,root_weight=False) 
        self.conv13 = TransformerConv2(embedding_size,embedding_size,heads=n_heads,dropout=0.0,edge_dim=edge_dim1,beta=False,concat=False,root_weight=False) 
        self.conv14 = TransformerConv2(embedding_size,1,heads=n_heads,dropout=0.0,edge_dim=edge_dim1,beta=False,concat=False,root_weight=False) 
        
        self.conv20 = TransformerConv2(feature_size,embedding_size,heads=n_heads,dropout=0.0,edge_dim=edge_dim1,beta=False,concat=False,root_weight=False) 
        self.conv21 = TransformerConv2(embedding_size,embedding_size,heads=n_heads,dropout=0.0,edge_dim=edge_dim1,beta=False,concat=False,root_weight=False) 
        self.conv22 = TransformerConv2(embedding_size,embedding_size,heads=n_heads,dropout=0.0,edge_dim=edge_dim2,beta=False,concat=False,root_weight=False) 
        self.conv23 = TransformerConv2(embedding_size,embedding_size,heads=n_heads,dropout=0.0,edge_dim=edge_dim2,beta=False,concat=False,root_weight=False) 
        self.conv24 = TransformerConv2(embedding_size,1,heads=n_heads,dropout=0.0,edge_dim=edge_dim2,beta=False,concat=False,root_weight=False) 
        
        self.conv30 = TransformerConv2(feature_size,embedding_size,heads=n_heads,dropout=0.0,edge_dim=edge_dim1,beta=False,concat=False,root_weight=False) 
        self.conv31 = TransformerConv2(embedding_size,embedding_size,heads=n_heads,dropout=0.0,edge_dim=edge_dim1,beta=False,concat=False,root_weight=False) 
        self.conv32 = TransformerConv2(embedding_size,embedding_size,heads=n_heads,dropout=0.0,edge_dim=edge_dim2,beta=False,concat=False,root_weight=False) 
        self.conv33 = TransformerConv2(embedding_size,embedding_size,heads=n_heads,dropout=0.0,edge_dim=edge_dim2,beta=False,concat=False,root_weight=False) 
        self.conv34 = TransformerConv2(embedding_size,1,heads=n_heads,dropout=0.0,edge_dim=edge_dim2,beta=False,concat=False,root_weight=False) 
        
        self.conv40 = TransformerConv2(feature_size,embedding_size,heads=n_heads,dropout=0.0,edge_dim=edge_dim1,beta=False,concat=False,root_weight=False) 
        self.conv41 = TransformerConv2(embedding_size,embedding_size,heads=n_heads,dropout=0.0,edge_dim=edge_dim1,beta=False,concat=False,root_weight=False) 
        self.conv42 = TransformerConv2(embedding_size,embedding_size,heads=n_heads,dropout=0.0,edge_dim=edge_dim2,beta=False,concat=False,root_weight=False) 
        self.conv43 = TransformerConv2(embedding_size,embedding_size,heads=n_heads,dropout=0.0,edge_dim=edge_dim2,beta=False,concat=False,root_weight=False) 
        self.conv44 = TransformerConv2(embedding_size,1,heads=n_heads,dropout=0.0,edge_dim=edge_dim2,beta=False,concat=False,root_weight=False) 
        
    def forward(self, x, edge_attr, edge_index,uvec, edge_attr2, edge_index2,uvec2, edge_attr3, edge_index3,uvec3, edge_attr4, edge_index4,uvec4):
        
        hidden1 = self.conv10(x, edge_index, edge_attr)
        hidden1=torch.tanh(hidden1)
        hidden1 = self.conv11(hidden1, edge_index, edge_attr)
        hidden1=torch.tanh(hidden1)
        hidden1 = self.conv12(hidden1, edge_index, edge_attr)
        hidden1=torch.tanh(hidden1)
        hidden1 = self.conv13(hidden1, edge_index, edge_attr)
        hidden1=torch.tanh(hidden1)
        out1,outx1,outy1,outz1 = self.conv14(hidden1, edge_index,edge_attr,uvec)
        
        
        hidden2 = self.conv20(x, edge_index2, edge_attr2)
        hidden2=torch.tanh(hidden2)
        hidden2 = self.conv21(hidden2, edge_index2, edge_attr2)
        hidden2=torch.tanh(hidden2)
        hidden2 = self.conv22(hidden2, edge_index2, edge_attr2)
        hidden2=torch.tanh(hidden2)
        hidden2 = self.conv23(hidden2, edge_index2, edge_attr2)
        hidden2=torch.tanh(hidden2)
        out2,outx2,outy2,outz2 = self.conv24(hidden2, edge_index2,edge_attr2,uvec2)
        
        
        hidden3 = self.conv30(x, edge_index3, edge_attr3)
        hidden3=torch.tanh(hidden3)
        hidden3 = self.conv31(hidden3, edge_index3, edge_attr3)
        hidden3=torch.tanh(hidden3)
        hidden3 = self.conv32(hidden3, edge_index3, edge_attr3)
        hidden3=torch.tanh(hidden3)
        hidden3 = self.conv33(hidden3, edge_index3, edge_attr3)
        hidden3=torch.tanh(hidden3)
        out3,outx3,outy3,outz3 = self.conv34(hidden3, edge_index3,edge_attr3,uvec3)

        hidden4 = self.conv40(x, edge_index4, edge_attr4)
        hidden4=torch.tanh(hidden4)
        hidden4 = self.conv41(hidden4, edge_index4, edge_attr4)
        hidden4=torch.tanh(hidden4)
        hidden4 = self.conv42(hidden4, edge_index4, edge_attr4)
        hidden4=torch.tanh(hidden4)
        hidden4 = self.conv43(hidden4, edge_index4, edge_attr4)
        hidden4=torch.tanh(hidden4)
        out4,outx4,outy4,outz4 = self.conv44(hidden4, edge_index4,edge_attr4,uvec4)
        
        outx=outx1+outx2+outx3+outx4
        outy=outy1+outy2+outy3+outy4
        outz=outz1+outz2+outz3+outz4
        out1=torch.hstack((outx,outy))
        outf=torch.hstack((out1,outz))
        return outf
        
stem='v2_'+str(sys.argv[1])+'_'+str(sys.argv[2])+'_3_'+prefixer
model_params={}
model_params["model_embedding_size"]=int(sys.argv[1])
model_params["model_attention_heads"]=int(sys.argv[2])
model_params["model_edge_dim1"]=12
model_params["model_edge_dim2"]=12
print(datain[0].y.shape[1])
model=GNN_bond(feature_size=2, model_params=model_params)
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
        out=model(data.x.float(), data.edge_attr.float(),data.edge_index,data.uvec, data.edge_attr2.float(),data.edge_index2,data.uvec2, data.edge_attr3.float(),data.edge_index3,data.uvec3, data.edge_attr4.float(),data.edge_index4,data.uvec4)
        out=out[np.arange(np.shape(out)[0])%4!=3]
        data.y=data.y[np.arange(np.shape(data.y)[0])%4!=3]
        #print(np.shape(out),np.shape(data.y),np.shape(data.x))
        loss=criterion(out,data.y)
        loss.backward()
        optimizer.step()
        #print(loss.item,data.num_graphs)
        totloss+=loss.item()*data.num_graphs
    totloss/=len(loader.dataset)
    losses.append(totloss)
    if epoch % 10 == 0:
        print(f'Epoch: {epoch:03d}, Loss: {totloss:.4f}')
        if totloss<prevloss:
            torch.save(model.state_dict(), 'bond_'+stem+'.pt')
            prevloss=totloss
            print("model_saved",epoch)
        if math.isnan(totloss):
            break
