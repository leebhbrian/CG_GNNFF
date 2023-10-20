import torch
import torch.nn.functional as F 
from torch.nn import Linear, BatchNorm1d, ModuleList
import torch_geometric
from torch_geometric.data import Data
from transconv2 import TransformerConv2
from transconv3 import TransformerConv3

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