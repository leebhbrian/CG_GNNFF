import math
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_sparse import SparseTensor

from torch_geometric.nn.conv import MessagePassing2
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_geometric.utils import softmax
from torch_scatter import gather_csr, scatter,scatter_sum, segment_csr

class TransformerConv2(MessagePassing2):
    _alpha: OptTensor

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        beta: bool = False,
        dropout: float = 0.,
        edge_dim: Optional[int] = None,
        bias: bool = True,
        root_weight: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super(TransformerConv2, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.beta = beta and root_weight
        self.root_weight = root_weight
        self.concat = concat
        self.dropout = dropout
        self.edge_dim = edge_dim
        self._alpha = None
        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_key = Linear(in_channels[0], heads * out_channels)
        self.lin_query = Linear(in_channels[1], heads * out_channels)
        self.lin_value = Linear(in_channels[0], heads * out_channels)
        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False)
        else:
            self.lin_edge = self.register_parameter('lin_edge', None)

        if concat:
            self.lin_skip = Linear(in_channels[1], heads * out_channels,
                                   bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * heads * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)
        else:
            self.lin_skip = Linear(in_channels[1], out_channels, bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        if self.edge_dim:
            self.lin_edge.reset_parameters()
        self.lin_skip.reset_parameters()
        if self.beta:
            self.lin_beta.reset_parameters()
    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_attr: OptTensor = None,uvec:OptTensor=None, return_attention_weights=None):
        
        H, C = self.heads, self.out_channels
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        query = self.lin_query(x[1]).view(-1, H, C)
        key = self.lin_key(x[0]).view(-1, H, C)
        value = self.lin_value(x[0]).view(-1, H, C)
        dim_sz=query.size()[0]
        # propagate_type: (query: Tensor, key:Tensor, value: Tensor, edge_attr: OptTensor) # noqa
        out,out2in = self.propagate(edge_index, query=query, key=key, value=value,
                             edge_attr=edge_attr, size=None,uvec=uvec)

        if uvec is not None:
            out2inx=out2in*uvec[:,0].reshape(-1,1,1)
            out2iny=out2in*uvec[:,1].reshape(-1,1,1)
            out2inz=out2in*uvec[:,2].reshape(-1,1,1)
            outx=scatter_sum(out2inx,edge_index[1,:],dim=0,dim_size=dim_sz)
            outy=scatter_sum(out2iny,edge_index[1,:],dim=0,dim_size=dim_sz)
            outz=scatter_sum(out2inz,edge_index[1,:],dim=0,dim_size=dim_sz)
            outtot=torch.sqrt(torch.square(outx)+torch.square(outy)+torch.square(outz))
            
            if self.concat:
                outtot=outtot.view(-1, self.heads * self.out_channels)
                outx=outx.view(-1, self.heads * self.out_channels)
                outy=outy.view(-1, self.heads * self.out_channels)
                outz=outz.view(-1, self.heads * self.out_channels)
            else:
                outtot=outtot.mean(dim=1)
                outx=outx.mean(dim=1)
                outy=outy.mean(dim=1)
                outz=outz.mean(dim=1)
        
            if self.root_weight:
                x_r = self.lin_skip(x[1])
                if self.lin_beta is not None:
                    betatot = self.lin_beta(torch.cat([outtot, x_r, outtot - x_r], dim=-1))
                    betatot = betatot.sigmoid()
                    outtot = betatot * x_r + (1 - betatot) * outtot
                    
                    betax = self.lin_beta(torch.cat([outx, x_r, outx - x_r], dim=-1))
                    betax = betax.sigmoid()
                    outx = betax * x_r + (1 - betax) * outx
                    
                    betay = self.lin_beta(torch.cat([outy, x_r, outy - x_r], dim=-1))
                    betay = betay.sigmoid()
                    outy = betay * x_r + (1 - betay) * outy
                    
                    betaz = self.lin_beta(torch.cat([outz, x_r, outz - x_r], dim=-1))
                    betaz = betaz.sigmoid()
                    outz = betaz * x_r + (1 - betaz) * outz
                else:
                    outtot = outtot + x_r
                    outx = outx + x_r
                    outy = outy + x_r
                    outz = outz + x_r
                    
        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)
        if self.root_weight:
            x_r = self.lin_skip(x[1])
            if self.lin_beta is not None:
                beta = self.lin_beta(torch.cat([out, x_r, out - x_r], dim=-1))
                beta = beta.sigmoid()
                out = beta * x_r + (1 - beta) * out
            else:
                out = out + x_r
                
        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            if uvec is not None:
                return outtot,outx,outy,outz
            else:
                return out
    
    def message(self, query_i: Tensor, key_j: Tensor, value_j: Tensor,
                edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:

        if self.lin_edge is not None:
            assert edge_attr is not None
            edge_attr = self.lin_edge(edge_attr).view(-1, self.heads,
                                                      self.out_channels)
            key_j = key_j + edge_attr

        alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(self.out_channels)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        out = value_j
        if edge_attr is not None:
            out = out + edge_attr
        out = out * alpha.view(-1, self.heads, 1)
        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')