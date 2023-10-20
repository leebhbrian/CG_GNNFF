import time
import sys
import numpy as np
import torch
import torch.nn.functional as F 
from GNN_force_strach import GNN_nb, GNN_bond
from scipy.spatial import cKDTree
from LDutil_str_gauss import *
prefixer='gauss'
nbmodel_params={}
nbmodel_params["model_embedding_size"]=8
nbmodel_params["model_attention_heads"]=5
if prefixer=='raw':
    nbmodel_params["model_edge_dim"]=1
else:
    nbmodel_params["model_edge_dim"]=4
    

bondmodel_params={}
bondmodel_params["model_embedding_size"]=8
bondmodel_params["model_attention_heads"]=5
if prefixer=='raw':
    bondmodel_params["model_edge_dim1"]=3
    bondmodel_params["model_edge_dim2"]=3
else:
    bondmodel_params["model_edge_dim1"]=12
    bondmodel_params["model_edge_dim2"]=12

ftype=sys.argv[1]
temper=int(sys.argv[2])
pressin=sys.argv[3]
gamma=10
xtype='n'
fname=ftype+'_'+str(temper)+'_'+str(pressin)
modelnb=GNN_nb(feature_size=2, model_params=nbmodel_params)
if xtype=='n':
    xfeat=2
else:
    xfeat=3
modelbond=GNN_bond(feature_size=xfeat, model_params=bondmodel_params)
modelnb.load_state_dict(torch.load('inter_model_g3.pt'))
modelbond.load_state_dict(torch.load('intra_model_g3.pt'))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
modelnb = modelnb.to(device)
modelbond = modelbond.to(device)
modelnb.eval()
modelbond.eval()


dt=1
masser,typer,pos,box_size=initialize(fname)
vel=createvel(masser,temper)

kbtm=1.380649*10**(-23)*temper*1000.0*6.02214*10**(23)/masser
c1=np.exp(-gamma*dt)
c2=np.sqrt(1-c1*c1)*np.sqrt(kbtm)*10**(-5)
t1 = time.time()
pos=rearrange(pos,box_size)
forcer=finfer(pos,typer,box_size,device,modelnb,modelbond,xtype)

dumpfreq=100
rstfreq=10000
nstep=200000
writecond=True
for i1 in range(nstep):
    pos,vel,forcer=integrator(pos,vel,forcer,masser,typer,box_size,dt,c1,c2,modelnb,modelbond,device,xtype)
    tmat=calctemp(masser,vel)
    tnow=tmat[3]
    if i1%dumpfreq==0:
        if writecond:
            wtype='w'
            writecond=False
        else:
            wtype='a'
        t2 = time.time()
        outdump(pos,str(ftype)+'_'+str(temper)+'_'+str(pressin)+'_'+str(xtype),wtype)
        print(i1,tnow,t2-t1)    