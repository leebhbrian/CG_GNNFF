import math
import numpy as np
import os
import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Dataset, Data, Batch, InMemoryDataset
from scipy.spatial import cKDTree

class CGFFDataset(Dataset):
    def __init__(self, root, file_stem,prefixer, test=False, transform=None, pre_transform=False):
        self.test = test
        self.file_stem= file_stem
        self.nframetest=30
        self.nframetrain=240
        self.prefixer=prefixer
        self.nframetot=self.nframetrain+self.nframetest
        self.fmat=['nb2_cryst_g0_amb.data']
        self.flen=[270]
        super(CGFFDataset, self).__init__(root,transform, pre_transform)

    @property
    def raw_file_names(self):
        return self.fmat

    @property
    def processed_file_names(self):
        if self.test:
            return [str(self.prefixer)+'_test_'+str(i)+'.pt' for i in np.arange(self.nframetest)]
        else:
            return [str(self.prefixer)+'_train_'+str(i)+'.pt' for i in np.arange(self.nframetrain)]
        
    def download(self):
        pass
    
    def process(self):
        traincount=0
        testcount=0
        dgauss=np.array([[0,0.15],[3.5,2],[5.5,5],[6.4,5]])
        rcutrho=9.0
        wconst=84.0/(5.0*3.141592*rcutrho**3.0)

        for ifile in range(len(self.fmat)):
            fname=self.fmat[ifile]
            rf=open(os.path.join(self.raw_dir,fname),'r')
        
            for ifr in range(self.flen[ifile]):
                tr=rf.readline().split()
                nnode=int(tr[1])
                box_size=np.zeros(3)
                positions=np.zeros((nnode,3))
                datsave=np.zeros((nnode,7))
                tr=rf.readline().split()
                for idim in range(3):
                    box_size[idim]=float(tr[idim])
                rf.readline()
                for iat in range(nnode):
                    tr=rf.readline().split()
                    for idim in range(7):
                        datsave[iat,idim]=float(tr[idim])
                rf.readline()
                xdat=np.zeros((nnode,2))
                for iat in range(nnode):
                    for idim in range(3):
                        if datsave[iat,idim+1]<0.0:
                            mult=math.ceil(-datsave[iat,idim+1]/box_size[idim])
                            datsave[iat,idim+1]=datsave[iat,idim+1]+box_size[idim]*mult
                        elif datsave[iat,idim+1]>box_size[idim]:
                            mult=math.floor(datsave[iat,idim+1]/box_size[idim])
                            datsave[iat,idim+1]=datsave[iat,idim+1]-box_size[idim]*mult
                        positions[iat,idim]=datsave[iat,idim+1]
                    if datsave[iat,0]==1:
                        xdat[iat,0]=1.0
                        xdat[iat,1]=0.0
                    elif datsave[iat,0]==2:
                        xdat[iat,0]=0.0
                        xdat[iat,1]=1.0
                receivers_list =cKDTree(positions, boxsize=box_size).query_pairs(r=9.0)    
                num_nodes = len(positions)
                edge_index=[]
                edge_attr=[]
                edge_attrraw=[]
                edge_attrfc=[]
                uvec=[]
                uvec2=[]
                count=0
                moldat=np.zeros((num_nodes,1))
                nmol=int(num_nodes/4)
                atcount=-1
                ccount=np.zeros((nnode,1))
                for imol in range(nmol):
                    for isub in range(4):
                        atcount+=1
                        moldat[atcount,0]=imol
                for a in receivers_list:
                    if moldat[a[0],0]!=moldat[a[1],0]:
                        edge_index.append([a[0],a[1]])
                        edge_index.append([a[1],a[0]])
                        diff1=np.zeros(4)
                        diff2=np.zeros(4)
                        diffadd1=np.zeros(4)
                        diffadd2=np.zeros(4)
                        diffadd1raw=np.zeros(1)
                        diffadd2raw=np.zeros(1)
                        diffadd1fc=np.zeros(1)
                        diffadd2fc=np.zeros(1)
                        uv1=np.zeros(3)
                        uv2=np.zeros(3)
                        smuv1=np.zeros(3)
                        smuv2=np.zeros(3)
                        for idim in range(3):
                            difftemp=positions[a[0],idim]-positions[a[1],idim]
                            if difftemp>(box_size[idim]/2.0):
                                difftemp=difftemp-box_size[idim]
                            elif difftemp<(-box_size[idim]/2.0):
                                difftemp=difftemp+box_size[idim]
                            diff1[idim]=-1.0*difftemp
                            diff2[idim]=difftemp
                        diff1[3]=np.sqrt((diff1[0]*diff1[0]+diff1[1]*diff1[1]+diff1[2]*diff1[2]))
                        diff2[3]=diff1[3]
                        fc=0.5*(math.cos(3.141592*diff1[3]/rcutrho)+1.0)
                        for igauss in range(4):
                            diffadd1[igauss]=np.exp(-dgauss[igauss,1]*(diff1[3]-dgauss[igauss,0])*(diff1[3]-dgauss[igauss,0]))
                            diffadd2[igauss]=diffadd1[igauss]
                        diffadd1raw[0]=diff1[3]
                        diffadd2raw[0]=diffadd1raw[0]
                        diffadd1fc[0]=fc
                        diffadd2fc[0]=diffadd1fc[0]
                        
                        for idim in range(3):
                            smuv1[idim]=diff1[idim]/diff1[3]*fc
                            smuv2[idim]=diff2[idim]/diff2[3]*fc
                            uv1[idim]=diff1[idim]/diff1[3]
                            uv2[idim]=diff2[idim]/diff2[3]
                        edge_attr.append(diffadd1)
                        edge_attr.append(diffadd2)
                        edge_attrraw.append(diffadd1raw)
                        edge_attrraw.append(diffadd2raw)
                        edge_attrfc.append(diffadd1fc)
                        edge_attrfc.append(diffadd2fc)
                        uvec.append(uv1)
                        uvec.append(uv2)
                        uvec2.append(smuv1)
                        uvec2.append(smuv2)
                        
                x=xdat
                y=np.zeros((nnode,3))
                for iat in range(nnode):
                    y[iat,0]=datsave[iat,4]
                    y[iat,1]=datsave[iat,5]
                    y[iat,2]=datsave[iat,6]
                xtorch=torch.tensor(xdat,dtype=torch.float)
                ytorch=torch.tensor(y,dtype=torch.float)
                                
                edgetorch=torch.tensor(np.transpose(edge_index),dtype=torch.long)
                
                edgeattr_torch=torch.tensor(np.asarray(edge_attr),dtype=torch.float)
                edgeattrraw_torch=torch.tensor(np.asarray(edge_attrraw),dtype=torch.float)
                edgeattrfc_torch=torch.tensor(np.asarray(edge_attrfc),dtype=torch.float)
                
                uvec_torch=torch.tensor(np.asarray(uvec),dtype=torch.float)  
                uvec_torch2=torch.tensor(np.asarray(uvec2),dtype=torch.float)  
                
                
                data_ru2 = Data(x=xtorch, edge_index=edgetorch,edge_attr=edgeattrraw_torch, y=ytorch, uvec=uvec_torch2)
                data_fu2 = Data(x=xtorch, edge_index=edgetorch,edge_attr=edgeattrfc_torch, y=ytorch, uvec=uvec_torch2)
                data_gu2 = Data(x=xtorch, edge_index=edgetorch,edge_attr=edgeattr_torch, y=ytorch, uvec=uvec_torch2)
                
                frmod=ifr%10
                if frmod==9 or frmod==0:
                    fname='_test_'+str(testcount)+'.pt'
                    testcount+=1
                else:
                    fname='_train_'+str(traincount)+'.pt'
                    traincount+=1                
                torch.save(data_ru2,os.path.join(self.processed_dir,'ru2'+fname))
                torch.save(data_fu2,os.path.join(self.processed_dir,'fu2'+fname))
                torch.save(data_gu2,os.path.join(self.processed_dir,'gu2'+fname))
            rf.close()
        print("counts",traincount,testcount)
        self.nframetest=testcount
        self.nframetrain=traincount
        self.nframetot=self.nframetrain+self.nframetest
        
    def len(self):
        if self.test:
            tlen=self.nframetest
        else:
            tlen=self.nframetrain
        return tlen

    def get(self, idx):
        if self.test:
            data = torch.load(os.path.join(self.processed_dir,str(self.prefixer)+'_test_'+str(idx)+'.pt'))
        else:
            data = torch.load(os.path.join(self.processed_dir,str(self.prefixer)+'_train_'+str(idx)+'.pt'))        
        return data    