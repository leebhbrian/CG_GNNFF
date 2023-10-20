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
        self.prefixer=prefixer
        self.nframetest=60
        self.nframetrain=240
        self.nframetot=self.nframetrain+self.nframetest
        self.fmat=['intra_g0.data']
        self.flen=[300]
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
        dgauss=np.array([[2,5],[2.7,20],[3.1,20],[4,5]])
        impgauss=np.array([[0,5],[0.5,3],[1.3,20],[1.8,20]])
        agauss=np.array([[1.047197,10],[1.396263,20],[2.094395,5],[2.792526,10]])
        rcutrho=9.0
        wconst=84.0/(5.0*3.141592*rcutrho**3.0)
        
        for ifile in range(len(self.fmat)):
            fname=self.fmat[ifile]
            rf=open(os.path.join(self.raw_dir,fname),'r')
            print("fname",fname)
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
                gvol=(box_size[0]*box_size[1]*box_size[2])*10**(-24)
                grho=(nnode/4)*222.117/6.02214/10**23/gvol
                xdat=np.zeros((nnode,2))
                xdatg=np.zeros((nnode,3))
                xdatl=np.zeros((nnode,3))
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
                        xdatl[iat,0]=1.0
                        xdatl[iat,1]=0.0
                        xdatg[iat,0]=1.0
                        xdatg[iat,1]=0.0
                        xdatg[iat,2]=grho
                    elif datsave[iat,0]==2:
                        xdat[iat,0]=0.0
                        xdat[iat,1]=1.0
                        xdatl[iat,0]=0.0
                        xdatl[iat,1]=1.0
                        xdatg[iat,0]=0.0
                        xdatg[iat,1]=1.0
                        xdatg[iat,2]=grho
                        
                num_nodes = len(positions)
                nmol=int(num_nodes/4)
                edge_index=[]
                edge_attr_raw=[]
                edge_attr_gauss=[]
                uvec=[]
                edge_index2=[]
                edge_attr_raw2=[]
                edge_attr_gauss2=[]
                uvec2=[]
                edge_index3=[]
                edge_attr_raw3=[]
                edge_attr_gauss3=[]
                uvec3=[]
                edge_index4=[]
                edge_attr_raw4=[]
                edge_attr_gauss4=[]
                uvec4=[]

                count=0
                receivers_list =cKDTree(positions, boxsize=box_size).query_pairs(r=9.0)    
                ccount=np.zeros((nnode,1))
                moldat=np.zeros((nnode,1))
                atcount=-1
                for imol in range(nmol):
                    for isub in range(4):
                        atcount+=1
                        moldat[atcount,0]=imol
                for a in receivers_list:
                    diff1=np.zeros(4)
                    diff2=np.zeros(4)
                    diffadd1=np.zeros(1)
                    diffadd2=np.zeros(1)
                    uv1=np.zeros(3)
                    uv2=np.zeros(3)
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
                    weight=wconst*(1+(3.0*diff1[3]/2.0/rcutrho))*(1-(diff1[3]/rcutrho))**4.0
                    ccount[a[0],0]=ccount[a[0],0]+weight
                    ccount[a[1],0]=ccount[a[1],0]+weight
                for iat in range(nnode):
                    xdatl[iat,2]=ccount[iat,0]
                    
                for imol in range(nmol):
                    diffs=np.zeros((3,3))
                    ####Bond forces
                    dsave=np.zeros(3)
                    uvsave=np.zeros((3,3,2))
                    for igroup in range(3):
                        ng=imol*4+igroup
                        nr=imol*4+3                    
                        diff1=np.zeros(4)
                        diff2=np.zeros(4)
                        
                        for idim in range(3):
                            difftemp=positions[ng,idim]-positions[nr,idim]
                            if difftemp>(box_size[idim]/2.0):
                                difftemp=difftemp-box_size[idim]
                            elif difftemp<(-box_size[idim]/2.0):
                                difftemp=difftemp+box_size[idim]
                            diff1[idim]=-1.0*difftemp
                            diff2[idim]=difftemp
                            diffs[igroup,idim]=-1.0*difftemp
                        diff1[3]=np.sqrt((diff1[0]*diff1[0]+diff1[1]*diff1[1]+diff1[2]*diff1[2]))
                        diff2[3]=np.sqrt((diff2[0]*diff2[0]+diff2[1]*diff2[1]+diff2[2]*diff2[2]))
                        dsave[igroup]=diff1[3]   
                        for idim in range(3):
                            uvsave[igroup,idim,0]=diff1[idim]/diff1[3]
                            uvsave[igroup,idim,1]=diff2[idim]/diff2[3]  
                            
                    ####Angular forces
                    addmat=[(0,1),(0,2),(1,2)]    
                    asave=np.zeros((3,2))
                    acounter=np.zeros(3)
                    for igroup in range(3):
                        add1,add2=addmat[igroup]
                        ng=imol*4+add1
                        nr=imol*4+3                        
                        diffadd1raw=np.zeros(3)
                        diffadd2raw=np.zeros(3)
                        diffadd1gauss=np.zeros(12)
                        diffadd2gauss=np.zeros(12)
                        angnow=self.calcang(diffs[add1,:],diffs[add2,:])
                        ur1=np.linalg.norm(diffs[add1,:])
                        ur2=np.linalg.norm(diffs[add2,:])
                        if acounter[add1]>0.1:
                            asave[add1,1]=angnow
                        else:
                            asave[add1,0]=angnow
                            acounter[add1]=10
                        if acounter[add2]>0.1:
                            asave[add2,1]=angnow
                        else:
                            asave[add2,0]=angnow
                            acounter[add2]=10
                        diffadd1raw[0]=angnow
                        diffadd2raw[0]=angnow
                        diffadd1raw[1]=ur1
                        diffadd2raw[1]=ur1
                        diffadd1raw[2]=ur2
                        diffadd2raw[2]=ur2
                        
                        for igauss in range(4):
                            diffadd1gauss[igauss]=np.exp(-agauss[igauss,1]*(angnow-agauss[igauss,0])*(angnow-agauss[igauss,0]))
                            diffadd2gauss[igauss]=diffadd1gauss[igauss]
                        for igauss in range(4):
                            diffadd1gauss[igauss+4]=np.exp(-dgauss[igauss,1]*(ur1-dgauss[igauss,0])*(ur1-dgauss[igauss,0]))
                            diffadd2gauss[igauss+4]=diffadd1gauss[igauss+4]
                        for igauss in range(4):
                            diffadd1gauss[igauss+8]=np.exp(-dgauss[igauss,1]*(ur2-dgauss[igauss,0])*(ur2-dgauss[igauss,0]))
                            diffadd2gauss[igauss+8]=diffadd1gauss[igauss+8]
                        
                        uv1=self.calcdirang(diffs[add1,:],diffs[add2,:])/ur1
                        uv2=-uv1
                        if igroup==0:
                            edge_index2.append([ng,nr])
                            edge_index2.append([nr,ng])  
                            edge_attr_raw2.append(diffadd1raw)
                            edge_attr_raw2.append(diffadd2raw)
                            edge_attr_gauss2.append(diffadd1gauss)
                            edge_attr_gauss2.append(diffadd2gauss)
                            uvec2.append(uv1)
                            uvec2.append(uv2)
                        elif igroup==1:
                            edge_index3.append([ng,nr])
                            edge_index3.append([nr,ng])  
                            edge_attr_raw3.append(diffadd1raw)
                            edge_attr_raw3.append(diffadd2raw)
                            edge_attr_gauss3.append(diffadd1gauss)
                            edge_attr_gauss3.append(diffadd2gauss)
                            uvec3.append(uv1)
                            uvec3.append(uv2)
                        elif igroup==2:
                            edge_index4.append([ng,nr])
                            edge_index4.append([nr,ng])  
                            edge_attr_raw4.append(diffadd1raw)
                            edge_attr_raw4.append(diffadd2raw)
                            edge_attr_gauss4.append(diffadd1gauss)
                            edge_attr_gauss4.append(diffadd2gauss)
                            uvec4.append(uv1)
                            uvec4.append(uv2)
                        
                        ng=imol*4+add2
                        nr=imol*4+3                        
                        uv1=self.calcdirang(diffs[add2,:],diffs[add1,:])/ur2
                        uv2=-uv1
                        if igroup==0:
                            edge_index2.append([ng,nr])
                            edge_index2.append([nr,ng])  
                            edge_attr_raw2.append(diffadd1raw)
                            edge_attr_raw2.append(diffadd2raw)
                            edge_attr_gauss2.append(diffadd1gauss)
                            edge_attr_gauss2.append(diffadd2gauss)                            
                            uvec2.append(uv1)
                            uvec2.append(uv2)
                        elif igroup==1:
                            edge_index3.append([ng,nr])
                            edge_index3.append([nr,ng])  
                            edge_attr_raw3.append(diffadd1raw)
                            edge_attr_raw3.append(diffadd2raw)
                            edge_attr_gauss3.append(diffadd1gauss)
                            edge_attr_gauss3.append(diffadd2gauss)
                            uvec3.append(uv1)
                            uvec3.append(uv2)
                        elif igroup==2:
                            edge_index4.append([ng,nr])
                            edge_index4.append([nr,ng])  
                            edge_attr_raw4.append(diffadd1raw)
                            edge_attr_raw4.append(diffadd2raw)
                            edge_attr_gauss4.append(diffadd1gauss)
                            edge_attr_gauss4.append(diffadd2gauss)
                            uvec4.append(uv1)
                            uvec4.append(uv2)
                    for igroup in range(3):
                        ng=imol*4+igroup
                        nr=imol*4+3 
                        diffadd1raw=np.zeros(3)
                        diffadd2raw=np.zeros(3)
                        diffadd1gauss=np.zeros(12)
                        diffadd2gauss=np.zeros(12)
                        uv1=np.zeros(3)
                        uv2=np.zeros(3)
                        for idim in range(3):
                            uv1[idim]=uvsave[igroup,idim,0]
                            uv2[idim]=uvsave[igroup,idim,1]
                        
                        diffadd1raw[0]=dsave[igroup]
                        diffadd2raw[0]=dsave[igroup]
                        diffadd1raw[1]=asave[igroup,0]
                        diffadd2raw[1]=asave[igroup,0]
                        diffadd1raw[2]=asave[igroup,1]
                        diffadd2raw[2]=asave[igroup,1]
                        
                        for igauss in range(4):
                            diffadd1gauss[igauss]=np.exp(-dgauss[igauss,1]*(dsave[igroup]-dgauss[igauss,0])*(dsave[igroup]-dgauss[igauss,0]))
                            diffadd2gauss[igauss]=diffadd1gauss[igauss]
                        for igauss in range(4):
                            diffadd1gauss[igauss+4]=np.exp(-agauss[igauss,1]*(asave[igroup,0]-agauss[igauss,0])*(asave[igroup,0]-agauss[igauss,0]))
                            diffadd2gauss[igauss+4]=diffadd1gauss[igauss+4]
                        for igauss in range(4):
                            diffadd1gauss[igauss+8]=np.exp(-agauss[igauss,1]*(asave[igroup,1]-agauss[igauss,0])*(asave[igroup,1]-agauss[igauss,0]))
                            diffadd2gauss[igauss+8]=diffadd1gauss[igauss+8]
                        
                        edge_index.append([ng,nr])
                        edge_index.append([nr,ng])
                        edge_attr_raw.append(diffadd1raw)
                        edge_attr_raw.append(diffadd2raw)
                        edge_attr_gauss.append(diffadd1gauss)
                        edge_attr_gauss.append(diffadd2gauss)
                        uvec.append(uv1)
                        uvec.append(uv2)
                    
                y=np.zeros((nnode,3))
                for iat in range(nnode):
                    #y[iat,0]=np.sqrt((datsave[iat,4]*datsave[iat,4]+datsave[iat,5]*datsave[iat,5]+datsave[iat,6]*datsave[iat,6]))
                    y[iat,0]=datsave[iat,4]
                    y[iat,1]=datsave[iat,5]
                    y[iat,2]=datsave[iat,6]
                
                xtorchn=torch.tensor(xdat,dtype=torch.float)
                xtorchl=torch.tensor(xdatl,dtype=torch.float)
                ytorch=torch.tensor(y,dtype=torch.float)
                                
                edgetorch=torch.tensor(np.transpose(edge_index),dtype=torch.long)
                edgetorch2=torch.tensor(np.transpose(edge_index2),dtype=torch.long)
                edgetorch3=torch.tensor(np.transpose(edge_index3),dtype=torch.long)
                edgetorch4=torch.tensor(np.transpose(edge_index4),dtype=torch.long)
                
                edgeattrraw_torch=torch.tensor(np.asarray(edge_attr_raw),dtype=torch.float)
                edgeattrraw_torch2=torch.tensor(np.asarray(edge_attr_raw2),dtype=torch.float)
                edgeattrraw_torch3=torch.tensor(np.asarray(edge_attr_raw3),dtype=torch.float)
                edgeattrraw_torch4=torch.tensor(np.asarray(edge_attr_raw4),dtype=torch.float)
                
                edgeattrgauss_torch=torch.tensor(np.asarray(edge_attr_gauss),dtype=torch.float)
                edgeattrgauss_torch2=torch.tensor(np.asarray(edge_attr_gauss2),dtype=torch.float)
                edgeattrgauss_torch3=torch.tensor(np.asarray(edge_attr_gauss3),dtype=torch.float)
                edgeattrgauss_torch4=torch.tensor(np.asarray(edge_attr_gauss4),dtype=torch.float)
                
                uvec_torch=torch.tensor(np.asarray(uvec),dtype=torch.float)                
                uvec_torch2=torch.tensor(np.asarray(uvec2),dtype=torch.float)                
                uvec_torch3=torch.tensor(np.asarray(uvec3),dtype=torch.float)                
                uvec_torch4=torch.tensor(np.asarray(uvec4),dtype=torch.float)                
                                                
                datalr = Data(x=xtorchl, edge_index=edgetorch, edge_index2=edgetorch2, edge_index3=edgetorch3, edge_index4=edgetorch4
                                ,edge_attr=edgeattrraw_torch,edge_attr2=edgeattrraw_torch2,edge_attr3=edgeattrraw_torch3,edge_attr4=edgeattrraw_torch4
                                ,y=ytorch, uvec=uvec_torch, uvec2=uvec_torch2, uvec3=uvec_torch3, uvec4=uvec_torch4)                                

                datalg = Data(x=xtorchl, edge_index=edgetorch, edge_index2=edgetorch2, edge_index3=edgetorch3, edge_index4=edgetorch4
                                ,edge_attr=edgeattrgauss_torch,edge_attr2=edgeattrgauss_torch2,edge_attr3=edgeattrgauss_torch3,edge_attr4=edgeattrgauss_torch4
                                ,y=ytorch, uvec=uvec_torch, uvec2=uvec_torch2, uvec3=uvec_torch3, uvec4=uvec_torch4)                                   
                
                datanr = Data(x=xtorchn, edge_index=edgetorch, edge_index2=edgetorch2, edge_index3=edgetorch3, edge_index4=edgetorch4
                                ,edge_attr=edgeattrraw_torch,edge_attr2=edgeattrraw_torch2,edge_attr3=edgeattrraw_torch3,edge_attr4=edgeattrraw_torch4
                                ,y=ytorch, uvec=uvec_torch, uvec2=uvec_torch2, uvec3=uvec_torch3, uvec4=uvec_torch4)                                

                datang = Data(x=xtorchn, edge_index=edgetorch, edge_index2=edgetorch2, edge_index3=edgetorch3, edge_index4=edgetorch4
                                ,edge_attr=edgeattrgauss_torch,edge_attr2=edgeattrgauss_torch2,edge_attr3=edgeattrgauss_torch3,edge_attr4=edgeattrgauss_torch4
                                ,y=ytorch, uvec=uvec_torch, uvec2=uvec_torch2, uvec3=uvec_torch3, uvec4=uvec_torch4)   
                
                frmod=ifr%10
                if frmod==9 or frmod==0:
                    fname='test_'+str(testcount)+'.pt'
                    testcount+=1
                else:
                    fname='train_'+str(traincount)+'.pt'
                    traincount+=1                

                torch.save(datalg,os.path.join(self.processed_dir,'lg_'+fname))
                torch.save(datalr,os.path.join(self.processed_dir,'lr_'+fname))
                torch.save(datang,os.path.join(self.processed_dir,'ng_'+fname))
                torch.save(datanr,os.path.join(self.processed_dir,'nr_'+fname))
            rf.close()                
        print("counts",traincount,testcount)
        self.nframetest=testcount
        self.nframetrain=traincount
        self.nframetot=self.nframetrain+self.nframetest
    
    def calcang(self,vec1,vec2):
        uvec1 = vec1 / np.linalg.norm(vec1)
        uvec2 = vec2 / np.linalg.norm(vec2)
        dotter = np.dot(uvec1,uvec2)
        angle = np.arccos(dotter)
        return angle
        
    def calcdirang(self,vec1,vec2):
        dir1=np.cross(vec1,np.cross(vec1,vec2))
        return dir1/np.linalg.norm(dir1)
        
    def calcdirimp(self,r1,r2,r3):
        r31=np.zeros(3)
        r32=np.zeros(3)
        r31=r3[:]-r1[:]
        r32=r3[:]-r2[:]
        dir1=np.cross(r31,r32)
        return dir1/np.linalg.norm(dir1)
    
    def calcdihed(self,r1,r2,r3,r4):
        r12=np.zeros(3)
        r23=np.zeros(3)
        r14=np.zeros(3)
        r34=np.zeros(3)
        
        r12[:]=r1[:]-r2[:]
        r23[:]=r2[:]-r3[:]
        r14[:]=r1[:]-r4[:]
        r34[:]=r3[:]-r4[:]
        
        m=np.cross(r12,r23)
        n=np.cross(r14,r34)
        
        dotter=np.dot(m,n)
        Den=math.sqrt(m[0]*m[0]+m[1]*m[1]+m[2]*m[2])*math.sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2])
        div=dotter/Den
        if abs(div)>1:
            if div>0:
                div=1
            else:
                div=-1
        phi1=math.acos(div)
        return abs(phi1)
        
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