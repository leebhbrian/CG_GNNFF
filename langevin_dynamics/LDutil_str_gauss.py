import time
import sys
import numpy as np
import torch
import torch.nn.functional as F 
from torch.nn import Linear, BatchNorm1d, ModuleList
import torch_geometric
from GNN_force_strach import GNN_nb, GNN_bond
from torch_geometric.data import Data
from scipy.spatial import cKDTree
from transconv2 import TransformerConv2
import math

def initialize(fname):
    rf=open(fname+'.data','r')
    tr=rf.readline().split()
    nnode=int(tr[1])
    box_size=np.zeros(3)
    positions=np.zeros((nnode,3))
    typer=np.zeros((nnode,1))
    masser=np.zeros((nnode,1))
    tr=rf.readline().split()
    for idim in range(3):
        box_size[idim]=float(tr[idim])
    rf.readline()
    for iat in range(nnode):
        tr=rf.readline().split()
        typer[iat,0]=float(tr[0])
        if typer[iat,0]==1:
            masser[iat,0]=15.999*2+14.007
        else:
            masser[iat,0]=14.007*3+12.01*3+1.0079*6
        for idim in range(3):
            positions[iat,idim]=float(tr[idim+1])
        for idim in range(3):
            if positions[iat,idim]<0.0:
                positions[iat,idim]=positions[iat,idim]+box_size[idim]
            elif positions[iat,idim]>box_size[idim]:
                positions[iat,idim]=positions[iat,idim]-box_size[idim]
    rf.close()
    return masser,typer,positions,box_size
    
def finfer(pos,typer,box_size,device,modelnb,modelbond,xtype):
    nnode=np.shape(pos)[0]
    gvol=(box_size[0]*box_size[1]*box_size[2])*10**(-24)
    grho=(nnode/4)*222.117/6.02214/10**23/gvol
    dgauss=np.array([[2,5],[2.7,20],[3.1,20],[4,5]])
    impgauss=np.array([[0,5],[0.5,3],[1.3,20],[1.8,20]])
    agauss=np.array([[1.047197,10],[1.396263,20],[2.094395,5],[2.792526,10]])
    nbdgauss=np.array([[0,0.15],[3.5,2],[5.5,5],[6.4,5]])
    
    if xtype=='n':
        xdat=np.zeros((nnode,2))
    else:
        xdat=np.zeros((nnode,3))
    xdatnb=np.zeros((nnode,2))
    rcutrho=9.0
    wconst=84.0/(5.0*3.141592*rcutrho**3.0)
    for iat in range(nnode):
        if typer[iat,0]==1:
            xdat[iat,0]=1.0
            xdat[iat,1]=0.0
            xdatnb[iat,0]=1.0
            xdatnb[iat,1]=0.0
        elif typer[iat,0]==2:
            xdat[iat,0]=0.0
            xdat[iat,1]=1.0
            xdatnb[iat,0]=0.0
            xdatnb[iat,1]=1.0
    ## non-bonded edges
    receivers_list =cKDTree(pos, boxsize=box_size).query_pairs(r=9.0)    
    nbedge_index=[]
    nbedge_attr=[]
    nbuvec=[]
    count=0
    nb_prior=np.zeros((nnode,3))
    knb1=105.22
    dnb1=2.5
    dnblim1=2.5
    dnbsub1=-2.0*knb1*(dnblim1-dnb1)
    
    moldat=np.zeros((nnode,1))
    nmol=int(nnode/4)
    atcount=-1
    ccount=np.zeros((nnode,1))
    for imol in range(nmol):
        for isub in range(4):
            atcount+=1
            moldat[atcount,0]=imol
            
    for a in receivers_list:
        diff1=np.zeros(4)
        diff2=np.zeros(4)
        diffadd1=np.zeros(4)
        diffadd2=np.zeros(4)
        uv1=np.zeros(3)
        uv2=np.zeros(3)
        for idim in range(3):
            difftemp=pos[a[0],idim]-pos[a[1],idim]
            if difftemp>(box_size[idim]/2.0):
                difftemp=difftemp-box_size[idim]
            elif difftemp<(-box_size[idim]/2.0):
                difftemp=difftemp+box_size[idim]
            diff1[idim]=-1.0*difftemp
            diff2[idim]=difftemp
        diff1[3]=np.sqrt((diff1[0]*diff1[0]+diff1[1]*diff1[1]+diff1[2]*diff1[2]))
        diff2[3]=np.sqrt((diff2[0]*diff2[0]+diff2[1]*diff2[1]+diff2[2]*diff2[2]))
        nbdnow=diff1[3]
        nbdorig=nbdnow
        nbdpcond=False
        if nbdnow<dnblim1:
            nbdnow=dnblim1
            nbdpcond=True
        fc=0.5*(math.cos(3.141592*nbdnow/rcutrho)+1.0)
        weight=wconst*(1+(3.0*diff1[3]/2.0/rcutrho))*(1-(diff1[3]/rcutrho))**4.0
        ccount[a[0],0]=ccount[a[0],0]+weight
        ccount[a[1],0]=ccount[a[1],0]+weight
        for igauss in range(4):
            diffadd1[igauss]=np.exp(-dgauss[igauss,1]*(nbdnow-dgauss[igauss,0])*(nbdnow-dgauss[igauss,0]))
            diffadd2[igauss]=diffadd1[igauss]

        for idim in range(3):
            uv1[idim]=diff1[idim]/diff1[3]*fc
            uv2[idim]=diff2[idim]/diff2[3]*fc
        if moldat[a[0],0]!=moldat[a[1],0]:
            nbedge_index.append([a[0],a[1]])
            nbedge_index.append([a[1],a[0]])
            nbuvec.append(uv1)
            nbuvec.append(uv2)            
            nbedge_attr.append(diffadd1)
            nbedge_attr.append(diffadd2)
            if nbdpcond:
                fmag=-2.0*knb1*(nbdorig-dnb1)-dnbsub1
                for idim in range(3):
                    nb_prior[a[0],idim]=nb_prior[a[0],idim]-fmag*uv1[idim]
                    nb_prior[a[1],idim]=nb_prior[a[1],idim]-fmag*uv2[idim]
                    
            
    if xtype=='l':
        for iat in range(nnode):
            xdat[iat,2]=ccount[iat,0]
    elif xtype=='g':
        xdat[:,2]=grho
    ## bonded edges
    nmol=int(nnode/4)
    bondedge_index=[]
    bondedge_attr=[]
    bonduvec=[]
    bondedge_index2=[]
    bondedge_attr2=[]
    bonduvec2=[]
    bondedge_index3=[]
    bondedge_attr3=[]
    bonduvec3=[]
    bondedge_index4=[]
    bondedge_attr4=[]
    bonduvec4=[]
    count=0
    bond_prior=np.zeros((nnode,3))
    
    kdng1=176.09
    d01=2.4
    d0lim1=2.0
    d0sub1=-2.0*kdng1*(d0lim1-d01)
   
    kdng2=159.08
    d02=3.2
    d0lim2=3.6
    d0sub2=-2.0*kdng2*(d0lim2-d02)
    
    kang1=266.564
    the1=1.22173
    thelim1=1.047197
    thesub1=-2.0*kang1*(thelim1-the1)
    
    kang2=49.5704
    the2=2.617993
    thelim2=2.792526
    thesub2=-2.0*kang2*(thelim2-the2)
    for imol in range(nmol):
        diffs=np.zeros((3,3))
        dsave=np.zeros(3)
        uvsave=np.zeros((3,3,2))
        for igroup in range(3):
            ng=imol*4+igroup
            nr=imol*4+3
                               
            diff1=np.zeros(4)
            diff2=np.zeros(4)
            diffadd1=np.zeros(1)
            diffadd2=np.zeros(1)
            uv1=np.zeros(3)
            uv2=np.zeros(3)
            for idim in range(3):
                difftemp=pos[ng,idim]-pos[nr,idim]
                if difftemp>(box_size[idim]/2.0):
                    difftemp=difftemp-box_size[idim]
                elif difftemp<(-box_size[idim]/2.0):
                    difftemp=difftemp+box_size[idim]
                diff1[idim]=-1.0*difftemp
                diff2[idim]=difftemp
                diffs[igroup,idim]=-1.0*difftemp
            diff1[3]=np.sqrt((diff1[0]*diff1[0]+diff1[1]*diff1[1]+diff1[2]*diff1[2]))
            diff2[3]=diff1[3]
            dnow=diff1[3]
            dorig=dnow
            bdpcond1=False
            bdpcond2=False
            if dnow<d0lim1:
                dnow=d0lim1
                bdpcond1=True
            elif dnow>d0lim2:
                dnow=d0lim2
                bdpcond2=True
            dsave[igroup]=diff1[3]   
            for idim in range(3):
                uvsave[igroup,idim,0]=diff1[idim]/diff1[3]
                uvsave[igroup,idim,1]=diff2[idim]/diff2[3] 
            
            if bdpcond1:            
                fmag=-2.0*kdng1*(dorig-d01)-d0sub1
                for idim in range(3):
                    bond_prior[ng,idim]=bond_prior[ng,idim]-fmag*uv1[idim]
                    bond_prior[nr,idim]=bond_prior[nr,idim]-fmag*uv2[idim]
            elif bdpcond2:
                fmag=-2.0*kdng2*(dorig-d02)-d0sub2
                for idim in range(3):
                    bond_prior[ng,idim]=bond_prior[ng,idim]-fmag*uv1[idim]
                    bond_prior[nr,idim]=bond_prior[nr,idim]-fmag*uv2[idim]
            
        addmat=[(0,1),(0,2),(1,2)]    
        asave=np.zeros((3,2))
        acounter=np.zeros(3)
        for igroup in range(3):
            add1,add2=addmat[igroup]
            ng=imol*4+add1
            nr=imol*4+3                                     
            diffadd1=np.zeros(12)
            diffadd2=np.zeros(12)
            angnow=calcang(diffs[add1,:],diffs[add2,:])
            ur1=np.linalg.norm(diffs[add1,:])
            ur2=np.linalg.norm(diffs[add2,:])
            if ur1<d0lim1:
                ur1=d0lim1
            elif ur1>d0lim2:
                ur1=d0lim2
            if ur2<d0lim1:
                ur2=d0lim1
            elif ur2>d0lim2:
                ur2=d0lim2
            angorig=angnow
            bapcond1=False
            bapcond2=False
            if angnow<thelim1:
                angnow=thelim1
                bapcond1=True
            elif angnow>thelim2:
                angnow=thelim2
                bapcond2=True
                
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
            
            for igauss in range(4):
                diffadd1[igauss]=np.exp(-agauss[igauss,1]*(angnow-agauss[igauss,0])*(angnow-agauss[igauss,0]))
                diffadd2[igauss]=diffadd1[igauss]
            for igauss in range(4):
                diffadd1[igauss+4]=np.exp(-dgauss[igauss,1]*(ur1-dgauss[igauss,0])*(ur1-dgauss[igauss,0]))
                diffadd2[igauss+4]=diffadd1[igauss+4]
            for igauss in range(4):
                diffadd1[igauss+8]=np.exp(-dgauss[igauss,1]*(ur2-dgauss[igauss,0])*(ur2-dgauss[igauss,0]))
                diffadd2[igauss+8]=diffadd1[igauss+8]
                                
            uv1=dirang(diffs[add1,:],diffs[add2,:])/ur1
            uv2=-uv1
            if igroup==0:
                bondedge_index2.append([ng,nr])
                bondedge_index2.append([nr,ng])  
                bondedge_attr2.append(diffadd1)
                bondedge_attr2.append(diffadd2)
                bonduvec2.append(uv1)
                bonduvec2.append(uv2)
            elif igroup==1:
                bondedge_index3.append([ng,nr])
                bondedge_index3.append([nr,ng])  
                bondedge_attr3.append(diffadd1)
                bondedge_attr3.append(diffadd2)
                bonduvec3.append(uv1)
                bonduvec3.append(uv2)
            elif igroup==2:
                bondedge_index4.append([ng,nr])
                bondedge_index4.append([nr,ng])  
                bondedge_attr4.append(diffadd1)
                bondedge_attr4.append(diffadd2)
                bonduvec4.append(uv1)
                bonduvec4.append(uv2)
            if bapcond1:            
                fmag=-2.0*kang1*(angorig-the1)-thesub1
                for idim in range(3):
                    bond_prior[ng,idim]=bond_prior[ng,idim]-fmag*uv1[idim]
                    bond_prior[nr,idim]=bond_prior[nr,idim]-fmag*uv2[idim]
            elif bapcond2:
                fmag=-2.0*kang2*(angorig-the2)-thesub2
                for idim in range(3):
                    bond_prior[ng,idim]=bond_prior[ng,idim]-fmag*uv1[idim]
                    bond_prior[nr,idim]=bond_prior[nr,idim]-fmag*uv2[idim]
            
            ng=imol*4+add2
            nr=imol*4+3                        
            uv1=dirang(diffs[add2,:],diffs[add1,:])/ur2
            uv2=-uv1
            
            if igroup==0:
                bondedge_index2.append([ng,nr])
                bondedge_index2.append([nr,ng])  
                bondedge_attr2.append(diffadd1)
                bondedge_attr2.append(diffadd2)
                bonduvec2.append(uv1)
                bonduvec2.append(uv2)
            elif igroup==1:
                bondedge_index3.append([ng,nr])
                bondedge_index3.append([nr,ng])  
                bondedge_attr3.append(diffadd1)
                bondedge_attr3.append(diffadd2)
                bonduvec3.append(uv1)
                bonduvec3.append(uv2)
            elif igroup==2:
                bondedge_index4.append([ng,nr])
                bondedge_index4.append([nr,ng])  
                bondedge_attr4.append(diffadd1)
                bondedge_attr4.append(diffadd2)
                bonduvec4.append(uv1)
                bonduvec4.append(uv2)
            if bapcond1:            
                fmag=-2.0*kang1*(angorig-the1)-thesub1
                for idim in range(3):
                    bond_prior[ng,idim]=bond_prior[ng,idim]-fmag*uv1[idim]
                    bond_prior[nr,idim]=bond_prior[nr,idim]-fmag*uv2[idim]
            elif bapcond2:
                fmag=-2.0*kang2*(angorig-the2)-thesub2
                for idim in range(3):
                    bond_prior[ng,idim]=bond_prior[ng,idim]-fmag*uv1[idim]
                    bond_prior[nr,idim]=bond_prior[nr,idim]-fmag*uv2[idim]
                    
        for igroup in range(3):
            ng=imol*4+igroup
            nr=imol*4+3 
            diffadd1=np.zeros(12)
            diffadd2=np.zeros(12)
            uv1=np.zeros(3)
            uv2=np.zeros(3)
            for idim in range(3):
                uv1[idim]=uvsave[igroup,idim,0]
                uv2[idim]=uvsave[igroup,idim,1]
            
            for igauss in range(4):
                diffadd1[igauss]=np.exp(-dgauss[igauss,1]*(dsave[igroup]-dgauss[igauss,0])*(dsave[igroup]-dgauss[igauss,0]))
                diffadd2[igauss]=diffadd1[igauss]
            for igauss in range(4):
                diffadd1[igauss+4]=np.exp(-agauss[igauss,1]*(asave[igroup,0]-agauss[igauss,0])*(asave[igroup,0]-agauss[igauss,0]))
                diffadd2[igauss+4]=diffadd1[igauss+4]
            for igauss in range(4):
                diffadd1[igauss+8]=np.exp(-agauss[igauss,1]*(asave[igroup,1]-agauss[igauss,0])*(asave[igroup,1]-agauss[igauss,0]))
                diffadd2[igauss+8]=diffadd1[igauss+8]
            
            bondedge_index.append([ng,nr])
            bondedge_index.append([nr,ng]) 
            bonduvec.append(uv1)
            bonduvec.append(uv2)
            bondedge_attr.append(diffadd1)
            bondedge_attr.append(diffadd2)
            
    xtorch=torch.tensor(xdat,dtype=torch.float)
    xnbtorch=torch.tensor(xdatnb,dtype=torch.float)
    nbedgetorch=torch.tensor(np.transpose(nbedge_index),dtype=torch.long)
    nbedgeattr_torch=torch.tensor(np.asarray(nbedge_attr),dtype=torch.float)
    nbuvec_torch=torch.tensor(np.asarray(nbuvec),dtype=torch.float)       
    
    bondedgetorch=torch.tensor(np.transpose(bondedge_index),dtype=torch.long)
    bondedgetorch2=torch.tensor(np.transpose(bondedge_index2),dtype=torch.long)
    bondedgetorch3=torch.tensor(np.transpose(bondedge_index3),dtype=torch.long)
    bondedgetorch4=torch.tensor(np.transpose(bondedge_index4),dtype=torch.long)
    
    bondedgeattr_torch=torch.tensor(np.asarray(bondedge_attr),dtype=torch.float)
    bondedgeattr_torch2=torch.tensor(np.asarray(bondedge_attr2),dtype=torch.float)
    bondedgeattr_torch3=torch.tensor(np.asarray(bondedge_attr3),dtype=torch.float)
    bondedgeattr_torch4=torch.tensor(np.asarray(bondedge_attr4),dtype=torch.float)
    
    bonduvec_torch=torch.tensor(np.asarray(bonduvec),dtype=torch.float)                
    bonduvec_torch2=torch.tensor(np.asarray(bonduvec2),dtype=torch.float)                
    bonduvec_torch3=torch.tensor(np.asarray(bonduvec3),dtype=torch.float)                
    bonduvec_torch4=torch.tensor(np.asarray(bonduvec4),dtype=torch.float) 

    datanb = Data(x=xnbtorch, edge_index=nbedgetorch,edge_attr=nbedgeattr_torch, uvec=nbuvec_torch).to(device)
    databond = Data(x=xtorch, edge_index=bondedgetorch,edge_attr=bondedgeattr_torch, uvec=bonduvec_torch,
                edge_index2=bondedgetorch2,edge_attr2=bondedgeattr_torch2, uvec2=bonduvec_torch2,
                edge_index3=bondedgetorch3,edge_attr3=bondedgeattr_torch3, uvec3=bonduvec_torch3,
                edge_index4=bondedgetorch4,edge_attr4=bondedgeattr_torch4, uvec4=bonduvec_torch4).to(device)

    
    with torch.no_grad():
        prednb=modelnb(datanb.x.float(),datanb.edge_attr,datanb.edge_index,datanb.uvec)
        predbond=modelbond(databond.x.float(), databond.edge_attr.float(),databond.edge_index,databond.uvec, databond.edge_attr2.float(),databond.edge_index2,databond.uvec2, databond.edge_attr3.float(),databond.edge_index3,databond.uvec3, databond.edge_attr4.float(),databond.edge_index4,databond.uvec4)
    bondtot=predbond.cpu().detach().numpy()+bond_prior
    for imol in range(nmol):
        ider=imol*4+3
        bondtot[ider,:]=0.0
        for iat in range(3):
            bondtot[ider,:]=bondtot[ider,:]-bondtot[imol*4+iat,:]
    predtot=prednb.cpu().detach().numpy()+nb_prior+bondtot
    return predtot
    
def createvel(masser,temper):
    kbT=1.380649*10**(-23)*temper*1000.0*6.02214*10**(23)
    xi = np.random.standard_normal((len(masser), 3))
    vel0 = xi * np.sqrt(kbT/masser)*10**(-5)
    vel=vel0-np.mean(vel0,axis=0)
    return vel

def calctemp(masser,vel):
    const=1.0/(1.380649*10**(-23)*1000.0*6.02214*10**(23))
    tsq=np.square(vel)
    tx=tsq[:,0]*masser*const*10**(10)
    ty=tsq[:,1]*masser*const*10**(10)
    tz=tsq[:,2]*masser*const*10**(10)
    tempall=np.reshape(np.sum(tsq,axis=1),(-1,1))*masser/3.0*const*10**(10)
    tmat=np.array([np.mean(tx),np.mean(ty),np.mean(tz),np.mean(tempall)])
    return tmat
    
def rearrange(pos,box_size):
    for iat in range(len(pos)):
        for idim in range(3):
            if pos[iat,idim]>box_size[idim]:
                pos[iat,idim]-=box_size[idim]
            elif pos[iat,idim]<0:
                pos[iat,idim]+=box_size[idim]
    return pos

def outdump(pos,stem,wtype):
    wf=open(stem+'.xyz',wtype)
    npos=len(pos)
    nmol=int(npos/4)
    wf.write(str(npos)+"\n\n")
    aid=-1
    for imol in range(nmol):
        for ig in range(4):
            if ig==3:
                atype=2
            else:
                atype=1
            aid+=1
            tw=str(atype)+"\t"+str(pos[aid,0])+"\t"+str(pos[aid,1])+"\t"+str(pos[aid,2])+"\n"
            wf.write(tw)

#this is step A
def position_update(x,v,dt):
    x_new = x + v*dt/2.0
    return x_new

#this is step B
def velocity_update(v,F,dt,masser):
    v_new = v + F*dt/2.0/masser
    return v_new

def random_velocity_update(vel,c1,c2):
    R = np.random.standard_normal((np.shape(vel)))
    v_new = c1*vel + R*c2
    return v_new
    
def calcang(vec1,vec2):
    uvec1 = vec1 / np.linalg.norm(vec1)
    uvec2 = vec2 / np.linalg.norm(vec2)
    dotter = np.dot(uvec1,uvec2)
    angle = np.arccos(dotter)
    return angle
        
def dirang(vec1,vec2):
    dir1=np.cross(vec1,np.cross(vec1,vec2))
    return dir1/np.linalg.norm(dir1)

def integrator(pos,vel,forcer_old,masser,typer,box_size,dt,c1,c2,modelnb,modelbond,device,xtype):
    #B
    fconv=1.0*4184*1000.0*10**(-30)/(10**(-20))
    vel=velocity_update(vel,forcer_old*fconv,dt,masser)
    #A
    pos = position_update(pos,vel,dt)

    #O
    vel = random_velocity_update(vel,c1,c2)

    #A
    pos = position_update(pos,vel,dt)

    # B
    pos=rearrange(pos,box_size)
    forcer=finfer(pos,typer,box_size,device,modelnb,modelbond,xtype)
    #forcer=np.zeros((np.shape(pos)))
    vel=velocity_update(vel,forcer*fconv,dt,masser)
    
    return pos,vel,forcer            
    