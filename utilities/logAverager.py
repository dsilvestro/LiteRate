import os
import sys
import pandas as pd
import argparse
import numpy as np
def calcHPD(data, level=0.95) :
    assert (0 < level < 1)	
    d = list(data)
    d.sort()
    nData = len(data)
    nIn = int(round(level * nData))
    if nIn < 2 :
        raise RuntimeError("not enough data")
    i = 0
    r = d[i+nIn-1] - d[i]
    for k in range(len(d) - (nIn - 1)) :
        rk = d[k+nIn-1] - d[k]
        if rk < r :
            r = rk
            i = k
    assert 0 <= i <= i+nIn-1 < len(d)
    return np.array([d[i], d[i+nIn-1]])


def print_R_vec(name,v):
    new_v=[]
    if len(v)==0: vec= "%s=c()" % (name)
    elif len(v)==1: vec= "%s=c(%s)" % (name,v[0])
    elif len(v)==2: vec= "%s=c(%s,%s)" % (name,v[0],v[1])
    else:
        for j in range(0,len(v)): 
            value=v[j]
            if np.isnan(v[j]): value="NA"
            new_v.append(value)

        vec="%s=c(%s, " % (name,new_v[0])
        for j in range(1,len(v)-1): vec += "%s," % (new_v[j])
        vec += "%s)"  % (new_v[j+1])
    return vec

p=argparse.ArgumentParser() #description='<input file>')
p.add_argument('input', type=str, default='.')
p.add_argument('-o', nargs='?', type=argparse.FileType('w'), default=sys.stdout)
p.add_argument('-head',type=int,help="Header?",default=1,metavar=1)

args=p.parse_args()
if args.head==True: head=0
else: head=None

dat=pd.read_csv(args.input,sep='\t',header=head)
means=[]
minHPD=[]; maxHPD=[]
for col in dat:
    if col.find('l_')==-1: continue
    means.append(np.mean(dat[col]))
    minHPDc,maxHPDc=calcHPD(dat[col])
    minHPD.append(minHPDc); maxHPD.append(maxHPDc)
    print(col,np.mean(dat[col]),minHPDc,maxHPDc)

print(minHPD,maxHPD)
mean_str=print_R_vec("la",means)
minHPD_str=print_R_vec('la_min_HPD',minHPD)
maxHPD_str=print_R_vec('la_max_HPD',maxHPD)
print(mean_str)
print(minHPD_str)
print(maxHPD_str)

means=[]
minHPD=[]; maxHPD=[]
for col in dat:
    if col.find('m_')==-1: continue
    means.append(np.mean(dat[col]))
    minHPDc,maxHPDc=calcHPD(dat[col])
    print(col,np.mean(dat[col]),minHPDc,maxHPDc)
    minHPD.append(minHPDc); maxHPD.append(maxHPDc)

#print(minHPD,maxHPD)
mean_str=print_R_vec("mu",means)
minHPD_str=print_R_vec('mu_min_HPD',minHPD)
maxHPD_str=print_R_vec('mu_max_HPD',maxHPD)
print(mean_str)
print(minHPD_str)
print(maxHPD_str)

