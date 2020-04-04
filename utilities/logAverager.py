import os
import sys
import pandas as pd
import argparse
import numpy as np
from literate_library import print_R_vec, calcHPD

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

