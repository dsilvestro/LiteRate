import glob
import pandas as pd
import sys
import numpy as np
np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)  

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

files=glob.glob(sys.argv[1]+'/*div.log')
empirical_birth=[]
empirical_death=[]
net_diversity=[]

for f in files:
    csv=pd.read_csv(f,sep='\t')
    net_diversity.append(np.array(csv.br_length))
    empirical_birth.append(np.array(csv.sp_events)/np.array(csv.br_length))
    empirical_death.append(np.array(csv.ex_events)/np.array(csv.br_length))

empirical_death=np.array(empirical_death)
ed_mean=empirical_death.mean(axis=0)
ed_CI_min=empirical_death.min(axis=0)
ed_CI_max=empirical_death.max(axis=0)
print("EMPIRICAL DEATH")
print(print_R_vec('ed_mean',ed_mean))
print(print_R_vec('ed_min',ed_CI_min))
print(print_R_vec('ed_max',ed_CI_max))

empirical_birth=np.array(empirical_birth)
eb_mean=empirical_birth.mean(axis=0)
eb_CI_min=empirical_birth.min(axis=0)
eb_CI_max=empirical_birth.max(axis=0)
print("EMPIRICAL BIRTH")
print(print_R_vec('eb_mean',eb_mean))
print(print_R_vec('eb_min',eb_CI_min))
print(print_R_vec('eb_max',eb_CI_max))

net_diversity=np.array(net_diversity)
nd_mean=net_diversity.mean(axis=0)
nd_CI_min=net_diversity.min(axis=0)
nd_CI_max=net_diversity.max(axis=0)
print("NET DIVERSITY")
print(nd_mean)
print(nd_CI_min)
print(nd_CI_max)

