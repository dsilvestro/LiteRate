import os
import pandas as pd
burnin_pct=20
thin_lines=10
files=os.listdir('.')
first=files[0]
print(first)
csv=pd.read_csv('./'+first,sep='\t')
allcsv=csv.iloc[1001::10,]
for f in os.listdir('.')[1:]:
    if f[0]=='.': continue
    if f.find('.log')==-1: continue
    print(f)
    csv=pd.read_csv('./'+f,sep='\t')
    csv=csv.iloc[1001::10,]
    allcsv=allcsv.append(csv)
print(allcsv.shape)
print(allcsv.columns)
#allcsv['it']=list(range(0,allcsv.shape[0]+allcsv['it'].iloc[1],allcsv['it'].iloc[1]))
allcsv.insert(0,'newit',range(allcsv.shape[0]))
allcsv.to_csv('./ALL.log',sep='\t',index=False)
