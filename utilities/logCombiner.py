# -*- coding: utf-8 -*-
"""
Created on Sat May 25 14:16:23 2019

@author: bernard
"""

import argparse
import os
import sys

p=argparse.ArgumentParser() #description='<input file>') 
p.add_argument('dir', type=str, default='.')
p.add_argument('-o', nargs='?', type=argparse.FileType('w'), default=sys.stdout)
p.add_argument('-b',type=float,help="Pct Burnin",default=.1,metavar=.1)
p.add_argument('-head',type=int,help="Header?",default=1,metavar=1)
p.add_argument('-e', type=str,help='file extension', default='.log',metavar=.1)

args = p.parse_args()

file_list=os.listdir(args.dir)
total_log=[]

for file_name in file_list:
	if file_name.find(args.e) ==-1 :continue
	with open(args.dir+'/'+file_name) as f:
		file_log=f.readlines()
		if args.head==1: header=file_log[0]
		burnin=int(args.b*len(file_log[args.head:]))
		total_log+=file_log[burnin+args.head:]
if args.head==1:
	args.o.write(header)
	it_bool= (header.split('\t')[0]=='it')
else: it_bool=False
for i,l in enumerate(total_log):
	if it_bool:
		l=l.split('\t')
		l[0]=str(i)
		args.o.write('\t'.join(l))
	else:args.o.write(l)
