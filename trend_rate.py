#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 12:38:36 2019

@author: bernie
"""

import argparse,sys
import os, csv, glob
from numpy import *
import numpy as np
from scipy.special import gamma
from scipy.special import beta as f_beta
import random as rand
import csv
import scipy.stats
from literate_library.py import *
np.set_printoptions(suppress=True)
np.set_printoptions(precision=3) 



 
print("\n\n             TrendRate - 20190205\n")

####SET UP####

p = core_arguments()
p.add_argument('-corr_B', type=int, help="set to: 1 for linear, 2 for exp" , default=1, metavar=1)
p.add_argument('-corr_D', type=int, help="set to: 1 for linear, 2 for exp" , default=1, metavar=1)
#ADD ANY EXTRA ARGUMENTS YOU NEED AFTER

#parse arguments for globals
args = p.parse_args()
CONST_BIRTH=args.const_B
CONST_DEATH=args.const_D
CORR_B=args.corr_B
CORR_D=args.corr_D
if CONST_BIRTH==TRUE: CORR_B=0
if CONST_DEATH==TRUE: CORR_D=0

set_seed(args.seed)

TS,TE,PRESENT,ORIGIN=parse_ts_te(args.d,args.first_year,args.last_year,args.death_jitter)

N_SPEC, N_EXTI, DT, N_TIME_BINS, TIME_RANGE=create_bins(ORIGIN, PRESENT,TS,TE,args.rm_first_bin)

def likelihood_function(args):
	[l_min,m_min,l_corr,m_corr] = args
	if CONST_BIRTH==True:
		birth_rates = np.ones(n_time_bins)*l_min
#		birth_lik = np.sum(n_spec*log(birth_rates)-birth_rates*np.sum(Dt)) # log probability of speciation
	elif CORR_B==1:
		# lik speciation
		birth_rates = l_min + l_min * l_corr * TREND
	elif CORR_B==2:
		birth_rates = l_min + exp(l_corr * TREND)
	birth_lik = np.sum(log(birth_rates)*N_SPEC - birth_rates*DT)

	if CONST_DEATH==True:	
		death_rates = np.ones(n_time_bins) *m_min
	elif CORR_B ==1:
		death_rates =  m_min + m_min * m_corr * TREND
	elif CORR_B ==2:
		death_rates = m_min + exp(m_corr * TREND)
	death_lik = np.sum(log(death_rates)*N_EXTI - death_rates*DT)

	lik = np.array([birth_lik, death_lik])
	return [lik, birth_rates, death_rates]

def calc_priors(args):
	p = prior_gamma(args[0],a=1,s=10,l=0) #l_min
	p += prior_gamma(args[1],a=1,s=10,l=0) #m_min
	p += prior_norm(args[2],loc=0,scale=1) #l corr
	p += prior_norm(args[2],loc=0,scale=1) #m corr
	

def __main__(parsed_args):
	out=""
	if CONST_BIRTH:out+="_CONB"
	if CONST_DEATH:out+="_COND"
	if CORR_B==1:out+="_LINB"
	else:out+="_EXPB"
	if CORR_D==1:out+="_LIND"
	else: out+="_EXPD"
	
	outfile = "%s_%s%s.log" % (os.path.splitext(args.d)[0], seed, out)
	logfile = open(outfile , "w") 
	wlog=csv.writer(logfile, delimiter='\t')
	head =["it","posterior","likelihood","likelihood_birth","likelihood_death","prior","l_min","m_min","l_corr","m_corr"]
	for i in range(len(Dt)): head.append("l_%s" % i)
	for i in range(len(Dt)): head.append("m_%s" % i)
	wlog.writerow(head)


	#SETUP PROPOSALS
							
	update_multiplier = np.array([1,  1]) #l_min, m_min								
	update_sliding_window = np.array([0,  0]) #corr_l, corr_m
	if CONST_BIRTH==False:		update_sliding_window[0]=1
	if CONST_DEATH==False:		update_sliding_window[1]=1
	
	update_multiplier = update_multiplier/sum(update_multiplier)
	update_sliding_window = update_sliding_window/sum(update_sliding_window)

	###INITIAL VALUES###
	l_min=.001
	m_min=.001
	corr_l=0
	corr_m=0
	argsA=np.array([l_min, m_min, l_corr, m_corr])
	lik_res = likelihood_function(argsA)
	likA = np.sum(lik_res[0])
	likBirthA = lik_res[0][0]
	likDeathA = lik_res[0][1]
	birth_rates = lik_res[1]
	death_rates = lik_res[2]
	lik=likA
	priorA=calc_prior(argsA)
	prior=priorA
	
	###MCMC###
	iteration = 0
	while iteration != parsed_args.n_iterations:
		args = argsA+0.
		updated_ext = 0
		hastings= 0
		rr = np.random.random()
		if rr[1]<0.5 and CONST_K==0:
			res = update_slidwin_proposal_vec(res, m=-1, M=1, d=1.,f=update_sliding_window) #update midpoint (the only sliding window proposal)
		else:
			res = update_multiplier_proposal_vec(args,d=1.1,f=update_multiplier) #update everything with multipliers	
		[args, hastings] = res
		lik_res = likelihood_function(args,)
		lik = np.sum(lik_res[0])
		prior = calc_prior(args)
		if (lik - likA) + (prior - priorA) + hastings > log(np.random.random()) or iteration==0:
			argsA = args
			priorA = prior
			likA = lik
			likBirthA = lik_res[0][0]
			likDeathA = lik_res[0][1]
			birth_rates = lik_res[1]
			death_rates = lik_res[2]
			niche = lik_res[3]
			nicheFrac = lik_res[4]
		if iteration % sampling_freq==0:
			#print lik,prior, args
			argsO=deepcopy(argsA) #when you copy lists, makes sure you dont change things by reference
			print(iteration, likA, argsO) #, args
			l= [iteration,likA+priorA, likA,likBirthA,likDeathA, priorA] + list(argsO) + list(birth_rates) + list(death_rates)
			wlog.writerow(l)
			logfile.flush()
			os.fsync(logfile)
			
		iteration += 1

__main__(args)
   
