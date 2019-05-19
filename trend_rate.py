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
#ADD ANY EXTRA ARGUMENTS YOU NEED AFTER

#parse arguments for globals
p.add_argument('trend_data', metavar='<path to trend file>', type=str,help='Input trend file should be columns tab-separated with headers. No missing values.',default="")
p.add_argument('trend_index', type=int,help='Column of trend in trend file.',default=0,metavar=0)
p.add_argument('-const_B', type=bool, help='F) Vary rates with trend T) Constant rates', default=False,metavar=False)
p.add_argument('-const_D', type=bool, help='F) Vary rates with trend T) Constant rates', default=False,metavar=False)

args = p.parse_args()

CONST_BIRTH=args.const_B
CONST_DEATH=args.const_D

set_seed(args.seed)

TS,TE,PRESENT,ORIGIN=parse_ts_te(args.d,args.TBP,args.first_year,args.last_year,args.death_jitter)

ORIGIN, PRESENT, N_SPEC, N_EXTI, DT, N_TIME_BINS, TIME_RANGE=create_bins(ORIGIN, PRESENT,TS,TE,args.rm_first_bin)

B_EMP,D_EMP=print_empirical_rates(N_SPEC,N_EXTI,DT)

def parse_trend_data(trend_file_path, index):
	trend_matrix=t_file=np.loadtxt(trend_file_path, skiprows=1)
	trend=trend_matrix[:,index]
	max_t=np.max(trend)
	min_t=np.min(trend)
	trend=(trend-max_t)/(max_t-min_t)
	return trend

TREND=parse_trend_data(args.trend_file_path,args.index)

def likelihood_function(args):
	[l_min, m_min, alpha, beta,delta,gamma]= args
	if CONST_BIRTH==True:
		birth_rates = np.ones(n_time_bins)*l_min
#		birth_lik = np.sum(n_spec*log(birth_rates)-birth_rates*np.sum(Dt)) # log probability of speciation
	else:
		birth_rates = l_min + (l_min * TREND) ** delta

	birth_lik = np.sum(log(birth_rates)*N_SPEC - birth_rates*DT)

	if CONST_DEATH==True:	
		death_rates = np.ones(n_time_bins) *m_min
    else:
		death_rates =  m_min +  (m_min * TREND) ** gamma
	death_lik = np.sum(log(death_rates)*N_EXTI - death_rates*DT)

	lik = np.array([birth_lik, death_lik])
	return [lik, birth_rates, death_rates]

def calc_priors(args):
	p = prior_gamma(args[0],a=1,s=10,l.=001) #l_min
	p += prior_gamma(args[1],a=1,s=10,l=.001) #m_min
    	#p += prior_gamma(args[2],a=1,s=10,l=0) #alpha
    	#p += prior_gamma(args[3],a=1,s=10,l=0) #beta
	p += prior_norm(args[4],loc=0,scale=1) #delta
	p += prior_norm(args[5],loc=0,scale=1) #gamma
	

def __main__(parsed_args):
	out=""
	if CONST_BIRTH:out+="_CONB"
	else:out+="_EXPB"
	if CONST_DEATH:out+="_COND"
	else: out+="_EXPD"
	
	outfile = "%s_%s%s.log" % (os.path.splitext(args.d)[0], seed, out)
	logfile = open(outfile , "w") 
	wlog=csv.writer(logfile, delimiter='\t')
	head =["it","posterior","likelihood","likelihood_birth","likelihood_death","prior","l_min","m_min","alpha","beta","delta","gamma"]
	for i in range(len(Dt)): head.append("l_%s" % i)
	for i in range(len(Dt)): head.append("m_%s" % i)
	head+=["corr_coeff","rsquared","gelman_r2"]

	wlog.writerow(head)


	#SETUP PROPOSALS
							
	update_multiplier = np.array([1,  1, 1, 1]) #l_min, m_min, delta, gamma								
	if CONST_BIRTH==False:	
		update_multiplier[2]=1
	if CONST_DEATH==False:	
		update_sliding_window[1]=1
	
	update_multiplier = update_multiplier/sum(update_multiplier)
	#update_sliding_window = update_sliding_window/sum(update_sliding_window)

	###INITIAL VALUES###
	l_min=.001
	m_min=.001
	alpha=.001
	beta=.001
	delta=0
	gamma=0
	argsA=np.array([l_min, m_min, alpha, beta,delta,gamma])
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
		rr = np.random.random(2)
		[args, hastings]=update_multiplier_proposal_vec(args,d=1.1,f=update_multiplier) #update everything with multipliers	
		lik_res = likelihood_function(args)
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

		if iteration % sampling_freq==0:
			#print lik,prior, args
			argsO=deepcopy(argsA) #when you copy lists, makes sure you dont change things by reference
			print(iteration, likA, argsO) #, args
			adequacy=calculate_r_squared(B_EMP,D_EMP,birth_rates,death_rates)
			l= [iteration,likA+priorA, likA,likBirthA,likDeathA, priorA] + list(argsO) + list(birth_rates) + list(death_rates) + list(adequacy)
			wlog.writerow(l)
			logfile.flush()
			os.fsync(logfile)
			
		iteration += 1

__main__(args)
   
