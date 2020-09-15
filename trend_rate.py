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
from literate_library import *
from copy import deepcopy
np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)



print("\n\n             TrendRate - 20190205\n")

####SET UP####

p = core_arguments()
#ADD ANY EXTRA ARGUMENTS YOU NEED AFTER

#parse arguments for globals
p.add_argument('-trend_data', metavar='<path to trend file>', type=str,help='Input trend file should be columns tab-separated with headers. No missing values.',default="")
p.add_argument('-trend_index', type=int,help='Column of trend in trend file.',default=0,metavar=0)
p.add_argument('-const_B', type=bool, help='F) Vary rates with trend T) Constant rates', default=False,metavar=False)
p.add_argument('-const_D', type=bool, help='F) Vary rates with trend T) Constant rates', default=False,metavar=False)

args = p.parse_args()

CONST_BIRTH=args.const_B
CONST_DEATH=args.const_D

seed = set_seed(args.seed)

TS,TE,PRESENT,ORIGIN=parse_ts_te(args.d,args.TBP,args.first_year,args.last_year,args.death_jitter)

ORIGIN, PRESENT, N_SPEC, N_EXTI, DT, N_TIME_BINS, TIME_RANGE=create_bins(ORIGIN, PRESENT,TS,TE,args.rm_first_bin)

B_EMP,D_EMP=print_empirical_rates(N_SPEC,N_EXTI,DT)

SMALL_NUMBER = 0.000000000000001 #used for flooring rates


def parse_trend_data(trend_file_path, index, rm_first_bin):
    trend_matrix=t_file=np.genfromtxt(trend_file_path, skip_header=1)
    trend=trend_matrix[:,index]
    trend=trend[:-1] #always drop last bin
    print("TREND",trend)
    if rm_first_bin:
        trend=trend[1:]
    max_t=np.max(trend)
    min_t=np.min(trend); print("TREND",trend)
    trend=(trend-min_t)/(max_t-min_t)
    trend[trend==0]=SMALL_NUMBER
    return trend

TREND=parse_trend_data(args.trend_data,args.trend_index,args.rm_first_bin)
print("TREND",TREND)
def likelihood_function(args):
    [l_min, m_min, alpha, beta,delta,gamma]= args
    if CONST_BIRTH==True:
        birth_rates = np.ones(N_TIME_BINS)*l_min
    else:
        #this is a workaround for taking power of negative number
        birth_rates = l_min + alpha * TREND ** delta
        birth_rates[birth_rates<=0.0]=SMALL_NUMBER
    
    birth_lik = np.sum(log(birth_rates)*N_SPEC - birth_rates*DT)
    if CONST_DEATH==True:
        death_rates = np.ones(N_TIME_BINS) *m_min
    else:
        death_rates =  m_min + beta* TREND ** gamma
        death_rates[death_rates<=0.0]=SMALL_NUMBER
    death_lik = np.sum(log(death_rates)*N_EXTI - death_rates*DT)

    lik = np.array([birth_lik, death_lik])
    return [lik, birth_rates, death_rates]

def calc_prior(args):
    p = prior_gamma(args[0],a=1,s=10,l=.001) #l_min
    p += prior_gamma(args[1],a=1,s=10,l=.001) #m_min
    p += prior_norm(args[2],loc=0,scale=5) #alpha
    p += prior_norm(args[3],loc=0,scale=5) #beta
    p += prior_gamma(args[4],loc=3,scale=.5) #delta
    p += prior_gamma(args[5],loc=3,scale=.5) #gamma
    return p

def __main__(parsed_args):
    out=""
    if CONST_BIRTH:out+="_CONB"
    else:out+="_EXPB"
    if CONST_DEATH:out+="_COND"
    else: out+="_EXPD"

    outfile = "%s_%s%s_%s.trendrate.log" % (os.path.splitext(parsed_args.d)[0], seed, out,parsed_args.trend_index)
    logfile = open(outfile , "w")
    wlog=csv.writer(logfile, delimiter='\t')
    head =["it","posterior","likelihood","likelihood_birth","likelihood_death","prior","l_min","m_min","alpha","beta","delta","gamma"]
    for i in range(len(DT)): head.append("l_%s" % i)
    for i in range(len(DT)): head.append("m_%s" % i)
    head+=["corr_coeff","rsquared","gelman_r2"]

    wlog.writerow(head)


    #SETUP PROPOSALS
    update_multiplier = np.array([1,  1,1,1, 1, 1]) #l_min, m_min, alpha, beta, delta, gamma
    if CONST_BIRTH==True:
        update_multiplier[2]=0
    if CONST_DEATH==True:
        update_multiplier[3]=0

    update_multiplier = update_multiplier/sum(update_multiplier)

    ###INITIAL VALUES###
    l_min=.1
    m_min=.1
    alpha=0
    beta=0
    delta=1
    gamma=1
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
    #print("PRIOR",prior)

    ###MCMC###
    iteration = 0
    while iteration != parsed_args.n:
        args = argsA+0.
        updated_ext = 0
        hastings= 0
        res = argsA+0
        res = update_multiplier_proposal_vec(args,d=1.1,f=update_multiplier) #update everything with multipliers
        [args, hastings] = res
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
        if iteration % parsed_args.s==0:
            #print lik,prior, args
            argsO=deepcopy(argsA) #when you copy lists, makes sure you dont change things by reference
            print(iteration, likA, argsO) #, args
            adequacy=calculate_r_squared(B_EMP,D_EMP,birth_rates,death_rates)
            print(birth_rates)
            print(death_rates)
            l= [iteration,likA+priorA, likA,likBirthA,likDeathA, priorA] + list(argsO) + list(birth_rates) + list(death_rates) + list(adequacy)
            wlog.writerow(l)
            logfile.flush()
            os.fsync(logfile)

        iteration += 1

__main__(args)
