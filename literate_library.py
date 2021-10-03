#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 15:41:07 2019

@author: bernie
"""
import argparse, os,sys
from numpy import *
import numpy as np
from scipy.special import gamma
from scipy.special import beta as f_beta
import csv
from scipy.special import gdtr, gdtrix
from scipy.special import betainc
import scipy.stats
import scipy.misc
import random
from scipy import stats
from warnings import warn
import pandas as pd
###LITERATE LIBRARY###

###LOG PARSING STUFF###
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

###MISC####
def approx_log_fact(n):
	# http://mathworld.wolfram.com/StirlingsApproximation.html
	return np.log(np.sqrt((2*n+1./3)*np.pi)) + n*np.log(n) -n

def get_log_factorial(n):
	if n < 100: return np.log(scipy.misc.factorial(n))
	else: return approx_log_fact(n)


def random_choice(vector):
	ind = np.random.choice(range(len(vector)))
	return [vector[ind], ind]

# VECTORIZED LIK FUNCTIONS
def get_br(ts,te,t0,t1):
	s, e  = ts+0., te+0.
	s[s<t0] = t0
	e[e>t1] = t1
	dt = e - s
	return np.sum(dt[dt>0])

def precompute_events(ts,te,t0,t1):
	n_spec_events = len(np.intersect1d((ts >= t0).nonzero()[0], (ts < t1).nonzero()[0]))
	n_exti_events = len(np.intersect1d((te > t0).nonzero()[0], (te <= t1).nonzero()[0]))
	tot_br_length = get_br(ts,te,t0,t1)  
	return n_spec_events, n_exti_events, tot_br_length
	
def get_rate_index(times):
	if len(times)==2: 
		ind =np.zeros(n_bins).astype(int)
	else:
		times = np.round(times+0)
		dT = abs(np.diff(times)).astype(int)
		#print dT, sum(dT), times
		ind = []
		[ ind.extend([i]*dT[i]) for i in range(len(times)-1) ]
		ind = np.array(ind)
	return ind

def BD_lik_Keiding(L_acc_vec,M_acc_vec):
	# BD likelihood
	try:
		Blik = sum(log(L_acc_vec)*sp_events_bin - L_acc_vec*br_length_bin) 
		Dlik = sum(log(M_acc_vec)*ex_events_bin - M_acc_vec*br_length_bin) 
	except:
		print(len(L_acc_vec),len(M_acc_vec),len(sp_events_bin))
		sys.exit()
	return sum(Blik)+sum(Dlik)

def BDI_partial_lik(L_acc_vec,M_acc_vec):
	L = L_acc_vec * (1-model_BDI) # if model_BDI=0: BD, if model_BDI=1: ID
	M = M_acc_vec                 
	I = L_acc_vec * model_BDI     
	k = br_length_bin # diversity trajectory	
	Uk = sp_events_bin   # number up steps
	Dk = ex_events_bin   # number down steps
	#lik_BI = sum(log(L[k>0]*k[k>0]+I[k>0])*Uk[k>0] - (L[k>0]*k[k>0]+I[k>0])*Tk[k>0])
	#lik_D = sum(log(M*k)*Dk -(M*k*Tk))

	lik = Uk[k>0]*log(k[k>0]*L[k>0]+I[k>0]) + Dk[k>0]*log(M[k>0]*k[k>0]) - Tk[k>0]*(k[k>0]*(L[k>0]+M[k>0])+I[k>0])
	#quit()
	return sum(lik) # lik_BI + lik_D

####### PROPOSALS #######
def update_sliding_win(i, m=0, M=1, d=0.05): 
	ii = i+(np.random.random()-.5)*d
	if ii>M: ii=M-(ii-M)
	if m==0: ii = abs(ii)
	return ii

def update_sliding_win_log(i, m=1, M=np.e, d=0.05): 
	ii = i+(np.random.random()-.5)*d
	if ii>M: ii=M-(ii-M)
	elif ii<m: ii = m+(m-ii)
	return ii

def update_normal_nobound(i, d=0.05): 
        ii = i+np.random.normal(0,d)
        return ii

def update_normal_nobound_vec(i, d=0.05,f=.75): 
	S=np.shape(i)
	ff=np.random.binomial(1,f,S)
	m=np.random.normal(0,d,S)
	m[ff==0]=0.
	ii = i+m
	return ii, 0

def update_poisson_proposal(kt):
	ktp = np.random.poisson(kt)
	if ktp ==0: return kt,0
	backward_pr = logPoisson_pmf(kt,ktp)
	forward_pr = logPoisson_pmf(ktp,kt)
	hastings = backward_pr-forward_pr
	return ktp,hastings

def update_multiplier_proposal_vec(q,d=1.1,f=0.75):
	S=np.shape(q)
	ff=np.random.binomial(1,f,S)
	u = np.random.uniform(0,1,S)
	l = 2*log(d)
	m = exp(l*(u-.5))
	m[ff==0] = 1.
	new_q = q * m
	U=sum(log(m))
	return new_q,U

def update_multiplier_proposal(q,d=1.1):
	u = np.random.random()
	l = 2*log(d)
	m = exp(l*(u-.5))
	new_q = q * m
	U=log(m)
	return new_q,U


####### PRIORS #######

def logPoisson_pmf(x,l):
	log_pmf = (x*np.log(l) -l) - get_log_factorial(x)
	return log_pmf

def prior_gamma(x,a,s,l):
	# mean = a*s
	return scipy.stats.gamma.logpdf(x, a, scale=s,loc=l)

def prior_norm(x,l=0,s=1):
	return scipy.stats.norm.logpdf(x, loc=l, scale=s)

def prior_sym_beta(x,a): 
	return scipy.stats.beta.logpdf(x, a,a)

def prior_beta(x,a,b): 
	return scipy.stats.beta.logpdf(x, a,b)

####SET UP STUFF####			
def parse_ts_te(input_file,TBP,first_year,last_year,death_jitter):
	t_file=pd.read_csv(input_file, delimiter='\t').to_numpy()
	if t_file.shape[1]==4:
		warn('Four column (with clade) LiteRate input is deprecated. Use three columns.', FutureWarning)
		ts_years = t_file[:,2]
		te_years = t_file[:,3]
	else:
		ts_years = t_file[:,1]
		te_years = t_file[:,2]
	if TBP==True:
		if first_year!=-1:
			te_years=te_years[ts_years<=first_year]
			ts_years=ts_years[ts_years<=first_year]
		if last_year != -1:
			ts_years=ts_years[ts_years>=last_year]
			te_years=te_years[ts_years>=last_year]
			te_years[te_years<last_year]=last_year
		ts= max(ts_years)-ts_years
		te= max(ts_years)- te_years
	else:
		if first_year!=-1:
			te_years=te_years[ts_years>=first_year]
			ts_years=ts_years[ts_years>=first_year]
		if last_year!=-1:
			te_years=te_years[ts_years<=last_year]
			ts_years=ts_years[ts_years<=last_year]
			te_years[te_years>last_year]=last_year
		ts = ts_years
		te = te_years
	
	te = te + death_jitter
	present = max(te)
	origin  = min(ts)
	return ts, te, present, origin

def create_bins(origin, present,ts,te,rm_first_bin):
	n_spec = []
	n_exti = []
	Dt = []
	bins = np.arange(origin,present+1)
	for i in range(len(bins)-1):
		a,b,c = precompute_events(ts,te,bins[i],bins[i+1])
		n_spec.append(a)
		n_exti.append(b)
		Dt.append(c)
	
	#always drop last bin
	n_spec = np.array(n_spec)[:-1]
	n_exti = np.array(n_exti)[:-1]
	Dt = np.array(Dt)[:-1]
	
	if rm_first_bin:
		# remove first bin
		n_spec = n_spec[1:]
		n_exti = n_exti[1:]
		Dt = Dt[1:]
		origin +=1 
		
	n_time_bins = len(Dt)
	time_range = np.arange(n_time_bins).astype(float)

	return origin,present,n_spec, n_exti, Dt, n_time_bins, time_range

#this relies on globals so if you are using different names, it'll fail
def print_empirical_rates(n_spec,n_exti,Dt):
	print("EMPIRICAL BIRTH RATES:")
	print(n_spec/Dt)
	print("EMPIRICAL DEATH RATES:")
	print(n_exti/Dt)
	return(n_spec/Dt,n_exti/Dt)	
	sys.exit()

def calculate_r_squared(emp_birth,emp_death,est_birth,est_death):
	joint_emp=np.concatenate([emp_birth,emp_death])
	joint_est=np.concatenate([est_birth,est_death])
	res=np.linalg.lstsq(np.vstack(joint_emp),joint_est,rcond=None)
	coeff=res[0][0]
	ssres=res[1][0]
	r2=1-ssres/np.sum(joint_est**2)
	fitted=coeff*joint_emp
	resid=joint_est-fitted
	var_fitted=np.var(fitted,ddof=1)
	gelman_r2=var_fitted/(var_fitted+np.var(resid,ddof=1))
	return coeff,r2,gelman_r2
	
	
def set_seed(seed):
	if seed==-1:
		rseed=np.random.randint(0,9999)
	else: rseed=seed	
	random.seed(rseed)
	np.random.seed(rseed)
	return rseed
	
def core_arguments():
	p=argparse.ArgumentParser() #description='<input file>') 

	p.add_argument('-v',       action='version', version='%(prog)s')
	p.add_argument('-d',       type=str, help='data file', default="", metavar="") 
	p.add_argument('-n',       type=int, help='n. MCMC iterations', default=10000000, metavar=10000000)
	p.add_argument('-p',       type=int, help='print frequency', default=1000, metavar=1000) 
	p.add_argument('-s',       type=int, help='sampling frequency', default=1000, metavar=1000) 
	p.add_argument('-seed',    type=int, help='seed (set to -1 to make it random)', default= -1, metavar= -1)
	#p.add_argument('-present_year',    type=int, help="""set to: -1 for standard pyrate datasets (time BP), \
	#0: time AD and present set to most recent TE, 1: time AD present user defined """, default= 0, metavar= 0)
	p.add_argument('-TBP', help='Default is AD. Include for TBP.', default=False, action='store_true')
	p.add_argument('-first_year',    type=int, help='This is a convenience function if you would like to specify a different start to your dataset. Unspecified for TBP.', default= -1, metavar= -1)
	p.add_argument('-last_year',    type=int, help='This is a convenience function if you would like to specify a different end to your dataset. Unspecified for TBP.', default= -1, metavar= -1)
	p.add_argument('-death_jitter', type=float, help="""Determines the amount to jitter death times.\
	               If set to 0, lineages that lived and died in same time bin will be excluded from branch length.""", default= .5, metavar= .5)
	p.add_argument('-rm_first_bin',   type=float, help='if set to 1 it removes the first time bin (if max time is not the origin)', default= 0, metavar= 0)
	p.add_argument('-print_emp',  help='Prints empirical rates', default=False, action='store_true')
	return p
#ADD ANY ARGUMENTS YOU NEED AFTER
