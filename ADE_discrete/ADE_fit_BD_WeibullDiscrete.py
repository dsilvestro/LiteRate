#!/usr/bin/env python 
import argparse, os,sys
from numpy import *
import numpy as np
from scipy.special import gamma
from scipy.special import beta as f_beta
import platform, time
import csv
from scipy.special import gdtr, gdtrix
from scipy.special import betainc
import scipy.stats
np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)  
run_discrete = 1 # if set to 1 times of origination and extinctions are rounded to the closest integer

origin = 1960
present = 2010

def simulate_data(shape=1.,scale=5., n=1000):
	s = np.sort(np.random.uniform(origin,present-1,n))
	e = s + np.random.weibull(shape,n) *scale
	e[e>present] = present
	print "\nextant:", len(e[e==present]), shape, scale, "mean longevity:", mean(e[e<present]-s[e<present])
	if run_discrete:
		s = s.astype(int)
		e = e.astype(int)
	return(s,e)

# Aged dependet rate leading to a Weibull waiting time
def wr(t,W_shape,W_scale):
	# rate
	wr=(W_shape/W_scale)*(t/W_scale)**(W_shape-1)
	return wr


# Log of aged dependent rate leading to a Weibull waiting time
def log_wr(t,W_shape,W_scale):
	log_wr=log(W_shape/W_scale)+(W_shape-1)*log(t/W_scale)
	return log_wr

def cdf_WR(W_shape,W_scale,x):
	return (x/W_scale)**(W_shape)

def BDwwte(args):
	[l,W_shape,W_scale] = args
	d = e - s
	de = d[e<present] #takes only the extinct species times
	birth_lik = len(s)*log(l)-l*sum(d) # log probability of speciation
	death_lik_de = sum(log_wr(de, W_shape, W_scale)) # log probability of death event
	death_lik_wte = sum(-cdf_WR(W_shape,W_scale, d)) # log probability of waiting time until death event
	lik = birth_lik + death_lik_de + death_lik_wte
	return sum(lik)


def BDwwteDISCRETE(args):
	[l,W_shape,W_scale] = args
	d = e - s
	de = d[e<present] #takes only the extinct species times
	birth_lik = len(s)*log(l)-l*sum(d) # log probability of speciation
	
	# lik extinct	
	death_lik_de = wr(death_lik_i[e<present],W_shape,W_scale) 
	death_lik_wte = exp(-cdf_WR(W_shape,W_scale, death_lik_i[e<present]))
	lik_ext = death_lik_de * death_lik_wte
	death_lik = log(np.sum(lik_ext, axis=1)) - log(n_discrete_bins)
	
	# lik extant	
	death_lik = sum(death_lik) + np.sum(-cdf_WR(W_shape,W_scale, d[e==present]))
	
	lik = birth_lik + death_lik
	return sum(lik)

def update_multiplier_proposal(i,d=1.1,f=0.5):
	S=shape(i)
	u = np.random.uniform(0,1,S)
	l = 2*log(d)
	m = exp(l*(u-.5))
 	ii = i * m
	U=sum(log(m))
	return ii, U

logfile = open("Weib.log" , "wb") 
wlog=csv.writer(logfile, delimiter='\t')

wlog.writerow([ "it","true_shape","true_scale","true_longevity","est_shape","est_scale","este_longevity"  ])

for sim in range(1000):
	true_shape = exp(np.random.uniform(-2,2))
	true_longevity = np.random.uniform(3,10)
	true_scale = true_longevity/(gamma(1+1/true_shape)) 		
	# true_scale = np.random.uniform(3,10)		
	# true_longevity = true_scale*(gamma(1+1/true_shape)) 
		
	s, e = simulate_data(shape=true_shape,scale=true_longevity, n=1000)
	# 2D matrix to integrate out 1-year time bins
	d = e - s
	n_discrete_bins = 50
	discr = np.linspace(0.0001,0.999,n_discrete_bins)
	
	#K=n_discrete_bins-1.        
	#k=array(range(int(K+1)))
	#beta=k/K
	#alpha=0.3           # categories are beta distributed
	#discr=list(beta**(1./alpha))
	#discr[0]+= 0.0001 # avoid exactly 0 temp	
	#discr = np.array(discr)
	
	
	death_lik_i = np.zeros((len(d),n_discrete_bins))
	for i in range(len(discr)):
		death_lik_i[:,i] = d + discr[i]

	# speciation rate, Weibull shape and rate (extinction rate)
	parA =[ 0.2,1.,4.]
	if run_discrete: likA = BDwwteDISCRETE(parA)
	else: likA = BDwwte(parA)


	print likA
	for i in range(1000):
		#if i % 100 ==0: print i , likA, parA
		par = update_multiplier_proposal(parA)[0]
		if run_discrete: lik = BDwwteDISCRETE(par)
		else: lik = BDwwte(par)
		if lik > likA:
			parA = par
			likA = lik
			
	longevity = parA[2]* gamma(1+1/parA[1])
	print "Max liklihood:",likA
	print "ML parameters (birth, shape, scale)", parA
	print "mean longevity:", longevity
	l= [sim,true_shape,true_scale,true_longevity ] + list(parA[1:]) + [longevity] 
	#print l
	wlog.writerow(l)
	logfile.flush()
	os.fsync(logfile)
	