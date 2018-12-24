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


def simulate_data(origin, present,shape=1.,scale=5., n=1000,run_discrete=1):
	s_weibull = np.sort(np.random.uniform(origin,present-1,n))
	e_weibull = s_weibull + np.random.weibull(shape,n) *scale	
	
	e_weibull[e_weibull>present] = present
	if run_discrete:
		s_weibull = s_weibull.astype(int)
		e_weibull = e_weibull.astype(int)
	return(s_weibull,e_weibull)

def update_sliding_win(i, m=0, M=1, d=0.05): 
	ii = i+(np.random.random()-.5)*d
	if ii>M: ii=M-(ii-M)
	if m==0: ii = abs(ii)
	return ii



def update_multiplier_proposal(q,d=1.1,f=0.75):
	S=np.shape(q)
	ff=np.random.binomial(1,f,S)
	u = np.random.uniform(0,1,S)
	l = 2*log(d)
	m = exp(l*(u-.5))
	m[ff==0] = 1.
 	new_q = q * m
	U=sum(log(m))
	return new_q,U

# Aged dependet rate leading to a Weibull waiting time
def wr(t,W_shape,W_scale):
	# rate
	wr=(W_shape/W_scale)*((t/W_scale)**(W_shape-1))
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


def birth_rates_infection(Dt,l0=0.5,gam=0.2,thres=0.5,k0=5,mt=0,maxSP=0):
	D = Dt/max_sp
	Kvec = np.zeros(len(Dt))
	Kvec[0] = k0
	for i in range(1,len(Dt)): Kvec[i] = Kvec[i-1] + (D[i-1]*gam + thres) *(1-Kvec[i-1]) 	
	lt = l0 - (l0-mt)*(D/Kvec) + 0.000001 # avoid exactly 0 rates
	return lt, Kvec


def BDwwteDISCRETE(args):
	[l0,gam,thres,k0,W_shape,W_scale,max_sp_add] = args
	d = te - ts
	de = d[te<present] #takes only the extinct species times
	birth_lik1 = len(ts)*log(l0)-l0*sum(d) # log probability of speciation

	# lik speciation
	longevity = W_scale* gamma(1+1/W_shape)
	death_rates = 1./longevity
	birth_rates, Kvec = birth_rates_infection(Dt, l0,gam,thres,k0,mt=death_rates,maxSP=max_sp_add+exp(max_sp))
	birth_lik = log(birth_rates)*n_spec - birth_rates*Dt
	
	#print birth_lik1, sum(birth_lik)
	
	# lik extinct	
	death_lik_de = wr(death_lik_i[te<present],W_shape,W_scale) 
	death_lik_wte = exp(-cdf_WR(W_shape,W_scale, death_lik_i[te<present]))
	lik_ext = death_lik_de * death_lik_wte
	death_lik = log(np.sum(lik_ext, axis=1)) - log(n_discrete_bins)
	
	# lik extant	
	death_lik = sum(death_lik) + np.sum(-cdf_WR(W_shape,W_scale, d[te==present]))
	
	lik = birth_lik + death_lik
	return [sum(lik), birth_rates, death_rates, Kvec]


def get_DT(origin,present,s,e): # returns the Diversity Trajectory of s,e at times T (x10 faster)
	B=np.arange(origin,present+2)
	ss1 = np.histogram(s,bins=B)[0]
	ee2 = np.histogram(e,bins=B)[0]
	DD=cumsum(ss1-ee2)
	DD[len(DD)-1] = len(e[e==present])
	return DD


# read data
f= "/Users/danielesilvestro/Downloads/tste.5.2.7.2.tsv"
f = "/Users/danielesilvestro/Desktop/UCLA_projects/infectionsampler/all_bands_1.tsv"
print "reading data..."
tbl = np.loadtxt(f,skiprows=1)
print "done"

ts = tbl[:,2]
te = tbl[:,3]
#
#
#ts,te =simulate_data(1980, 2000, shape=1.,scale=5., n=10000)

present = max(te)
origin = min(ts)


# calc diversity trajectory and number of speciation events per time bin
div=0
Dt = []
n_spec = [] # speciation events
for i in range(int(min(ts)), 1+int(max(te))):
	te_temp = te[ts<=i]
	div = len(ts[ts<=i])-len(te_temp[te_temp<=i])
	#print i, len(ts[ts<=i]), len(te_temp[te_temp<i]), te_temp
	Dt.append(div)
	#print(div)
	n_spec.append(len(ts[ts==i]))
Dt[len(Dt)-1] = len(te[te==present])
Dt = np.array(Dt)
n_spec = np.array(n_spec)
print Dt
#print sum(n_spec), len(ts)


Dt = get_DT(origin,present,ts,te)
print Dt
#quit()
#



max_sp = max(Dt)+1.




# global

l0      = 0.525
gam     = 0.7
thres   = 0.2
k0      = (Dt[0])/max_sp
W_shape = 1.        # extinction parameters (discrete Weibull model)
W_scale = 1./0.208  # extinction parameters (discrete Weibull model)
max_sp_add = 1.

# discretize death lik
n_discrete_bins = 50
discr = np.linspace(0.0001,0.999,n_discrete_bins)
d = te-ts
death_lik_i = np.zeros((len(d),n_discrete_bins))
for i in range(len(discr)):
	death_lik_i[:,i] = d + discr[i]




logfile = open("WeibullEInfectionB.log" , "wb") 
wlog=csv.writer(logfile, delimiter='\t')
head =[ "it","lik","l0","gam","thres",'k0',"W_shape","W_scale","mu0","max_sp"]
for i in range(len(Dt)): head.append("l_%s" % i)
for i in range(len(Dt)): head.append("k_%s" % i)
wlog.writerow(head)



argsA=       np.array([l0,gam ,thres ,k0   ,W_shape,W_scale, max_sp_add])
update_freq =np.array([1.,0   ,0     ,1    ,1      ,1      , 1     ])   
update_freq = update_freq/sum(update_freq)
lik_res = BDwwteDISCRETE(argsA)
likA = lik_res[0]
birth_rates = lik_res[1]
lik=likA
for iteration in range(100000):
	args = argsA+0.
	if np.random.random()< 0.2:
		res = update_multiplier_proposal(args,d=1.1,f=update_freq)
		[args, hastings] = res
	else:
		indx = np.random.choice((update_freq == 0).nonzero()[0])
		if indx==3: indx =1 
		args[indx] = update_sliding_win(args[indx])
		hastings= 0
	lik_res = BDwwteDISCRETE(args)
	lik = lik_res[0]
	if lik - likA + hastings > log(np.random.random()):
		argsA = args
		likA = lik
		birth_rates = lik_res[1]
		death_rates = lik_res[2]
		K_vec = lik_res[3]
	if iteration % 100==0: print iteration, likA, argsA
	
	if iteration % 100==0: 
		l= [iteration, likA] + list(argsA) + [death_rates]+ list(birth_rates) + list(K_vec)
		wlog.writerow(l)
		logfile.flush()
		os.fsync(logfile)
	