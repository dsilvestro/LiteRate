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
import scipy.misc
import random
from copy import deepcopy

np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)  

########################

p = argparse.ArgumentParser() #description='<input file>') 

p.add_argument('-seed',              type=int, help='random seed', default=-1,metavar=-1)
p.add_argument('-s',                 type=int, help='sampling freq', default=10000,metavar=10000)
p.add_argument('-m_birth', type=int, help='0) use const b rates 1) niche dep b', default=1,metavar=1)
p.add_argument('-m_death', type=int, help='0) use const d rates 1) niche dep d', default=1,metavar=1)
p.add_argument('-constK', type=int, help='1) use const K 0) logistic K', default=0,metavar=0)
p.add_argument('-rm_first_bin',    type=float, help='if set to 1 it removes the first time bin', default= 0, metavar= 0)
p.add_argument('-death_jitter', type=float, help="""Determines the amount to jitter death times.\
               If set to 0, lineages that lived and died in same time bin will be excluded from branch length.""", default= .5, metavar= .5)
p.add_argument('-n',       type=int, help='n. MCMC iterations', default=1000000, metavar=1000000)
p.add_argument('-d',       type=str, help='data file', default="", metavar="", required=True) 



args = p.parse_args()
sampling_freq = args.s
n_iterations = args.n

M_BIRTH = args.m_birth
M_DEATH = args.m_death
CONST_K = args.constK


########################



if args.seed== -1:
	seed=int(random.random()*1000000)
else:
	seed=args.seed

SMALL_NUMBER = 0.000000000000001 #used for flooring rates


#Helper for precompute events. Calculates branch length (standing diversity) between lo and up
def get_br(t0,t1):
	s, e  = ts+0., te+0.
	s[s<t0] = t0
	e[e>t1] = t1
	dt = e - s
	return np.sum(dt[dt>0])

def precompute_events(arg):
	[t0,t1]=arg
	n_spec_events = len(np.intersect1d((ts >= t0).nonzero()[0], (ts < t1).nonzero()[0]))
	n_exti_events = len(np.intersect1d((te > t0).nonzero()[0], (te <= t1).nonzero()[0]))
	tot_br_length = get_br(t0,t1)  
	return n_spec_events, n_exti_events, tot_br_length


def approx_log_fact(n):
	# http://mathworld.wolfram.com/StirlingsApproximation.html
	return np.log(np.sqrt((2*n+1./3)*np.pi)) + n*np.log(n) -n

def get_log_factorial(n):
	if n < 100: return np.log(scipy.misc.factorial(n))
	else: return approx_log_fact(n)

def logPoisson_pmf(x,l):
	log_pmf = (x*np.log(l) -l) - get_log_factorial(x)
	return log_pmf

def prior_gamma(x,a,s,l):
	# mean = a*s
	return scipy.stats.gamma.logpdf(x, a, scale=s,loc=l)

def prior_norm(x,loc=0,scale=1):
	return scipy.stats.norm.logpdf(x, loc, scale)


def update_sliding_win(i, m=0, M=1, d=0.05): 
	ii = i+(np.random.random()-.5)*d
	if ii>M: ii=M-(ii-M)
	if m==0: ii = abs(ii)
	return ii

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


#following notation on wikipedia except that we have added a booster for the min
def get_logistic(x,L,k,x0,div_0,nu):
	return( div_0 + L/((1+exp(-k*(x-x0)))**(1/nu)) )

def get_const_K(x,L):
	return( np.ones(len(x))*L )

def get_rate(rate_max,niche_frac):
	rate =  rate_max - (rate_max)*niche_frac
	rate[rate<=0] = SMALL_NUMBER #no negative birth rates
	return(rate)



def likelihood_function(args):
	[l_max,  k, x0, 	div_0,   L,  m_max, nu] = args

	if M_BIRTH==0:
		birth_rates = np.ones(n_time_bins)*l_max
		Kvec = np.ones(n_time_bins)
		birth_lik = np.sum(n_spec*log(birth_rates)-birth_rates*np.sum(Dt)) # log probability of speciation
	else:
		# lik speciation
		if CONST_K:
			niche = get_const_K(time_range,L)
		else: 
			niche = get_logistic(time_range,L,k,x0,div_0,nu)
		niche_frac = Dt/niche
		birth_rates = get_rate(l_max,niche_frac)
		birth_lik = np.sum(log(birth_rates)*n_spec - birth_rates*Dt)

	if M_DEATH ==0:	
		death_rates = np.ones(n_time_bins) *m_max
	elif M_DEATH ==1:
		death_rates =  get_rate(m_max,niche_frac)

	death_lik = np.sum(log(death_rates)*n_exti - death_rates*Dt)

	lik = np.array([birth_lik, death_lik])
	return [lik, birth_rates, death_rates, niche, niche_frac]


def calc_prior(args):
	#argsA=             np.array([l_max,  k,    x0, 	div_0,   L,	mu_correlation,  m_max, nu])
	p = prior_gamma(args[0],a=1,s=10,l=0) #l_max
	p += prior_gamma(args[1],a=1,s=10,l=0) #k
	p += prior_gamma(args[6],a=1,s=10,l=0) #m_max
	p += prior_gamma(args[3],a=1,s=PRIOR_K0_L,l=0) #div_0
	p += prior_gamma(args[4],a=1,s=PRIOR_K0_L,l=0) #L
	if ORIGIN + args[2]>= PRESENT: p = -np.inf #if midpoint greater than present: fail
	return p
	
	









	  
######MAIN ######

# read data
#tbl = np.loadtxt("/home/bernie/Downloads/all_bands_1.tsv",skiprows=1)
tbl = np.loadtxt(args.d,skiprows=1)

ts = tbl[:,2]
te = tbl[:,3]
te = te + args.death_jitter



PRESENT = max(te)
ORIGIN  = min(ts)


####### PRECOMPUTE VECTORS #######

n_spec = []
n_exti = []
Dt = []
bins = np.arange(ORIGIN,PRESENT+1)
for i in range(len(bins)-1):
	a,b,c = precompute_events([bins[i],bins[i+1]])
	n_spec.append(a)
	n_exti.append(b)
	Dt.append(c)
del bins

n_spec = np.array(n_spec)[:-1]
n_exti = np.array(n_exti)[:-1]
Dt = np.array(Dt)[:-1]
print("Dt",Dt)


if args.rm_first_bin:
	# remove first bin
	n_spec = n_spec[1:]
	n_exti = n_exti[1:]
	Dt = Dt[1:]
	ORIGIN +=1 
n_time_bins = len(Dt)


out=""
if CONST_K: out = "_constK"

outfile = "%s_%s%s.log" % (os.path.splitext(args.d)[0], seed, out)
logfile = open(outfile , "w") 
wlog=csv.writer(logfile, delimiter='\t')
head =["it","posterior","likelihood","likelihood_birth","likelihood_death","prior","l_max","steepness_k","midpoint_x0",\
"initCarryingCap","maxCarryingCap","m_max","nu"]
for i in range(len(Dt)): head.append("l_%s" % i)
for i in range(len(Dt)): head.append("m_%s" % i)
for i in range(len(Dt)): head.append("niche_%s" % i)
for i in range(len(Dt)): head.append("nicheFrac_%s" % i)

wlog.writerow(head)



PRIOR_K0_L = np.max(Dt) # scale of Gamma(1,s) prior


L = 20000 # maximum
k = 1.5 # steepness
x0 = 12 # midpoint
div_0 = 10 # starting carrying capacity
l_max = 0.5 #max speciation rate
m_max = 0.20  #max extinction rate
nu = 1.

time_range = np.arange(n_time_bins).astype(float)


argsA=np.array([l_max,  k, x0, 	div_0,   L,  m_max, nu])


#figure out which params to update based on model Note that nu is an extended logistic param which we are not currently using

#constant birth and death
if M_BIRTH==0 and M_DEATH==0:
	#argsA=             np.array([l_max,  k,    x0, 	div_0,   L,	 m_max, nu])
	update_multiplier = np.array([1.,  0,	0,        0,   0,         1 , 0])   

elif CONST_K==1:
	#argsA=             np.array([l_max,  k,    x0, 	div_0,   L,	m_max, nu])
	update_multiplier = np.array([1.,  0,	0,        0,   1,    1 , 0])   

else:
	#argsA=             np.array([l_max,  k,    x0, 	div_0,   L,	 m_max, nu])
	update_multiplier = np.array([1.,  1,	0,        1,   1,      1 , 0])   


update_multiplier = update_multiplier/sum(update_multiplier)

#initialize likelihood
lik_res = likelihood_function(argsA)
likA = np.sum(lik_res[0])
likDeathA = lik_res[0][1]
birth_rates = lik_res[1]
death_rates = lik_res[2]
niche = lik_res[3]
nicheFrac = lik_res[4]
lik=likA
priorA = calc_prior(argsA)
prior=priorA

iteration = 0
while iteration != n_iterations:
	args = argsA+0.
	updated_ext = 0
	hastings= 0
	rr = np.random.random(2)
	if rr[1]<0.1 and CONST_K==0:
		res = argsA+0
		res[2] = update_sliding_win(res[2], m=0, M=PRESENT, d=1.5) #update midpoint (the only sliding window proposal)
		res = [res,0]
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
		argsO[2] = ORIGIN+argsO[2]
		argsO[6] = 1./argsO[6]
		if CONST_K is False: argsO[3]+argsO[5] #if K is min then 
		#argsO[3]=Dt[0]+argsA[3] #calculate Kmin
		#argsO[4]= max_obs_diversity+argsA[4] #calculate Kmax
		print(iteration, likA, argsO) #, args
		l= [iteration,likA+priorA, likA,likBirthA,likDeathA, priorA] + list(argsO) + list(birth_rates) + list(death_rates) + list(niche) + list(nicheFrac)
		wlog.writerow(l)
		logfile.flush()
		os.fsync(logfile)
		
	iteration += 1
	