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
run_discrete = 1 # if set to 1 times of origination and extinctions are rounded to the closest integer

########################

p = argparse.ArgumentParser() #description='<input file>') 

p.add_argument('-seed',              type=int, help='random seed', default=-1,metavar=-1)
p.add_argument('-extendedLogistic',            type=int, help='set to 0 forsimple logistic', default=0,metavar=0)
p.add_argument('-s',                 type=int, help='sampling freq', default=10000,metavar=10000)
p.add_argument('-m_birth', type=int, help='0) use const b rates 1) niche dep b', default=1,metavar=1)
p.add_argument('-m_death', type=int, help='0) use const d rates 1) niche dep d', default=1,metavar=1)
p.add_argument('-constK', type=int, help='1) use const K 0) logistic K', default=0,metavar=0)



args = p.parse_args()
sampling_freq = args.s
extendedLogistic = args.extendedLogistic

m_birth = args.m_birth
m_death = args.m_death
constK = args.constK


########################



if args.seed== -1:
	seed=int(random.random()*1000000)
else:
	seed=args.seed

small_number = 0.000000000000001


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


def get_logistic(x,L,k,x0,div_0,nu):
	return( div_0 + L/((1+exp(-k*(x-x0)))**(1/nu)) )

def get_const_K(x,L):
	return( np.ones(len(x))*L )

def get_lambda(l0,niche_frac):
	l_temp =  l0 - (l0)*niche_frac
	l_temp[l_temp<=0] = small_number
	return(l_temp)



def BDwwteDISCRETE(args,updated_ext, D_lik):
	#[l0,gam,thres,k0_t,km_t,W_shape,W_scale] = args
	[l_0,  k, x0, 	div_0,   L,	mu_correlation,  W_scale, nu] = args
	# d = te - ts
	# de = d[te<present] #takes only the extinct species times
	longevity = W_scale* gamma(1+1/W_shape)
	m_0 = 1./W_scale
	if m_birth==0:
		birth_rates = np.ones(n_time_bins)*l0
		Kvec = np.ones(n_time_bins)
		birth_lik = np.sum(n_spec*log(birth_rates)-birth_rates*np.sum(Dt)) # log probability of speciation
	else:
		# lik speciation
		if constK:
			niche = get_const_K(x,L)
		else: 
			niche = get_logistic(x,L,k,x0,div_0,nu)
		niche_frac = Dt/niche
		birth_rates = get_lambda(l_0,niche_frac)
		birth_lik = np.sum(log(birth_rates)*n_spec - birth_rates*Dt)

	if m_death ==0:	
		death_rates = np.ones(n_time_bins) *m_0
	elif m_death ==1:
		death_rates =  get_lambda(m_0,niche_frac)
	#elif m_death==2:
	#	death_rates = m_0 * np.exp(mu_correlation*log(Dt/float(max(Dt))))   
	#elif m_death==3:
	#	niche_delay = get_logistic(x,L,k,x0+mu_correlation,div_0,nu)
	#	niche_frac = Dt/niche_delay
	#	death_rates =  get_lambda(m_0,niche_frac)
	#	death_lik = np.sum(log(death_rates)*n_exti - death_rates*Dt)
		

	death_lik = np.sum(log(death_rates)*n_exti - death_rates*Dt)

	lik = np.array([birth_lik, death_lik])
	return [lik, birth_rates, death_rates, niche, niche_frac]


def calc_prior(args):
	#args=    np.array([l0,  k, x0, 	div_0,   L,	W_shape,  W_scale, nu])
	p = prior_gamma(args[0],a=1,s=10,l=0) + prior_gamma(args[1],a=1,s=10,l=0) + prior_gamma(1./args[6],a=1,s=10,l=0)
	p += prior_gamma(args[3],a=1,s=prior_k0_L,l=0)+prior_gamma(args[4],a=1,s=prior_k0_L,l=0)
	p += prior_norm(np.log(args[3]),loc=0,scale=2)
	if origin + args[2]>= present: p = -np.inf
	return p
	
	









	  


# read data
tbl = np.loadtxt("all_bands_1newdata.tsv",skiprows=1)
ts = tbl[:,2]
te = tbl[:,3]


present = max(te)
origin  = min(ts)

print present - origin


# check longevities of extinct
ss = ts[te < present]
ee = te[te < present]
d=ee-ss
[len(d[d==i]) for i in range(0,30)]


# calc diversity trajectory and number of speciation events per time bin
div=0
Dt = []
n_spec = [] # speciation events
n_exti = [] # ext events
for i in range(int(min(ts)), 1+int(max(te))):
	te_temp = te[ts<=i]
	div = len(ts[ts<=i])-len(te_temp[te_temp<=i])
	#print i, len(ts[ts<=i]), len(te_temp[te_temp<i]), te_temp
	Dt.append(div)
	print(div)
	n_spec.append(len(ts[ts==i]))
	n_exti.append(len(te[te==i]))
n_exti[len(n_exti)-1] = 0
Dt[len(Dt)-1] = len(te[te==present])
Dt = np.array(Dt)
n_spec = np.array(n_spec)
n_exti = np.array(n_exti)
n_time_bins = len(Dt)
print Dt
print sum(n_spec), len(ts)
print "n sp :", n_spec
print "n ex :", n_exti
prior_k0_L = np.max(Dt) # scale of Gamma(1,s) prior







# discretize death lik
n_discrete_bins = 50
discr = np.linspace(0.0001,0.999,n_discrete_bins)
d = te-ts
death_lik_i = np.zeros((len(d),n_discrete_bins))
for i in range(len(discr)):
	death_lik_i[:,i] = d + discr[i]
log_n_discrete_bins = np.log(n_discrete_bins)

out=""
if extendedLogistic==1: out = "_extLog"
if constK: out = "_constK"


outfile = "allbands_%s%s.log" % (seed, out)
logfile = open(outfile , "wb") 
wlog=csv.writer(logfile, delimiter='\t')
head =["it","posterior","likelihood","likelihood_birth","likelihood_death","prior","l0","steepness_k","midpoint_x0",\
"initCarryingCap","maxCarryingCap_L","W_shape","m0","nu"]
for i in range(len(Dt)): head.append("l_%s" % i)
for i in range(len(Dt)): head.append("m_%s" % i)
for i in range(len(Dt)): head.append("niche_%s" % i)
for i in range(len(Dt)): head.append("nicheFrac_%s" % i)

wlog.writerow(head)




L = 20000 # maximum
k = 1.5 # steepness
x0 = 12 # midpoint
div_0 = 10 # starting carrying capacity
l0 = 0.5
m_0 = 0.13
W_shape = 0.1        # extinction parameters (discrete Weibull model)
W_scale = 1./0.20  # extinction parameters (discrete Weibull model)
nu = 1.

x = np.arange(n_time_bins).astype(float) # 31 years discretized in 100 steps
dT = x[1] # duration in years of each time step




argsA=                         np.array([l0,  k, x0, 	div_0,   L,	W_shape,  W_scale, nu])

BDwwteDISCRETE(argsA,updated_ext=0,D_lik=0)








if m_birth==1 and m_death==1:
	#argsA=             np.array([l0,  k,    x0, 	div_0,   L,	mu_correlation,  W_scale, nu])
	update_multiplier = np.array([1.,  1,	0,        1,   1,              0,        1 , 1])   

if m_birth==1 and m_death==2:
	#argsA=             np.array([l0,  k,    x0, 	div_0,   L,	mu_correlation,  W_scale, nu])
	update_multiplier = np.array([1.,  1,	0,        1,   1,              1,        1 , 1])   

if m_birth==1 and m_death==3:
	#argsA=             np.array([l0,  k,    x0, 	div_0,   L,	mu_correlation,  W_scale, nu])
	update_multiplier = np.array([1.,  1,	0,        1,   1,              1,        1 , 1])   

#if m_birth==0 and m_death==2:
#	#argsA=          np.array([l0,  k, x0, 	div_0,   L,	mu_correlation,  W_scale, nu])
#	update_sliding = np.array([1 ,  1,	0,        1,   1,              1,        1 , 1])   
#

if extendedLogistic==0:
	#argsA=             np.array([l0,  k,    x0, 	div_0,   L,	mu_correlation,  W_scale, nu])
	update_multiplier = np.array([1.,  1,	0,        1,   1,              0,        1 , 0])   

if constK:
	#argsA=             np.array([l0,  k,    x0, 	div_0,   L,	mu_correlation,  W_scale, nu])
	update_multiplier = np.array([1.,  0,	0,        0,   1,              0,        1 , 0])   


update_multiplier = update_multiplier/sum(update_multiplier)
#update_sliding = np.array([1.,  0,	0,       1,     1,      1,        1      , 0])   
update_sliding = np.array([0 ,  1,	1,       0,     0,      0,        0      , 0])   
lik_res = BDwwteDISCRETE(argsA,updated_ext=1,D_lik=0)
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
while iteration != 50000000:
	args = argsA+0.
	updated_ext = 0
	hastings= 0
	rr = np.random.random(2)
	if rr[1]<0.1 and constK==0:
		res = argsA+0
		res[2] = update_sliding_win(res[2], m=0, M=present, d=1.5)
		res = [res,0]
	else:
		res = update_multiplier_proposal_vec(args,d=1.1,f=update_multiplier)

	[args, hastings] = res
	lik_res = BDwwteDISCRETE(args,updated_ext,likDeathA)
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
		argsO=deepcopy(argsA)
		argsO[2] = origin+argsO[2]
		argsO[6] = 1./argsO[6]
		#argsO[3]=Dt[0]+argsA[3] #calculate Kmin
		#argsO[4]= max_obs_diversity+argsA[4] #calculate Kmax
		print iteration, likA, argsO #, args
		l= [iteration,likA+priorA, likA,likBirthA,likDeathA, priorA] + list(argsO) + list(birth_rates) + list(death_rates) + list(niche) + list(nicheFrac)
		wlog.writerow(l)
		logfile.flush()
		os.fsync(logfile)
		
	iteration += 1
	