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
p.add_argument('-useADE',            type=int, help='1) use ADE 0) constant extinction', default=1,metavar=1)
p.add_argument('-s',                 type=int, help='sampling freq', default=1000,metavar=1000)
p.add_argument('-useInfectionModel', type=int, help='1) use Infection model 0) constant speciation', default=1,metavar=1)


args = p.parse_args()
useInfectionModel = args.useInfectionModel
sampling_freq = args.s
useADE = args.useADE

########################



present = 2000
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


def birth_rates_infection(Dt,l0=0.5,gam=0.2,thres=0.5,k0t=1,kmt=1,mt=0):
	D = Dt/(max(Dt)+kmt)
	Kvec = np.zeros(len(Dt))
	Kvec[0] = (Dt[0]+k0t)/(max(Dt)+kmt)
	for i in range(1,len(Dt)): Kvec[i] = Kvec[i-1] + (D[i-1]*gam + thres) *(1-Kvec[i-1]) 	
	lt = l0 - (l0-mt)*(D/Kvec) + 0.000001 # avoid exactly 0 rates
	return lt, Kvec


def get_logistic(x,L,k,x0):
	return( L/(1+exp(-k*(x-x0))) )


def get_lambda(l0,m0,niche_frac):
	l_temp =  l0 - (l0-m0)*niche_frac
	l_temp[l_temp<=0] = small_number
	return(l_temp)



def BDwwteDISCRETE(args,updated_ext, D_lik):
	#[l0,gam,thres,k0_t,km_t,W_shape,W_scale] = args
	[l_0,  k, x0, 	div_0,   L,	W_shape,  W_scale] = args
	# d = te - ts
	# de = d[te<present] #takes only the extinct species times
	longevity = W_scale* gamma(1+1/W_shape)
	m_0 = 1./longevity
	if useInfectionModel==0:
		birth_rates = np.ones(n_time_bins)*l0
		Kvec = np.ones(n_time_bins)
		birth_lik = np.sum(n_spec*log(birth_rates)-birth_rates*np.sum(Dt)) # log probability of speciation
	else:
		# lik speciation
		niche = get_logistic(x,L,k,x0) + div_0
		niche_frac = Dt/niche
		birth_rates = get_lambda(l_0,m_0,niche_frac)
		birth_lik = np.sum(log(birth_rates)*n_spec - birth_rates*Dt)
	# lik extinct
	if updated_ext==1:
		death_lik_de = wr(death_lik_i[te<present],W_shape,W_scale) 
		#avg_death_rates=np.mean(death_lik_de,axis=0)
		death_lik_wte = np.exp(-cdf_WR(W_shape,W_scale, death_lik_i[te<present]))
		lik_ext = death_lik_de * death_lik_wte
		death_lik_extinct = log(np.sum(lik_ext, axis=1)) - log_n_discrete_bins	
		# lik extant	
		death_lik_extant = log(np.sum(np.exp(-cdf_WR(W_shape,W_scale, death_lik_i[te==present])), axis=1)) - log_n_discrete_bins
		death_lik = np.sum(death_lik_extinct) + np.sum(death_lik_extant)
	else:
		death_lik = D_lik
	
	lik = np.array([birth_lik, death_lik])
	return [lik, birth_rates, 1./longevity, niche, niche_frac]


def calc_prior(args):
	#args=  [l0,  gam,	thres,   k0_t,  km_t,	W_shape,  W_scale]
	p = 0 #prior_gamma(args[4]+max_obs_diversity,a=shapeG,s=scaleG,l=locG)
	return p
	
	









	  


# read data
tbl = np.loadtxt("/Users/danielesilvestro/Software/LiteRate/all_bands_1.tsv",skiprows=1)
ts = tbl[:,2]
te = tbl[:,3]

# calc diversity trajectory and number of speciation events per time bin
div=0
Dt = []
n_spec = [] # speciation events
for i in range(int(min(ts)), 1+int(max(te))):
	te_temp = te[ts<=i]
	div = len(ts[ts<=i])-len(te_temp[te_temp<=i])
	#print i, len(ts[ts<=i]), len(te_temp[te_temp<i]), te_temp
	Dt.append(div)
	print(div)
	n_spec.append(len(ts[ts==i]))

Dt[len(Dt)-1] = len(te[te==present])
Dt = np.array(Dt)
n_spec = np.array(n_spec)
n_time_bins = len(Dt)
print Dt
print sum(n_spec), len(ts)
# global







# discretize death lik
n_discrete_bins = 50
discr = np.linspace(0.0001,0.999,n_discrete_bins)
d = te-ts
death_lik_i = np.zeros((len(d),n_discrete_bins))
for i in range(len(discr)):
	death_lik_i[:,i] = d + discr[i]
log_n_discrete_bins = np.log(n_discrete_bins)

out=""
if useADE==1: out = "_ADE"
if useInfectionModel==1: out += "_Inf"


outfile = "allbands_%s%s.log" % (seed, out)
logfile = open(outfile , "wb") 
wlog=csv.writer(logfile, delimiter='\t')
head =["it","lik","prior","l0","steepness_k","midpoint_x0","initCarryingCap","maxCarryingCap_L","W_shape","W_scale","m0"]
for i in range(len(Dt)): head.append("l_%s" % i)
for i in range(len(Dt)): head.append("niche_%s" % i)
for i in range(len(Dt)): head.append("nicheFrac_%s" % i)

wlog.writerow(head)




L = 20000 # maximum
k = 1.5 # steepness
x0 = 12 # midpoint
div_0 = 10 # starting carrying capacity
l0 = 0.5
m_0 = 0.13
W_shape = 1.        # extinction parameters (discrete Weibull model)
W_scale = 1./0.20  # extinction parameters (discrete Weibull model)

x = np.arange(n_time_bins).astype(float) # 31 years discretized in 100 steps
dT = x[1] # duration in years of each time step




argsA=                         np.array([l0,  k, x0, 	div_0,   L,	W_shape,  W_scale])

BDwwteDISCRETE(argsA,updated_ext=0,D_lik=0)











#argsA=                         np.array([l0,  gam,	thres,   k0_t,  km_t,	W_shape,  W_scale])
if args.useADE==1: update_freq =np.array([1.,  0,	0,       0,     0,      1,        1      ])   
else:              update_freq =np.array([1.,  0,	0,       1,     1,      0,        1      ])   
update_freq = update_freq/sum(update_freq)
#update_sliding = np.array([1.,  0,	0,       1,     1,      1,        1      ])   
update_sliding = np.array([0 ,  1,	1,       0,     0,      0,        0      ])   
lik_res = BDwwteDISCRETE(argsA,updated_ext=1,D_lik=0)
likA = np.sum(lik_res[0])
likDeathA = lik_res[0][1]
birth_rates = lik_res[1]
death_rate = lik_res[2]
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
	#if rr[0] < 0.1 or useInfectionModel==0:
	if rr[1]<0.1: # update extinction
		if useADE: res = update_multiplier_proposal_vec(args,d=1.1,f=np.array([0.,0,0,0,0,1,1]))
		else: res = update_multiplier_proposal_vec(args,d=1.1,f=np.array([0.,0,0,0,0,0,1]))
		updated_ext = 1
	else:
		res = update_multiplier_proposal_vec(args,d=1.1,f=np.array([1,1,1,1,1,0,0.]))	
	[args, hastings] = res
	#elif rr[0] < 0.35:
	#	res = update_multiplier_proposal_vec(args,d=1.25,f=np.array([0.,0,0,1,1,0,0]))
	#	args = res[0]
	#	hastings = res[1]
	#	#res1 = update_poisson_proposal(args[3])
	#	#args[3] = res1[0]
	#	#hastings += res1[1]
	#elif useInfectionModel:
	#	indx = np.random.choice((update_sliding == 1).nonzero()[0]) 
	#	args[indx] = update_sliding_win(args[indx])
	lik_res = BDwwteDISCRETE(args,updated_ext,likDeathA)
	#lik_res = BDwwteDISCRETE(args,0,likDeathA)
	lik = np.sum(lik_res[0])
	prior = calc_prior(args)
	#if args[1] + args[2] > 1: prior = -np.inf #dkDT cannot grow by more than (1-K), 
	if (lik - likA) + (prior - priorA) + hastings > log(np.random.random()):
		argsA = args
		priorA = prior
		likA = lik
		likDeathA = lik_res[0][1]
		birth_rates = lik_res[1]
		death_rate  = lik_res[2]
		niche = lik_res[3]
		nicheFrac = lik_res[4]
	if iteration % sampling_freq==0:
		#print lik,prior, args
		argsO=deepcopy(argsA)
		#argsO[3]=Dt[0]+argsA[3] #calculate Kmin
		#argsO[4]= max_obs_diversity+argsA[4] #calculate Kmax
		print iteration, likA, argsO #, args
		l= [iteration, likA, priorA] + list(argsO) + [death_rate] + list(birth_rates) + list(niche) + list(nicheFrac)
		wlog.writerow(l)
		logfile.flush()
		os.fsync(logfile)
		
	iteration += 1
	