#!/usr/bin/env python 
# Created by Daniele Silvestro on 20/08/2019
import argparse, os,sys, platform, time, csv
from numpy import *
import numpy as np
from scipy.special import gamma
from scipy.special import beta as f_beta
import random as rand
from scipy.special import gdtr, gdtrix
from scipy.special import betainc
import scipy.stats
#np.set_printoptions(suppress=True)
#np.set_printoptions(precision=3)  
from time import time
t1=time()
SEED=12345
np.random.seed(SEED)

def cond_alpha_proposal(hp_gamma_shape,hp_gamma_rate,current_alpha,k,n):
	z = [current_alpha + 1.0, float(n)]
	f = np.random.dirichlet(z,1)[0]
	eta = f[0]
	u = np.random.uniform(0,1,1)[0]
	x = (hp_gamma_shape + k - 1.0) / ((hp_gamma_rate - np.log(eta)) * n)
	if (u / (1.0-u)) < x: new_alpha = np.random.gamma( (hp_gamma_shape+k), (1./(hp_gamma_rate-np.log(eta))) )
	else: new_alpha = np.random.gamma( (hp_gamma_shape+k-1.), 1./(hp_gamma_rate-np.log(eta)) )
	return new_alpha

def get_rate_HP(n,target_k,hp_gamma_shape):
	def estK(alpha,N):
		return sum([alpha/(alpha+i-1) for i in range(1,int(N+1))])

	def opt_gamma_rate(a):
		a= abs(a[0])
		ea =estK(a,n)
		return exp(abs( ea-target_k ))
	
	from scipy.optimize import fmin_powell as Fopt1
	opt = Fopt1(opt_gamma_rate, [np.array(0.001)], full_output=1, disp=0)
	expected_cp=abs(opt[0])
	hp_gamma_rate = expected_cp/hp_gamma_shape
	return hp_gamma_rate

# select random element based on fixed probabilities
def random_choice_P(vector):
	ind = np.random.choice(np.arange(len(vector)), p=vector/np.sum(vector))
	return [vector[ind], ind]

def calc_rel_prob(log_lik):
	rel_prob=exp(log_lik-max(log_lik))
	return rel_prob/sum(rel_prob)

def F(X,m,sd=1):
	return scipy.stats.norm.pdf(X,loc=m,scale=sd)

def logLikNorm(X,par):
	m, sd = par
	return scipy.stats.norm.logpdf(X,loc=m,scale=sd)

def logLikBeta(X,par):
	a, b = par
	return scipy.stats.beta.logpdf(X,a,b)

def get_log_pmf_beta_discrete(num_bins, a, b):
	bins = np.linspace(0, 1, num_bins + 1)
	cdf_b = scipy.stats.beta.cdf(bins, a, b, loc=0, scale=1)
	return np.log(np.diff(cdf_b))


def G0_norm_mean(n=1):
	#return np.random.gamma(shape=alpha,scale=1./beta,size=n)
	return np.random.uniform(1,20,size=n)

def G0_norm_std(n=1):
	#return np.random.gamma(shape=alpha,scale=1./beta,size=n)
	return np.random.uniform(0.1,2,size=n)

def G0_norm_std_(n=1):
	#return np.random.gamma(shape=alpha,scale=1./beta,size=n)
	return np.ones(1)


def G0_beta_shape(n=1):
	return np.exp(np.random.uniform(-2.3, 1.61, size=n))


def update_multiplier_proposal(q,d=1.1,f=0.25):
	S=np.shape(q)
	ff=np.random.binomial(1,f,S)
	u = np.random.uniform(0,1,S)
	l = 2*log(d)
	m = exp(l*(u-.5))
	m[ff==0] = 1.
	new_q = q * m
	U=sum(log(m))
	return new_q,U


def update_normal(i, d=0.05):
	I=np.random.choice(len(i))
	z=np.zeros(len(i))+i
	z[I]=fabs(z[I]+np.random.normal(0,d))
	return z, 0

def DDP_gibbs_sampler(arg): 
	[data,n_data,parA,ind,alpha_par_Dir]=arg
	# GIBBS SAMPLER for NUMBER OF CATEGORIES - Algorithm 4. (Neal 2000)
	par=[np.copy(j) for j in parA] # list of parameters, one array for each 
	eta = np.array([len(ind[ind==j]) for j in range(len(par[0]))]) # number of elements in each category
	#eta = np.unique(ind, return_count=True)  # number of elements in each category
	u1 = np.random.uniform(0,1,n_data) # init random numbers
	new_lik_vec=np.zeros(n_data) # store new sampled likelihoods
	#new_alpha_par_Dir = alpha_par_Dir 
	new_alpha_par_Dir = 0 + cond_alpha_proposal(hp_gamma_shape,hp_gamma_rate,alpha_par_Dir,len(eta),n_data)
	for i in range(n_data):
		d= data[i]
		k1 = len(par[0])

		if len(ind[ind==ind[i]])==1: # is singleton
			k1 = k1 - 1
			par_k1 = [np.copy(j) for j in par]
			if u1[i]<= k1/(k1+1.): pass
			else: ind[i] = k1 + 1 # this way n_ic for singleton is not 0
		else: # is not singleton
			par_k1 = [np.copy(j) for j in par]
			for j in range(len(par)):
				f_g = list_of_G_functions[j]
				#print par[j], par_k1, f_g()
				par_k1[j] = np.concatenate((par_k1[j],f_g()), axis=0)

		# construct prob vector FAST!
		lik_vec=np.array([logLik(d_i,par_k1) for d_i in d])
		lik_vec = np.sum(lik_vec, axis=0)
		rel_lik = calc_rel_prob(lik_vec)
		if len(par_k1[0])>len(eta): # par_k1 add one element only when i is not singleton
			eta[ind[i]] -= 1
			eta_temp=np.append(eta,new_alpha_par_Dir/(k1+1.))
		else: eta_temp = eta
		#print eta_temp, rel_lik
		P=eta_temp*rel_lik

		# randomly sample a new value for indicator ind[i]
		IND = random_choice_P(P)[1]  # numpy.random.choice(a, size=None, replace= 1, p=None)
		ind[i] = IND # update ind vector
		if IND==(len(par_k1[0])-1): par = par_k1 # add category


		# Change the state to contain only those par are now associated with an observation
		# create vector of number of elements per category
		eta = np.array([len(ind[ind==j]) for j in range(len(par[0]))])
		# remove parameters for which there are no elements
		#print eta, par
		for l in range(len(par)):
			par[l] = par[l][eta>0]
		# rescale indexes
		ind_rm = (eta==0).nonzero()[0] # which category has no elements
		if len(ind_rm)>0: ind[ind>=ind_rm] = ind[ind>=ind_rm]-1
		# update eta
		eta = np.delete(eta,ind_rm)

		# Update lik vec
		new_lik_vec[i]=lik_vec[IND]

	likA = np.sum(new_lik_vec)
	parA = par
	return likA,parA, ind,new_alpha_par_Dir


run_normal = 1
run_beta   = 0
# simulate NORMAL data
if run_normal:
	a1=np.random.normal(15,.1,50)
	a2=np.random.normal(10,1 ,50)
	a3=np.random.normal(15,.1,50)
	a4=np.random.normal(10,1 ,50)
	data=np.concatenate((a1,a2,a3,a4), axis=0)
	data = data.reshape((20,10))
	data_list = [i for i in data]
	data =data_list
	list_of_G_functions = [G0_norm_mean,G0_norm_std]
	list_proposals = [ update_normal, update_multiplier_proposal]
	parA = [np.random.uniform(10,15,1),np.random.uniform(10,15,1)] # vector of unique parameters (of len = K)
	logLik = logLikNorm

if run_beta:
	a1=np.random.beta(4.0,4.0,50)
	a2=np.random.beta(4.0,4.0,50)
	a3=np.random.beta(1.5,0.2,50)
	a4=np.random.beta(1.0,1.0,50)
	data=np.concatenate((a1,a2,a3,a4), axis=0)
	data = data.reshape((20,10))
	data_list = [i for i in data]
	data =data_list
	print(data)
	list_of_G_functions = [G0_beta_shape,G0_beta_shape]
	parA = [np.random.uniform(0.8,1.1,1),np.random.uniform(0.8,1.1,1)] # vector of unique parameters (of len = K)
	list_proposals = [ update_multiplier_proposal, update_multiplier_proposal]
	logLik = logLikBeta



n_data= len(data)

# init psi, indicators
ind = np.zeros(n_data).astype(int)        # indicators (of len = n_data)

target_k = 1
hp_gamma_shape = 2.
hp_gamma_rate  = get_rate_HP(n_data,target_k,hp_gamma_shape)
print(hp_gamma_rate)

alpha_par_Dir = 1


out_log = "mcmc_DPP.log" 
logfile = open(out_log , "w") 
head="it\tlikelihood\tK\tDir_alpha"
for i in range(len(data)): head+= "\tm_%s" % (i+1)
for i in range(len(data)): head+= "\ts_%s" % (i+1)
head=head.split("\t")
wlog=csv.writer(logfile, delimiter='\t')
wlog.writerow(head)

for IT in range(10000):
	
	if np.random.uniform()< 0.5 or IT == 0:
		
		arg = [data,n_data,parA,ind,alpha_par_Dir]			
		likA,parA, ind,alpha_par_Dir = DDP_gibbs_sampler(arg)
		
	else:
		# STANDARD MCMC
		par = [0,0]
		par[0], hasting1 = list_proposals[0](parA[0])
		par[1], hasting2 = list_proposals[1](parA[1])
		lik = np.sum([logLik(data[i],[par[0][ind[i]],par[1][ind[i]]]) for i in range(n_data)])
		
		if lik-likA + hasting1+hasting2 >= log(np.random.uniform(0,1)):
			likA = lik
			parA = par
	
	if IT % 10 == 0: print(round(likA,3), parA, len(parA[0]))
	
	if IT % 50 == 0: 
		log_state= [IT,likA,len(parA[0]),alpha_par_Dir] 
		
		for i in range(len(parA)):
			log_state = log_state+list(np.around(parA[i][ind],3))
		
		wlog.writerow(log_state)
		logfile.flush()
		os.fsync(logfile)
	
	
	
print(ind)

print( "elapsed time:", time()-t1)

	























