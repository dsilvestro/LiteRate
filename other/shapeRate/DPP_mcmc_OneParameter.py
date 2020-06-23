#!/usr/bin/env python 
# Created by Daniele Silvestro on 02/03/2012 => pyrate.help@gmail.com 
import argparse, os,sys, platform, time, csv
from numpy import *
import numpy as np
from scipy.special import gamma
from scipy.special import beta as f_beta
import random as rand
from scipy.special import gdtr, gdtrix
from scipy.special import betainc
import scipy.stats
np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)  
from time import time
t1=time()


# select random element based on fixed probabilities
def random_choice_P(vector):
	ind = np.random.choice(np.arange(len(vector)), p=vector/np.sum(vector))
	#probDeath=np.cumsum(vector/sum(vector)) # cumulative prob (used to randomly sample one 
	#r=rand.random()                          # parameter based on its deathRate)
	#probDeath=sort(append(probDeath, r))
	#ind=np.where(probDeath==r)[0][0] # just in case r==1
	return [vector[ind], ind]


def F(X,m,sd=1):
	return scipy.stats.norm.pdf(X,loc=m,scale=sd)

def logLik(X,m,sd=1):
	return sum(scipy.stats.norm.logpdf(X,loc=m,scale=sd))

def G0(alpha=2,beta=2,n=1):
	#return np.random.gamma(shape=alpha,scale=1./beta,size=n)
	return np.random.uniform(1,20,size=n)

def update_multiplier_proposal(i,d=1.5):
	S=shape(i)
	u = np.random.uniform(0,1,S)*np.rint(np.random.uniform(0,.65,S))
	l = 2*log(d)
	m = exp(l*(u-.5))
	#print "\n",u,m,"\n"
 	ii = i * m
	U=sum(log(m))
	return ii, U

def update_normal(i, d=0.5):
	I=np.random.choice(len(i))
	z=np.zeros(len(i))+i
	z[I]=fabs(z[I]+np.random.normal(0,d))
	return z, 0


# CONCENTRATION PARAMETER
alpha = 1
print "\nConcentration parameter:", alpha,"\n"


# simulate data
a1=np.random.normal(10,1,25)
a2=np.random.normal(15,1,25)
a3=np.random.normal(10,1,25)
a4=np.random.normal(15,1,25)
Y = A=np.concatenate((a1,a2,a3,a4), axis=0)

n_data= len(Y)

# init psi, indicators
parA = np.array([np.random.uniform(10,15)]) # vector of unique parameters (of len = K)
ind = np.zeros(n_data).astype(int)        # indicators (of len = n_data)

out_log = "mcmc_alpha_%s.log" % (alpha)
logfile = open(out_log , "w",0) 
head="it\tlikelihood\tK"
for i in range(len(Y)): head+= "\tm_%s" % (i+1)
head=head.split("\t")
wlog=csv.writer(logfile, delimiter='\t')
wlog.writerow(head)

for IT in range(1000):
	
	if np.random.uniform()<0.5 or IT == 0:
		# GIBBS SAMPLER for NUMBER OF CATEGORIES - Algorithm 4. (Neal 2000)
		par=parA # parameters for each category
		eta = np.array([len(ind[ind==i]) for i in range(len(par))]) # number of elements in each category
		u1 = np.random.uniform(0,1,n_data) # init random numbers
		for i in range(0,n_data):
			k1 = len(par)
	
			if len(ind[ind==ind[i]])==1: # is singleton
				k1 = k1 - 1
				par_k1 = par			
				if u1[i]<= k1/(k1+1.): pass
				else: ind[i] = k1 + 1 # this way n_ic for singleton is not 0
			else: # is not singleton
				par_k1 = np.concatenate((par,G0()), axis=0)
		
			#__	 construct prob vector (slow version)
			#__	P=list()
			#__	for c in range(0,k1+1):
			#__		# no. of eleents with ind=c (except element i)
			#__		n_ic = len(ind[ind==c])
			#__		if ind[i]==c: n_ic = n_ic - 1
		        #__	
			#__		if c+1 <= k1: 
			#__			#print c, k1, n_ic,F(Y[i],par[c]), n_ic * F(Y[i],par[c]) 
			#__			P.append( n_ic * F(Y[i],par_k1[c]) )
			#__		else:
			#__			#print "HERE:",alpha/(k1+1.) * F(Y[i],par_k1) ,alpha/(k1+1.), F(Y[i],par_k1)
			#__			P.append( alpha/(k1+1.) * F(Y[i],par_k1[c]) )
		
			
			
			# construct prob vector FAST!
			F_values= F(Y[i],par_k1) 
			if len(par_k1)>len(eta): # par_k1 add one element only when i is not singleton
				eta[ind[i]] -= 1
				eta_temp=np.append(eta,alpha/(k1+1.))
			else: eta_temp = eta
			P=eta_temp*F_values
		
			# randomly sample a new value for indicator ind[i]
			IND = random_choice_P(P)[1] 
			ind[i] = IND # update ind vector
			if IND==(len(par_k1)-1): par = par_k1 # add category
	
	
			# Change the state to contain only those par are now associated with an observation
			# create vector of number of elements per category
			eta = np.array([len(ind[ind==i]) for i in range(len(par))])
			# remove parameters for which there are no elements
			par = par[eta>0]
			# rescale indexes
			ind_rm = (eta==0).nonzero()[0] # which category has no elements
			if len(ind_rm)>0: ind[ind>=ind_rm] = ind[ind>=ind_rm]-1
			# update eta
			eta = np.delete(eta,ind_rm)
	
		likA = logLik(Y,par[ind])
		parA = par
	else:
		# STANDARD MCMC
		par, hasting = update_normal(parA)
		lik = logLik(Y,par[ind])
		
		if lik-likA + hasting >= log(np.random.uniform(0,1)):
			likA = lik
		        parA = par
	
	if IT % 10 == 0: print round(likA,3), par, eta, len(par)
	
	if IT % 10 == 0: 
		log_state= [IT,likA,len(parA)]+list(np.around(parA[ind],3))
		wlog.writerow(log_state)
		logfile.flush()
		os.fsync(logfile)
	
	
	
print ind

print "elapsed time:", time()-t1

	























