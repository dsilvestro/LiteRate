#!/usr/bin/env python 
import argparse,sys
import os, csv, glob
from numpy import *
import numpy as np
from scipy.special import gamma
from scipy.special import beta as f_beta
import random as rand
import platform, time
import multiprocessing, thread
import multiprocessing.pool
import csv
from scipy.special import gdtr, gdtrix
from scipy.special import betainc
import scipy.stats
np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)  
shape_beta_RJ = 10.
print "\n\n             LiteRate - 20180201\n"

####### BEGIN FUNCTIONS for RJMCMC #######
def prior_sym_beta(x,a): 
	return scipy.stats.beta.logpdf(x, a,a)

def random_choice(vector):
	ind = np.random.choice(range(len(vector)))
	return [vector[ind], ind]

def add_shift_RJ_weighted_mean(rates,times): # args: vector of rates and vector of times (t_origin, t_shifts, t_end)
	r_time, r_time_ind = random_choice(np.diff(times))
	delta_t_prime           = np.random.uniform(0,r_time)
	t_prime                 = times[r_time_ind] + delta_t_prime
	times_prime             = np.sort(np.array(list(times)+[t_prime]))[::-1]
	time_i1                 = times[r_time_ind]
	time_i2                 = times[r_time_ind+1]
	p1 = (time_i1-t_prime)/(time_i1-time_i2)
	p2 = (t_prime-time_i2)/(time_i1-time_i2)
	u = np.random.beta(shape_beta_RJ,shape_beta_RJ)  #np.random.random()
	rate_i                  = rates[r_time_ind]
	rates_prime1            = exp( log(rate_i)-p2*log((1-u)/u) )
	rates_prime2            = exp( log(rate_i)+p1*log((1-u)/u) )	
	rates_prime             = np.insert(rates,r_time_ind+1,rates_prime2)
	rates_prime[r_time_ind] = rates_prime1
	log_q_prob              = log(abs(r_time))-prior_sym_beta(u,shape_beta_RJ) # prob latent parameters: Gamma pdf
	Jacobian                = 2*log(rates_prime1+rates_prime2)-log(rate_i)
	# return new rate vector, new time vector, prod between hastings ratio and Jacobian
	return rates_prime,times_prime,log_q_prob+Jacobian

def remove_shift_RJ_weighted_mean(rates,times):
	random_indx = np.random.choice(range(1,len(times)-1))
	rm_shift_ind  = random_indx
	t_prime       = times[rm_shift_ind]
	time_i1       = times[rm_shift_ind-1]
	time_i2       = times[rm_shift_ind+1]
	dT            = abs(times[rm_shift_ind+1]-times[rm_shift_ind-1]) # if rm t_i: U[t_i-1, t_i+1]
	times_prime   = times[times != t_prime]	
	p1 = (time_i1-t_prime)/(time_i1-time_i2) # weights based on amount of time assigned to each rate
	p2 = (t_prime-time_i2)/(time_i1-time_i2)	
	rate_i1       = rates[rm_shift_ind-1] # remove rates from rate vector
	rate_i2       = rates[rm_shift_ind]
	rate_prime    = exp(p1 *log(rate_i1) + p2 *log(rate_i2)) # new rate based on wehgted mean of old rates	
	rm_rate       = rates[rm_shift_ind]
	rates_prime   = rates[rates != rm_rate]
	rates_prime[rm_shift_ind-1] = rate_prime	
	u             = 1./(1+rate_i2/rate_i1) # == rate_i1/(rate_i1+rate_i2)
	log_q_prob    = -log(dT)+prior_sym_beta(u,shape_beta_RJ) # log_q_prob_rm = 1/(log_q_prob_add)
	Jacobian      = log(rate_prime)-(2*log(rate_i1+rate_i2))
	# return new rate vector, new time vector, prod between hastings ratio and Jacobian
	return rates_prime,times_prime,log_q_prob+Jacobian
	
def RJMCMC(arg):
	# args = birth-rate vector (L), death rates (M), rate shifts for L and M 
	[L,M, timesL, timesM]=arg
	r=np.random.random(2)
	newL,newtimesL,log_q_probL = L,timesL,0
	newM,newtimesM,log_q_probM = M,timesM,0
	sample_shift_mu = 0.5
	min_allowed_n_rates = 1
	# update birth model with 50% chance
	if r[0]>sample_shift_mu:
		# ADD/REMOVE SHIFT LAMBDA
		if r[1]>0.5: 
			newL,newtimesL,log_q_probL = add_shift_RJ_weighted_mean(L,timesL)
		# if 1-rate model this won't do anything, keeping the frequency of add/remove equal
		elif len(L)> min_allowed_n_rates: # defined for the edgeShift model
			newL,newtimesL,log_q_probL = remove_shift_RJ_weighted_mean(L,timesL) 
		update_L = 1 # indicator (which par was changed by RJ)
	# update death model with 50% chance
	else:
		# ADD/REMOVE SHIFT MU
		if r[1]>0.5: 
			newM,newtimesM,log_q_probM = add_shift_RJ_weighted_mean(M,timesM)
		# if 1-rate model this won't do anything, keeping the frequency of add/remove equal
		elif len(M)> min_allowed_n_rates: # defined for the edgeShift model
			newM,newtimesM,log_q_probM = remove_shift_RJ_weighted_mean(M,timesM) 
		update_L = 0
	# return new rates (newL, newM), new times of rate shift (newtimesL/M), hastngs ratio times Jacobian
	return newL,newtimesL,newM,newtimesM,log_q_probL+log_q_probM,update_L

def get_post_rj_HP(xl,xm): # returns rate parameter for the Poisson distribution sampled directly from the posterior
	# Gamma hyper-prior on the rate parameter of the Poisson priors on number of rates
	G_shape_rjHP = 2. # 1.1
	G_rate_rjHP  = 1. # 0.1 # mode at 1
	n = 2 # sp, ex
	a = G_shape_rjHP + xl + xm
	b = G_rate_rjHP + n
	Poi_lambda_rjHP = np.random.gamma(a,1./b)
	#print "Mean Poi_lambda:", a/b
	return Poi_lambda_rjHP


####### LIKELIHOOD FUNCTIONS #######

def BD_partial_lik(arg):
	# time window defined by up and lo (max/min ages)
	# ts = times of speciation, te = times of extinction
	# par="l" calc likelihood for speciation
	[up,lo,rate,par]=arg
	# indexes of the species within time frame
	if par=="l": i_events=np.intersect1d((ts <= up).nonzero()[0], (ts > lo).nonzero()[0])
	else: i_events=np.intersect1d((te <= up).nonzero()[0], (te > lo).nonzero()[0])	
	# get total time lived (or tot branch length) within time window
	n_S = get_sp_in_frame_br_length(up,lo)
	# number of events (n. sp or ex events)
	no_events = len(i_events)
	lik= log(rate)*no_events -rate*sum(n_S)
	return lik

def get_BDlik(times,rates,par):
	lik =0
	for i in range(len(rates)):
		up = times[i]
		lo = times[i+1]
		rate = rates[i]
		lik += BD_partial_lik([up,lo,rate,par])
	return lik

# VECTORIZED LIK FUNCTIONS
def get_sp_in_frame_br_length(up,lo):
	# index species present in time frame
	n_all_inframe = np.intersect1d((ts >= lo).nonzero()[0], (te <= up).nonzero()[0])
	# tot br length within time frame
	n_t_ts,n_t_te=zeros(len(ts)),zeros(len(ts))
	n_t_ts[n_all_inframe]= ts[n_all_inframe]   # speciation events before time frame
	n_t_ts[(n_t_ts>up).nonzero()]=up           # for which length is accounted only from $up$ rather than from $ts$	
	n_t_te[n_all_inframe]= te[n_all_inframe]   # extinction events in time frame
	n_t_te[np.intersect1d((n_t_te<lo).nonzero()[0], n_all_inframe)]=lo     # for which length is accounted only until $lo$ rather than to $te$
	# vector of br lengths within time frame  #(scaled by rho)
	n_S=((n_t_ts[n_all_inframe]-n_t_te[n_all_inframe])) #*rhos[n_all_inframe])
	return n_S

def precompute_events(arg):
	[up,lo]=arg
	# indexes of the species within time frame
	L_events=np.intersect1d((ts <= up).nonzero()[0], (ts > lo).nonzero()[0])
	M_events=np.intersect1d((te <= up).nonzero()[0], (te > lo).nonzero()[0])	
	# get total time lived (or tot branch length) within time window
	n_S = get_sp_in_frame_br_length(up,lo)
	return len(L_events), len(M_events), sum(n_S)
	
def get_rate_index(times,n_bins):
	if len(times)==2: 
		ind =np.zeros(n_bins).astype(int)
	else:
		times = np.round(times+0)
		dT = abs(np.diff(times)).astype(int)
		#print dT, sum(dT), times
		ind = []
		[ ind.extend([i]*dT[i]) for i in range(len(times)-1) ]
		ind = np.array(ind)
		
		# alternative way to compute it
		#shifts = times[1:-1]
		#h = np.histogram(shifts,bins =rev_bins)[0][::-1]
		#ind = np.cumsum(h)
	return ind

def vect_lik(L_acc_vec,M_acc_vec,sp_events_bin,ex_events_bin,br_length_bin):
	# BD likelihood
	try:
		Blik = sum(log(L_acc_vec)*sp_events_bin - L_acc_vec*br_length_bin) 
		Dlik = sum(log(M_acc_vec)*ex_events_bin - M_acc_vec*br_length_bin) 
	except:
		print len(L_acc_vec),len(M_acc_vec),len(sp_events_bin)
		sys.exit()
	return sum(Blik)+sum(Dlik)

# TRAIT DEPENDENT LIK FUNCTIONS
def get_likelihood_continuous_trait(l0,m0,alphaL,alphaM,tranf_rate_func):
	lik1 = sum(log(tranf_rate_func(l0,alphaL,tr_birth_events)))
	lik2 = -sum(tranf_rate_func(l0,alphaL,tr_waiting_times))
	lik3 = sum(log(tranf_rate_func(m0,alphaM,tr_death_events)))
	lik4 = -sum(tranf_rate_func(m0,alphaM,tr_waiting_times))
	#print lik1,lik2,lik3,lik4,l0,m0,alphaL,alphaM
	return lik1+lik2+lik3+lik4

def get_likelihood_continuous_trait_vect(l0,m0,alphaL,alphaM,tranf_rate_func,list_indx):
	[ind_l2,ind_l1,ind_m2,ind_m1] = list_indx
	lik1 = sum(log(tranf_rate_func(l0[ind_l1],alphaL,tr_birth_events)))
	lik2 = -sum(tranf_rate_func(l0[ind_l2],alphaL,tr_waiting_times))
	lik3 = sum(log(tranf_rate_func(m0[ind_m1],alphaM,tr_death_events)))
	lik4 = -sum(tranf_rate_func(m0[ind_m2],alphaM,tr_waiting_times))
	#print lik1,lik2,lik3,lik4,l0,m0,alphaL,alphaM
	return lik1+lik2+lik3+lik4

####### TRAIT-DEPENDENT FUNCTIONS ######## 
def transform_rate_logistic(r0,prm,trait):
	# r0 is the max rate
	x0,k = prm # mid point and steepness
	rate_at_trait = r0 / ( 1. + exp( -k * (trait-x0) )    )
	return rate_at_trait

def map_trait_time(ts,te,trait_list):
	tm_waiting_times =list()
	tm_birth_events  =list()
	tm_death_events  =list()
	for i in range(len(ts)):
		tm_waiting_times += range(te[i],ts[i]+1)
		tm_birth_events  += [ts[i]]
		if te[i]> 0: tm_death_events  += [te[i]] # only add if taxon is extinct
	
	tm_waiting_times = np.array(tm_waiting_times)	
	tm_birth_events  = np.array(tm_birth_events )	
	tm_death_events  = np.array(tm_death_events )	
		
	return tm_waiting_times,tm_birth_events,tm_death_events

def get_rate_index_trait(times,tm_events):
	
	indx_tm_waiting_times = np.zeros(len(tm_waiting_times)).astype(int)
	indx_tm_events  = np.zeros(len(tm_events)).astype(int)
	
	for i in range(2,len(times)):
		t0= times[i-1]
		t1= times[i]
		indx = i
		indx_tm_waiting_times[tm_waiting_times>t0]= indx_tm_waiting_times[tm_waiting_times>t0]+1
		indx_tm_events[tm_events>t0]  = indx_tm_events[tm_events>t0]  +1
	
	return np.array(indx_tm_waiting_times),np.array(indx_tm_events)
	
####### PROPOSALS #######
def update_multiplier_freq(q,d=1.1,f=0.75):
	S=np.shape(q)
	ff=np.random.binomial(1,f,S)
	u = np.random.uniform(0,1,S)
	l = 2*log(d)
	m = exp(l*(u-.5))
	m[ff==0] = 1.
	# new vector of rates
 	new_q = q * m
	# Hastings ratio
	U=sum(log(m))
	return new_q,U

def update_multiplier(q,d=1.1):
	u = np.random.uniform(0,1)
	l = 2*log(d)
	m = exp(l*(u-.5))
	# new vector of rates
 	new_q = q * m
	# Hastings ratio
	U=sum(log(m))
	return new_q,U

def update_sliding_win(i, d = 1): 
	# boundaries within which we can have a rate shift
	m, M = min_time, max_time
	ii = i+(np.random.random()-.5)*d
	if ii<m: ii=(ii-m)+m
	if ii>M: ii=(M-(ii-M))
	if ii<m: ii=i
	else: ii=i
	return ii

def update_sliding_win_unbounded(i, d = 1): 
	ii = i+(np.random.random()-.5)*d
	return ii

def update_sliding_win_unbounded_vec(i, d = 1): 
	ii = i+(np.random.random(len(i))-.5)*d
	return ii

def update_times(times):
	rS= times+0.
	indx = np.random.choice(range(1,len(times)-1))
	rS[indx] = update_sliding_win(rS[indx])
	#
	#for i in : 
	#	rS[i]=update_parameter(times[i])	
	return np.sort(rS)[::-1]

####### PRIORS #######
def Poisson_prior(k,rate):
	return k*log(rate) - rate - sum(log(np.arange(1,k+1)))

def prior_gamma(L,a=2,b=2):  
	return sum(scipy.stats.gamma.logpdf(L, a, scale=1./b,loc=0))

def prior_normal(L,sd): 
	#print L, sd, scipy.stats.norm.logpdf(L,loc=0,scale=sd)
	return sum(scipy.stats.norm.logpdf(L,loc=0,scale=sd))

def prior_exponential(L,rate): 
	return sum(scipy.stats.expon.logpdf(L, scale=1./rate))

def get_rate_HP(l,m): 
	rates = np.array(list(l)+list(m))
	post_rate_prm = np.random.gamma( shape=hpGamma_shape+Gamma_shape*len(rates), scale=1./(hpGamma_rate+sum(rates)) )
	return post_rate_prm

####### MCMC looop #######
def runMCMC(arg):
	# initial values of rates, times, correlations
	[L_acc, M_acc, timesLA, timesMA]  = arg
	Poi_lambda_rjHP = 1
	Gamma_rate = 1
	alphaLA,alphaMA = np.zeros(2),np.zeros(2)
	
	# init priors
	priorA = prior_gamma(L_acc) + prior_gamma(M_acc)
	priorA += -log(max_time-min_time)*(len(L_acc)-1+len(M_acc)-1)
	priorPoiA = Poisson_prior(len(L_acc),Poi_lambda_rjHP)+Poisson_prior(len(M_acc),Poi_lambda_rjHP)
	priorA += priorPoiA
	
	# conpute index arrays for baseline rates
	indx_tm_birth_waiting_timesA,indx_tm_birth_eventsA = get_rate_index_trait(timesLA,tm_birth_events)
	indx_tm_death_waiting_timesA,indx_tm_death_eventsA = get_rate_index_trait(timesMA,tm_death_events)
	
	# init likelihood
	list_indexes = [indx_tm_birth_waiting_timesA,indx_tm_birth_eventsA,indx_tm_death_waiting_timesA,indx_tm_death_eventsA]
	likA = get_likelihood_continuous_trait_vect(L_acc,M_acc,alphaLA,alphaMA,tranform_rate_func,list_indexes)
	
	# check likelihood LiteRate
	# sp_events_bin = []
	# ex_events_bin = []
	# br_length_bin = []
	# bins = np.arange(min_time,max_time+1)[::-1]
	# for i in range(len(bins)-1):
	# 	a,b,c = precompute_events([bins[i],bins[i+1]])
	# 	sp_events_bin.append(a)
	# 	ex_events_bin.append(b)
	# 	br_length_bin.append(c)
      # 
	# sp_events_bin = np.array(sp_events_bin)
	# ex_events_bin = np.array(ex_events_bin)
	# br_length_bin = np.array(br_length_bin)
	# n_bins = len(sp_events_bin)
	# 
	# # remove first bin
	# sp_events_bin = sp_events_bin[1:]
	# ex_events_bin = ex_events_bin[1:]
	# br_length_bin = br_length_bin[1:]
	# #max_time = max_time-1
	# indLA = get_rate_index(timesLA,n_bins)
	# indMA = get_rate_index(timesMA,n_bins)
	# L_acc,M_acc = L_acc/2.,M_acc/2.
	# likA_no_trait = sum(vect_lik(L_acc[indLA],M_acc[indMA],sp_events_bin,ex_events_bin,br_length_bin))
	# print "WEGFDS", likA, likA_no_trait
	# quit()
	
	
	
	
	
	iteration = 0
	while iteration < n_iterations:		
		r = np.random.random(3)
		L,timesL = L_acc+0,timesLA+0
		M,timesM = M_acc+0,timesMA+0
		alphaL,alphaM = alphaLA+0,alphaMA+0
		indx_tm_birth_waiting_times,indx_tm_birth_events = indx_tm_birth_waiting_timesA+0,indx_tm_birth_eventsA+0
		indx_tm_death_waiting_times,indx_tm_death_events = indx_tm_death_waiting_timesA+0,indx_tm_death_eventsA+0
		hasting = 0
		gibbs=0
		priorPoi = 0
		if r[0]< 0.25:
			# update birth part
			if r[1] < .5 or len(L_acc)==1:
				# update rates
				L, hasting = update_multiplier_freq(L_acc)
			else:
				# update times (hastings = 0 because we are doing symmetric update)
				timesL = update_times(timesLA)
				indx_tm_birth_waiting_times,indx_tm_birth_events = get_rate_index_trait(timesL,tm_birth_events)
			
		elif r[0] < 0.5:
			# update M 
			if r[1] < .5 or len(M_acc)==1:
				# update rates
				M, hasting = update_multiplier_freq(M_acc)
			else:
				# update times (hastings = 0 because we are doing symmetric update)
				timesM = update_times(timesMA)
				indx_tm_death_waiting_times,indx_tm_death_events = get_rate_index_trait(timesM,tm_death_events)
			
		elif r[0] < 0.8 and args.model != 0:
			d_win = np.array([0.5,0.1])
			if r[2] < .5:
				alphaL= update_sliding_win_unbounded_vec(alphaLA,d=d_win)
			else:
				alphaM= update_sliding_win_unbounded_vec(alphaMA,d=d_win)
			
		elif r[0] < 0.99:
			# do RJ
			L,timesL, M,timesM, hasting, update_L = RJMCMC([L_acc,M_acc, timesLA, timesMA])
			if update_L==1: 
				indx_tm_birth_waiting_times,indx_tm_birth_events = get_rate_index_trait(timesL,tm_birth_events)
			else: 
				indx_tm_death_waiting_times,indx_tm_death_events = get_rate_index_trait(timesM,tm_death_events)
			priorPoi = Poisson_prior(len(L),Poi_lambda_rjHP)+Poisson_prior(len(M),Poi_lambda_rjHP)
			
		else: 
			# update HPs 
			Poi_lambda_rjHP = get_post_rj_HP(len(L_acc),len(M_acc))
			Gamma_rate = get_rate_HP(L_acc,M_acc)
			gibbs=1
		
		# prevent super small time frames
		if min(abs(np.diff(timesL)))<=min_allowed_t or min(abs(np.diff(timesM)))<=min_allowed_t: 
			prior = -np.inf	
			lik =  -np.inf			
		else:
			# calc acceptance ratio
			# prior on rate
			prior = prior_gamma(L,Gamma_shape,Gamma_rate) + prior_gamma(M,Gamma_shape,Gamma_rate)
			# prior on times of rate shift
			prior += -log(max_time-min_time)*(len(L)-1+len(M)-1)
			# priors on logistic parameters
			prior += prior_normal(alphaL[1],1)+prior_normal(alphaM[1],1)
			if min([alphaL[0],alphaM[0]]) < allowed_x0_range[0] or max([alphaL[0],alphaM[0]]) > allowed_x0_range[1]:
				prior = -np.inf
			# prior on number of shifts
			if priorPoi != 0: 
				prior += priorPoi
			else: 
				prior += priorPoiA
				priorPoi = priorPoiA
			if gibbs==0:
				list_indexes = [indx_tm_birth_waiting_times,indx_tm_birth_events,indx_tm_death_waiting_times,indx_tm_death_events]
				lik = get_likelihood_continuous_trait_vect(L,M,alphaL,alphaM,tranform_rate_func,list_indexes)
			else: 
				lik = likA
		
		# debug
		if check_lik==1:
			lik_old = get_BDlik(np.floor(timesL),L,"l") + get_BDlik(np.floor(timesM),M,"m")
			if iteration % 100==0: 
				print lik_old-lik 
		
		
		# print lik, likA, prior, priorA
		
		if lik-likA + prior-priorA + hasting >= log(np.random.random()) or gibbs==1:
			# update accepted values to proposed ones
			L_acc, M_acc, timesLA, timesMA = L,M,timesL, timesM
			# update lik, prior
			likA,priorA = lik, prior
			priorPoiA = priorPoi
			alphaLA,alphaMA= alphaL,alphaM
			indx_tm_birth_waiting_timesA,indx_tm_birth_eventsA = indx_tm_birth_waiting_times,indx_tm_birth_events
			indx_tm_death_waiting_timesA,indx_tm_death_eventsA = indx_tm_death_waiting_times,indx_tm_death_events
		
		if iteration % s_freq ==0:
			# MCMC log
			log_state = map(str,[iteration,likA+priorA,likA,priorA,mean(L_acc),mean(M_acc),len(L_acc),len(M_acc),alphaLA[0],alphaLA[1],alphaMA[0],alphaMA[1],max_time,min_time,Gamma_rate,Poi_lambda_rjHP])
			mcmc_logfile.write('\t'.join(log_state)+'\n')
			mcmc_logfile.flush()
			# log marginal rates/times
			log_state = map(str,list(L_acc) + list(timesLA[1:len(timesLA)-1]))
			sp_logfile.write('\t'.join(log_state)+'\n')
			sp_logfile.flush()
			log_state = map(str,list(M_acc) + list(timesMA[1:len(timesMA)-1]))
			ex_logfile.write('\t'.join(log_state)+'\n')
			ex_logfile.flush()
		
		if iteration % p_freq ==0:
			print iteration, likA, priorA
			# print on screen
			print "\tsp.times:", timesLA
			print "\tex.times:", timesMA
			print "\tsp.rates:", L_acc
			print "\tex.rates:", M_acc
			print "\tsp.alpha:", alphaLA
			print "\tex.alpha:", alphaMA
		
		iteration +=1 

####### parse arguments #######
p = argparse.ArgumentParser() #description='<input file>') 

p.add_argument('-v',       action='version', version='%(prog)s')
p.add_argument('-d',       type=str, help='data file', default="", metavar="") 
p.add_argument('-n',       type=int, help='n. MCMC iterations', default=10000000, metavar=10000000)
p.add_argument('-model',   type=int, help='0: no correlation, 1: logistic correlation', default=1, metavar=1)
p.add_argument('-p',       type=int, help='print frequency', default=1000, metavar=1000) 
p.add_argument('-s',       type=int, help='sampling frequency', default=1000, metavar=1000) 
p.add_argument('-seed',    type=int, help='seed (set to -1 to make it random)', default= 1, metavar= 1)
p.add_argument('-present_year',    type=int, help="""set to: -1 for standard pyrate datasets (time BP), \
0: time AD and present set to most recent TE, 1: time AD present user defined """, default= 0, metavar= 0)

args = p.parse_args()

if args.seed==-1:
	rseed=np.random.randint(0,9999)
else: rseed=args.seed	
random.seed(rseed)
np.random.seed(rseed)


n_iterations = args.n
s_freq = args.s
p_freq = args.p


####### Parse DATA #######
#f = args.d
f = "/Users/danielesilvestro/Software/LiteRate/example_dataTAD.txt"
t_file=np.loadtxt(f, skiprows=1)
ts_years = t_file[:,2].astype(int)
te_years = t_file[:,3].astype(int)

#te_years = np.round(np.random.uniform(1950,2017,1000)).astype(int)
#ts_years = np.round(np.random.uniform(1950,te_years,1000)).astype(int)
#


if args.present_year== -1: # to load regular pyrate input
	ts = ts_years
	te = te_years
elif args.present_year==0: # find max year and set to present
	ts = max(te_years) - ts_years 
	te = max(te_years) - te_years 
else: # user-spec present year
	ts = args.present_year - ts_years 
	te = args.present_year - te_years 

#ts,te = np.round(ts),np.round(te)
max_time = max(ts)
min_time = min(te)



species_durations = (ts+1)-te # consider year of origination as a lived year

trait_list_of_arrays  = []

tr_waiting_times =[]
tr_birth_events =[]
tr_death_events =[]

list_all_values = []
for i in species_durations:
	# make up some trait data
	species_trait_array = np.sort(np.random.normal(0,2,int(i)) )   # skewed values
	species_trait_array = np.sort(np.random.uniform(-5,5,int(i)) ) # severely skewed values
	#species_trait_array = np.random.uniform(-5,5,int(i))          # no trait effects	
	
	# precompute stuff
	list_all_values += list(species_trait_array)
	trait_list_of_arrays.append(species_trait_array)
	tr_waiting_times += list(species_trait_array) # all trait values
	tr_birth_events  += [species_trait_array[0]] # trait value at origination
	tr_death_events  += [species_trait_array[-1]] # trait value at extinction



	
# define correlation function
tranform_rate_func = transform_rate_logistic
delta_trait = 0.1 # the x0 parmater can only range between min/max trait values +/- 10%
allowed_x0_range = np.array([ min(tr_waiting_times)*(1-delta_trait), max(tr_waiting_times)*(1+delta_trait)  ])

#convert to array
tr_waiting_times = np.array(tr_waiting_times)
tr_birth_events  = np.array(tr_birth_events )
tr_death_events  = np.array(tr_death_events )[te>0]
tm_waiting_times,tm_birth_events,tm_death_events = map_trait_time(ts,te,species_trait_array)

print tm_birth_events[0:10], tm_death_events[0:10]


# init params
l0A=np.array([0.2,0.1])
m0A=np.array([0.1,0.1])
alphaLA=np.array([0,1.])
alphaMA=np.array([0,1.])
timesLA = np.array([max_time, 30.,min_time])
timesMA = np.array([max_time, 30.,min_time])

# conpute index arrays for baseline rates
indx_tm_birth_waiting_times,indx_tm_birth_events = get_rate_index_trait(timesLA,tm_birth_events)
indx_tm_death_waiting_times,indx_tm_death_events = get_rate_index_trait(timesMA,tm_death_events)

# check stuff
print len(tr_waiting_times),len(tm_waiting_times),len(indx_tm_birth_waiting_times),len(indx_tm_death_waiting_times)
print len(indx_tm_birth_events),len(indx_tm_death_events) #, m0A[indx_tm_death_events]
print get_likelihood_continuous_trait(l0A[0],m0A[0],alphaLA,alphaMA,tranform_rate_func)
list_indexes = [indx_tm_birth_waiting_times,indx_tm_birth_events,indx_tm_death_waiting_times,indx_tm_death_events]
print get_likelihood_continuous_trait_vect(l0A,m0A,alphaLA,alphaMA,tranform_rate_func,list_indexes)

out_dir= os.path.dirname(f)




# setup output files
print out_dir
if out_dir=="": 
	out_dir= os.getcwd()
file_name = os.path.splitext(os.path.basename(f))[0]

####### MCMC log files #######
out_dir = "%s/pyrate_mcmc_logs" % (out_dir)
try: os.mkdir(out_dir) 
except: pass

out_log = "%s/%s_trait_mcmc.log" % (out_dir, file_name)
mcmc_logfile = open(out_log , "w",0) 
mcmc_logfile.write('\t'.join(["it","posterior","likelihood","prior","lambda_avg","mu_avg","K_l","K_m","x0_l","kappa_l","x0_m","kappa_m","root_age","death_age","gamma_rate_hp","poisson_rate_hp"])+'\n')
out_log = "%s/%s_trait_sp_rates.log" % (out_dir, file_name)
sp_logfile = open(out_log , "w",0) 
out_log = "%s/%s_trait_ex_rates.log" % (out_dir, file_name)
ex_logfile = open(out_log , "w",0) 	

####### init parameters #######
L_acc= np.random.gamma(2,2,1)
M_acc= np.random.gamma(2,2,1)
timesLA = np.array([max_time, min_time])
timesMA = np.array([max_time, min_time])

####### GLOBAL variables #######
min_allowed_t = 1   # minimum allowed distance between shifts (to avoid numerical issues)
Gamma_shape = 2.    # shape parameter of Gamma prior on B/D rates
hpGamma_shape = 1.2 # shape par of Gamma hyperprior on rate of Gamma priors on B/D rates
hpGamma_rate =  0.1 # rate par of Gamma hyperprior on rate of Gamma priors on B/D rates

check_lik = 0 # debug (set to 1 to compare vectorized likelihood against 'traditional' one)
runMCMC([L_acc,M_acc,timesLA,timesMA])

quit()

