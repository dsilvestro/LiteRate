#!/usr/bin/env python 
import argparse,sys
import os, csv, glob
from numpy import *
import numpy as np
from scipy.special import gamma
from scipy.special import beta as f_beta
import random as rand
import platform, time
import csv
from scipy.special import gdtr, gdtrix
from scipy.special import betainc
import scipy.stats
from literate_library import *

np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)  
print("\n\n             LiteRate - 20200206\n")

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
	times_prime             = np.sort(np.array(list(times)+[t_prime]))
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

# VECTORIZED LIK FUNCTIONS
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
def update_multiplier_freq(q,d=1.1,f=0.75):
	S=np.shape(q)
	ff=np.random.binomial(1,f,S)
	u = np.random.uniform(0,1,S)
	l = 2*log(d)
	m = exp(l*(u-.5))
	m[ff==0] = 1.
	new_q = q * m
   # new vector of rates
	# Hastings ratio
	U=sum(log(m))
	return new_q,U

def update_sliding_win(i, d = 1): 
	# boundaries within which we can have a rate shift
	m, M = start_time, end_time
	ii = i+(np.random.random()-.5)*d
	if ii<m: ii=(ii-m)+m
	if ii>M: ii=(M-(ii-M))
	if ii<m: ii=i
	else: ii=i
	return ii

def update_times(times):
	rS= times+0.
	indx = np.random.choice(range(1,len(times)-1))
	rS[indx] = update_sliding_win(rS[indx])
	#
	#for i in : 
	#	rS[i]=update_parameter(times[i])	
	return np.sort(rS)

####### PRIORS #######
def Poisson_prior(k,rate):
	return k*log(rate) - rate - sum(log(np.arange(1,k+1)))

def prior_gamma(L,a=2,b=2):  
	return sum(scipy.stats.gamma.logpdf(L, a, scale=1./b,loc=0))

def prior_normal(L,sd): 
	return scipy.stats.norm.logpdf(L,loc=0,scale=sd)

def prior_exponential(L,rate): 
	return sum(scipy.stats.expon.logpdf(L, scale=1./rate))

def get_rate_HP(rate): 
	rates = np.array(list(rate))
	post_rate_prm = np.random.gamma( shape=hpGamma_shape+Gamma_shape*len(rates), scale=1./(hpGamma_rate+sum(rates)) )
	return post_rate_prm

####### MCMC looop #######
def runMCMC(arg):
	# initial values of rates, and times
	[L_acc, M_acc, timesLA, timesMA]  = arg
	
	if Poisson_HP==0: Poi_lambda_rjHP = 1
	else: Poi_lambda_rjHP = Poisson_HP
	Gamma_rate = [1.,1.]
	# init lik
	indLA = get_rate_index(timesLA)
	indMA = get_rate_index(timesMA)
	likA = calc_likelihood(L_acc[indLA],M_acc[indMA])
	priorA = prior_gamma(L_acc) + prior_gamma(M_acc)
	priorA += -log(end_time-start_time)*(len(L_acc)-1+len(M_acc)-1)
	priorPoiA = Poisson_prior(len(L_acc),Poi_lambda_rjHP)+Poisson_prior(len(M_acc),Poi_lambda_rjHP)
	priorA += priorPoiA
	
	iteration = 0
	while iteration < n_iterations:		
		r = np.random.random(2)
		L,timesL = L_acc+0,timesLA+0
		M,timesM = M_acc+0,timesMA+0
		indL =indLA
		indM =indMA
		hasting = 0
		gibbs=0
		priorPoi = 0
		if r[0]< 0.4:
			# update birth part
			if r[1] < .5 or len(L_acc)==1:
				# update rates
				L, hasting = update_multiplier_freq(L_acc)
			else:
				# update times (hastings = 0 because we are doing symmetric update)
				timesL = update_times(timesLA)
				indL = get_rate_index(np.floor(timesL))
			
		elif r[0] < 0.8:
			# update M 
			if r[1] < .5 or len(M_acc)==1:
				# update rates
				M, hasting = update_multiplier_freq(M_acc)
			else:
				# update times (hastings = 0 because we are doing symmetric update)
				timesM = update_times(timesMA)
				indM = get_rate_index(np.floor(timesM))
			
		elif r[0] < 0.99 and const_rates==0:
			# do RJ
			L,timesL, M,timesM, hasting, update_L = RJMCMC([L_acc,M_acc, timesLA, timesMA])
			if update_L==1: indL = get_rate_index(np.floor(timesL))
			else: indM = get_rate_index(np.floor(timesM))
			priorPoi = Poisson_prior(len(L),Poi_lambda_rjHP)+Poisson_prior(len(M),Poi_lambda_rjHP)
			
		else: 
			# update HPs 
			if Poisson_HP==0:
				Poi_lambda_rjHP = get_post_rj_HP(len(L_acc),len(M_acc))
			if use_rate_HP:
				Gamma_rate = [get_rate_HP(L_acc),get_rate_HP(M_acc)]
			gibbs=1
		
		# prevent super small time frames
		if min(abs(np.diff(timesL)))<=min_allowed_t or min(abs(np.diff(timesM)))<=min_allowed_t: 
			prior = -np.inf	
			lik =  -np.inf			
		else:
			# calc acceptance ratio
			# prior on rate
			prior = prior_gamma(L,Gamma_shape,Gamma_rate[0]) + prior_gamma(M,Gamma_shape,Gamma_rate[1])
			# prior on times of rate shift
			prior += -log(end_time-start_time)*(len(L)-1+len(M)-1)
			# prior on 
			if priorPoi != 0: 
				prior += priorPoi
			else: 
				prior += priorPoiA
				priorPoi = priorPoiA
			if gibbs==0:
				lik = calc_likelihood(L[indL],M[indM])
			else: 
				lik = likA
				
		
		# print lik, likA, prior, priorA
		
		if lik-likA + prior-priorA + hasting >= log(np.random.random()) or gibbs==1:
			# update accepted values to proposed ones
			L_acc, M_acc, timesLA, timesMA = L,M,timesL, timesM
			# update lik, prior
			likA,priorA = lik, prior
			indLA,indMA = indL, indM
			priorPoiA = priorPoi
		
		if iteration % s_freq ==0:
			# MCMC log
			#compute adequacy stats
			adequacy=calculate_r_squared(B_EMP,D_EMP,L_acc[indLA],M_acc[indMA])
			log_state = map(str,[iteration,likA+priorA,likA,priorA,mean(L_acc),mean(M_acc),len(L_acc),len(M_acc),start_time,end_time] + Gamma_rate + [Poi_lambda_rjHP] + list(adequacy) )
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
			print(iteration, likA, priorA)
			# print on screen
			print("\tsp.times:", timesLA)
			print("\tex.times:", timesMA)
			print("\tsp.rates:", L_acc)
			print("\tex.rates:", M_acc)
			print("\tR^2:",adequacy[1])

		
		iteration +=1 

####### parse arguments #######
p = argparse.ArgumentParser() #description='<input file>') 

p.add_argument('-v',       action='version', version='%(prog)s')
p.add_argument('-d',       type=str, help='data file', default="", metavar="") 
p.add_argument('-n',       type=int, help='n. MCMC iterations', default=10000000, metavar=10000000)
#p.add_argument('-p',       type=int, help='print frequency', default=1000, metavar=1000) 
p.add_argument('-s',       type=int, help='sampling frequency', default=1000, metavar=1000) 
p.add_argument('-seed',    type=int, help='seed (set to -1 to make it random)', default= -1, metavar= -1)
#p.add_argument('-present_year',    type=int, help="""set to: -1 for standard pyrate datasets (time BP), \
#0: time AD and present set to most recent TE, 1: time AD present user defined """, default= 0, metavar= 0)
p.add_argument('-const_rates',    type=int, help="set to: 1 for constant B/I and D rates" , default= 0, metavar= 0)
p.add_argument('-model_BDI',      type=int, help='0: birth-death; 1: immigration-death; 2 birth-death (Keiding likelihood)', default= 0, metavar= 0)
p.add_argument('-TBP', help='Default is AD. Include for TBP.', default=False, action='store_true')
p.add_argument('-first_year',    type=int, help='This is a convenience function if you would like to specify a different start to your dataset. Unspecified for TBP.', default= -1, metavar= -1)
p.add_argument('-last_year',    type=int, help='This is a convenience function if you would like to specify a different end to your dataset. Unspecified for TBP.', default= -1, metavar= -1)
p.add_argument('-death_jitter', type=float, help="""Determines the amount to jitter death times.\
               If set to 0, lineages that lived and died in same time bin will be excluded from branch length.""", default= .5, metavar= .5)
p.add_argument('-use_rate_HP',    type=int, help='0: no hyper-prior on rates, 1: hyper-prior on rates', default= 1, metavar= 1)
p.add_argument('-Poisson_prior',  type=float, help='0: use hyper-prior on n. shifts, >0:  fixed prior on n. shifts', default= 0, metavar= 0)
p.add_argument('-rm_first_bin',   type=float, help='if set to 1 it removes the first time bin (if max time is not the origin)', default= 0, metavar= 0)

args = p.parse_args()

if args.seed==-1:
	rseed=np.random.randint(0,9999)
else: rseed=args.seed	
random.seed(rseed)
np.random.seed(rseed)


n_iterations = args.n
s_freq = args.s
p_freq = args.s
TBP=args.TBP

model_BDI = args.model_BDI
if model_BDI==0: out_name = "_BD"
if model_BDI==1: out_name = "_ID"
if model_BDI==2: out_name = "_BDk"

if model_BDI<=1: calc_likelihood = BDI_partial_lik # define likelihood function
elif model_BDI==2: calc_likelihood = BD_lik_Keiding

use_rate_HP = args.use_rate_HP
Poisson_HP = args.Poisson_prior # if 0 use HP, else fixed Poi 
rm_first_bin = args.rm_first_bin
const_rates  = args.const_rates
####### Parse DATA #######
f = args.d
t_file=np.genfromtxt(f, skip_header=1)
ts_years = t_file[:,1]
te_years = t_file[:,2]
if TBP==True:
    ts= max(ts_years)-ts_years
    te= max(ts_years)- te_years
else:
	if args.first_year!=-1:
		ts_years=ts_years[ts_years>=args.first_year]
		te_years=te_years[ts_years>=args.first_year]
	if args.last_year!=-1:
		ts=ts_years[ts_years<=args.last_year]
		te=te_years[ts_years<=args.last_year]
		te[te>args.last_year]=args.last_year
	else:
		ts = ts_years
		te = te_years

te = te + args.death_jitter


start_time = min(ts)
end_time = max(te)
bins = np.arange(start_time,end_time+1)


out_dir= os.path.dirname(f)

if out_dir=="": 
	out_dir= os.getcwd()
file_name = os.path.splitext(os.path.basename(f))[0]

####### MCMC log files #######
out_dir = "%s/literate_mcmc_logs" % (out_dir)
try:
	os.mkdir(out_dir) 
except OSError as e:
	print(e)
	pass

out_log = "%s/%s%s_mcmc.log" % (out_dir, file_name,out_name)
mcmc_logfile = open(out_log , "w") 
mcmc_logfile.write('\t'.join(["it","posterior","likelihood","prior","lambda_avg","mu_avg",\
"K_l","K_m","root_age","death_age","gamma_rate_hp_BI","gamma_rate_hp_D","poisson_rate_hp",\
"corr_coeff","rsquared","gelman_r2"
])+'\n')
out_log = "%s/%s%s_sp_rates.log" % (out_dir, file_name, out_name )
sp_logfile = open(out_log , "w") 
out_log = "%s/%s%s_ex_rates.log" % (out_dir, file_name, out_name )
ex_logfile = open(out_log , "w") 
out_log = "%s/%s%s_div.log" % (out_dir, file_name,out_name)
div_logfile = open(out_log , "w") 

####### PRECOMPUTE VECTORS #######
sp_events_bin = []
ex_events_bin = []
br_length_bin = []

for i in range(int(np.min(ts)),int(np.max(te))):	
	a,b,c =  precompute_events([i,i+1])
	sp_events_bin.append(a)
	ex_events_bin.append(b)
	br_length_bin.append(c)
	#print i, i+1, b

if rm_first_bin:
	# remove first bin
	sp_events_bin = sp_events_bin[1:]
	ex_events_bin = ex_events_bin[1:]
	br_length_bin = br_length_bin[1:]

#write diversity. plotRj uses this
div_rows = zip(sp_events_bin,ex_events_bin,br_length_bin)
div_logfile.write('sp_events\tex_events\tbr_length\n')
writer = csv.writer(div_logfile,delimiter='\t')
for row in div_rows: writer.writerow(row)


sp_events_bin = np.array(sp_events_bin)
ex_events_bin = np.array(ex_events_bin)
br_length_bin = np.array(br_length_bin)

n_bins = len(sp_events_bin)
Tk = np.ones(n_bins) # time spent in state

B_EMP,D_EMP=print_empirical_rates(sp_events_bin,ex_events_bin,br_length_bin)

####### init parameters #######
L_acc= np.random.gamma(2,2,1)
M_acc= np.random.gamma(2,2,1)
timesLA = np.array([start_time, end_time])
timesMA = np.array([start_time, end_time])

####### GLOBAL variables #######
shape_beta_RJ = 10.
min_allowed_t = 1   # minimum allowed distance between shifts (to avoid numerical issues)
Gamma_shape = 2.    # shape parameter of Gamma prior on B/D rates
hpGamma_shape = 1.2 # shape par of Gamma hyperprior on rate of Gamma priors on B/D rates
hpGamma_rate =  0.1 # rate par of Gamma hyperprior on rate of Gamma priors on B/D rates

check_lik = 0 # debug (set to 1 to compare vectorized likelihood against 'traditional' one)
runMCMC([L_acc,M_acc,timesLA,timesMA])
