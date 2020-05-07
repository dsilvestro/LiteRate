#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 22:24:32 2018

@author: bernie
"""

#!/usr/bin/env python 
# Created by Daniele Silvestro on 20/01/2017 => daniele.silvestro@bioenv.gu.se
from numpy import *
import numpy as np
import sys, os
from scipy.special import gamma as gamma_func
from scipy.stats import weibull_min as weibull
from scipy.stats import uniform
from scipy.stats import gamma
from scipy.stats import beta
from matplotlib.pyplot import plot as plt
from collections import OrderedDict
import multiprocessing as mp

print("Birth-Death Sampler 18\n")


##########################################################################
###########                 SIMULATION SETTINGS                 ##########
##########################################################################

class Simulator(object):
	
	def __init__(self):
		
		###SIMULATION SETTINGS###
		
		self.n_reps = 20 # number of simulations
		self.t_as_AD=True
		self.present_year=2000
		# CONSTRAINTS on DATA SIZE (simulations will run until size requirements are met)
		self.s_species=1   # number of starting species
		self.minSP=200     # min size data set
		self.maxSP=1000     # max size data set
		self.minEX_SP=500    # minimum number of extinct lineages allowed
		
		
		self.scale=100.0 #this parameter is used to stretch out the simulation, approximating continuous time
		self.root_age = -40.0	# amount of time to simulate. specified in TBP
		
		#BIRTH LOGISTIC GROWTH PARAMETERS
		self.b_logistic_params={
		'l0':.4,
		'K':self.maxSP, #max carrying cappacity
		'A0':5, #min carrying capacity
		'B':.5, #growth rate
		'x0':self.root_age/2.0, #x value at midpoint
		'v':1.0 #skew where growth is occuring the most greater than 1 skewed towards beginning, less than 1 skewed towards end.
		#Must be greater than 0
		}
		
		#DEATH LOGISTIC GROWTH PARAMS
		self.d_logistic_params={
		'm0':.2, # this is used in
		'K':1.0, #max % of baseline death rate
		'A0':0.6, #min % of baseline death rate
		'B':.5, #growth rate
		'x0':self.root_age/2.0, #x value at midpoint
		'v':2.0 #skew where growth is occuring the most greater than 1 skewed towards beginning, less than 1 skewed towards end.
		#Must be greater than 0
		}
		
		#AGE-DEPENDENT EXTINCTION
		self.mean_lifespan=2.7
		self.w_shape=0.5
		self.w_scale = self.mean_lifespan/ gamma_func(1+1/self.w_shape)

	def __get_K_t(self,t,switch): #get carrying capacity at time t from logistic function
		#THIS IS A GENERALIZED LOGISTIC FUNCTION
		if switch == 'b': p=self.b_logistic_params
		else: p=self.d_logistic_params
		t=t/self.scale
		return p['A0'] + ( (p['K'] - p['A0']) /(1 + exp(-p['B']*(t-p['x0'] ) ) )**(1/p['v']) )  #the scale * -1 part is not part of the equation
	
	def simulate(self,b_params=None,d_params=None):
		if b_params!=None: self.b_logistic_params=b_params
		if d_params!=None: self.d_logistic_params=d_params
		
		LOtrue=[0]
		n_extinct=-0
		tries=0
		eb_rates=[]
		ed_rates=[]
		while len(LOtrue) < self.minSP or len(LOtrue) > self.maxSP or n_extinct < self.minEX_SP:

			if tries>100: return None,None 
			
			ts=repeat(self.root_age,self.s_species)
			te=repeat(0,self.s_species)
			m=repeat(max(0.0,self.b_logistic_params['l0']-(self.b_logistic_params['l0']-self.d_logistic_params['m0'])*self.s_species/self.b_logistic_params['A0']),self.s_species) #death rate for those born at time 0
			#print("PREM",m)			
			for t in range(int(self.root_age*self.scale),0):
				D_t=len(te[te==0])*1.0
				K_t = self.__get_K_t(t,'b')
				l_t = max(0.0,self.b_logistic_params['l0']-(self.b_logistic_params['l0']-self.d_logistic_params['m0'])*D_t/K_t)
				m_t = self.d_logistic_params['m0'] * self.__get_K_t(t,'d')
				#print("TEST",t,D_t,K_t,l_t,m_t,len(te))
				#print(t,self.__get_K_t(t,'d'))
				TE=len(te)
				if TE>self.maxSP:
					#print("GOT HERE")
					break
				#print(t,self.__get_K_t(t,'b'))		
				ran_vec=np.random.random(TE) #this is between 0 and 100 because we've scaled down by 100
				#print(ran_vec)
				#print("LT",t,l_t,ran_vec[0])
				r_sp_indx = (ran_vec < l_t).nonzero()[0]
				r_ex_indx = np.intersect1d((ran_vec > l_t).nonzero()[0], (ran_vec < l_t+m).nonzero()[0])
				
				#print("BIRTHS/DEATHS",len(r_sp_indx),len(r_ex_indx))
				te[r_ex_indx]=t #kill em all	
				#birth em
				ts=append(ts,repeat(t,len(r_sp_indx)))
				m=append(m,repeat(m_t,len(r_sp_indx)))
				te=append(te,repeat(0,len(r_sp_indx)))

				eb_rates.append(len(r_sp_indx)/max(D_t,.0001))
				ed_rates.append(len(r_ex_indx)/max(D_t,.0001))

				
			LOtrue=te
			n_extinct = len(te[te<0])
			#print(self.minSP,self.minEX_SP,self.maxSP)
			#print("LIVED",len(LOtrue),"EXTINCT",n_extinct)
			tries+=1
			#print(LOtrue)
		ts, te= rints(-array(ts)/self.scale), rint(-(te)/self.scale)
		if self.t_as_AD: ts, te = self.present_year-ts, self.present_year-te
		#return ts, te
		#reduce down to years
		eb_rates= eb_rates[:(-1*(len(eb_rates)%100))] #truncate to multiple of 100
		ed_rates= ed_rates[:(-1*(len(ed_rates)%100))]
		eb_rates = np.mean(np.array(eb_rates).reshape(-1, int(100)), axis=1) #this averages every 100 times to get 
		ed_rates = np.mean(np.array(ed_rates).reshape(-1, int(100)), axis=1)
		
		return eb_rates, ed_rates

test=Simulator()
crap=test.simulate()

class ABCMachine(object):
	
	def __init__(self,eb_rates,ex_rates,outfile):
		#print(eb_rates,ex_rates,outfile)
		self.nreps=100000
		self.eb_rates=eb_rates
		self.ex_rates=ex_rates
		self.threshold=.9
		self.seed=1337
		
		self.outfile=open(outfile,'w')
		self.outfile.write('l0\bK\tbA0\tbB\tx0\tv\tm0\tdK\tdA0\tdB\tdx0\tdv\n')
		   #BIRTH PRIORS
		self.b_priors=OrderedDict()
		self.b_priors['A0']=5 #min carrying capacity
		self.b_priors['K']=uniform(0,3000) #max carrying cappacity
		self.b_priors['B']=gamma(2,2) #growth rate
		self.b_priors['x0']=uniform(-1*(len(eb_rates)/2),(len(eb_rates)/2)) #x value at midpoint
		self.b_priors['v']=gamma(2,2) #skew where growth is occuring the most greater than 1 skewed towards beginning, less than 1 skewed towards end.
		#Must be greater than 0
		self.b_priors['l0']=gamma(1.5,3)
		
		#DEATH PRIORS
		self.d_priors=OrderedDict()
		self.d_priors['K']=1.0 #max % of baseline death rate
		self.d_priors['A0']=beta(2,5) #min % of baseline death rate
		self.d_priors['B']=gamma(2,2) #growth rate
		self.d_priors['x0']=uniform(-1*(len(eb_rates)/2),(len(eb_rates)/2)) #x value at midpoint
		self.d_priors['v']=gamma(2,2) #skew where growth is occuring the most greater than 1 skewed towards beginning, less than 1 skewed towards end.
		self.d_priors['m0']=gamma(2,2)
		#Must be greater than 0

		
		
		self.b_samples=OrderedDict()
		#initial params
		for p in self.b_priors.keys():
			if type(self.b_priors[p]) not in [int, float]:
				self.b_samples[p]=self.b_priors[p].rvs()
			else: self.b_samples[p]=self.b_priors[p]
		
		self.d_samples=OrderedDict()
		#initial params
		for p in self.d_priors.keys():
			if type(self.d_priors[p]) not in [int, float]:
				self.d_samples[p]=self.d_priors[p].rvs()
			else: self.d_samples[p]=self.d_priors[p]
	
	def __sample_prior(self,ltype,param='random'):
		p=param
		if ltype == 'b': 		#any prior except param 1 which is fixed for now
			print("KEYS",self.b_priors[random.choice(self.b_priors.keys()[1:])].rvs())
			if param == 'random': p = random.choice(self.b_priors.keys()[1:]) 
			self.b_samples[p]=self.b_priors[p].rvs()
		else:
			if param == 'random': p = random.choice(self.d_priors.keys()[1:]) 
			self.d_samples[p]=self.d_priors[p].rvs()
			
	def __summary_statistic(self,s_brates,s_xrates):
		e_brates=self.e_brates[:len(s_brates)] #truncate to as long as the other one got. May have to devise some penalty.
		e_xrates=self.e_xrates[:len(s_xrates)] #truncate to as long as the other one got. May have to devise some penalty.
		return mean((e_brates-s_brates)/e_brates)+mean((e_xrates-s_xrates)/e_xrates)
		
	def sample(self, simulator):
		for n in range(self.nreps):
			if random.random()>.5: self.__sample_prior('b')
			else: self.__sample_prior('d')
			print("BSAMPLES",self.b_samples)
			print("DSAMPLES",self.d_samples)
			s_brates,s_erates=simulator.simulate(self.b_samples,self.d_samples)
			if s_brates != None and s_erates != None:
				summary_stat=self.__summary_statistic(s_brates,s_erates)
				print("SUCCESS",len(s_brates),summary_stat)
				if summary_stat <= self.threshold:
					self.outfile.write('\t'.join(self.b_samples.values())+'\t'.join(self.d_samples.values())+'\n')
			else:
				print("FAIL")


#MULTIPROCESS THIS!

def parallelize(emp_sp_rates,emp_ex_rates,outfile):
	sim=Simulator()
	abc=ABCMachine(empirical_sp_rate,empirical_ex_rate,outfile)
	abc.sample(sim)

'''
def __main__():
	empirical_sp_rates=np.array([0.275565637997, 0.274703454773,0.27382017623,0.273027688529,0.272145880404,0.272341223862,0.272602043178,0.273192351184,0.274909369282,0.291294628986,0.45606347023,0.572956859205,0.572956859205,0.572956859205,0.572956859205,0.571048846552,0.48505916213,0.48386290256,0.48386290256,0.392408250501,0.390340447576,0.38872033143,0.387629471653,0.387607939252,0.386328933623,0.320384071071,0.319745658695,0.286189569147,0.258880394049,0.255415687464,0.232465656996,0.230663483108,0.230663483108,0.23066431989,0.264374217524,0.264694436171,0.264694436171,0.263485564804,0.252624165748,0.226067939548,0.198929346948,0.198929346948,0.18171923032,0.199468823083,0.199468823083,0.199585706952,0.199916227924,0.199916227924])
	empirical_ex_rates=np.array([0.0635015731708, 0.0577535838904,0.0504168278498,0.0490163958319,0.0457606421071,0.0425822570584,0.0403190784357,0.0402238723914,0.040057385048,0.0403207303407,0.0403620030284,0.206914351849,0.208536723598,0.210959599507,0.214525736645,0.2398323818,0.243133246464,0.243133246464,0.243133246464,0.243151891906,0.243145464298,0.243135765253,0.243126081769,0.243148606991,0.243148606991,0.202161267215,0.202134126184,0.201985002242,0.154442578843,0.154442578843,0.129922258396,0.129912550963,0.129912550963,0.129916040371,0.130276678344,0.16832018558,0.187560398572,0.189388292963,0.189388292963,0.189388292963,0.174868681008,0.174808014281,0.174808014281,0.187962691938,0.206694081708,0.252940373704,0.314543487592,0.314543487592])
	num_workers = 4
	pool=mp.Pool(num_workers)
	for i in range(num_workers:
		pool.apply_async(parallelize,args =(empirical_sp_rates,empirical_ex_rates,'./test'+slave_no+'.txt' ))
	pool.join()
	pool.close()
	
__main__()
'''		
empirical_sp_rates=np.array([0.275565637997, 0.274703454773,0.27382017623,0.273027688529,0.272145880404,0.272341223862,0.272602043178,0.273192351184,0.274909369282,0.291294628986,0.45606347023,0.572956859205,0.572956859205,0.572956859205,0.572956859205,0.571048846552,0.48505916213,0.48386290256,0.48386290256,0.392408250501,0.390340447576,0.38872033143,0.387629471653,0.387607939252,0.386328933623,0.320384071071,0.319745658695,0.286189569147,0.258880394049,0.255415687464,0.232465656996,0.230663483108,0.230663483108,0.23066431989,0.264374217524,0.264694436171,0.264694436171,0.263485564804,0.252624165748,0.226067939548,0.198929346948,0.198929346948,0.18171923032,0.199468823083,0.199468823083,0.199585706952,0.199916227924,0.199916227924])
empirical_ex_rates=np.array([0.0635015731708, 0.0577535838904,0.0504168278498,0.0490163958319,0.0457606421071,0.0425822570584,0.0403190784357,0.0402238723914,0.040057385048,0.0403207303407,0.0403620030284,0.206914351849,0.208536723598,0.210959599507,0.214525736645,0.2398323818,0.243133246464,0.243133246464,0.243133246464,0.243151891906,0.243145464298,0.243135765253,0.243126081769,0.243148606991,0.243148606991,0.202161267215,0.202134126184,0.201985002242,0.154442578843,0.154442578843,0.129922258396,0.129912550963,0.129912550963,0.129916040371,0.130276678344,0.16832018558,0.187560398572,0.189388292963,0.189388292963,0.189388292963,0.174868681008,0.174808014281,0.174808014281,0.187962691938,0.206694081708,0.252940373704,0.314543487592,0.314543487592])
sim=Simulator()
abc=ABCMachine(empirical_sp_rate,empirical_ex_rate,'test.txt')
abc.sample(sim)

		
