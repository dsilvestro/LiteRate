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
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from collections import OrderedDict
import multiprocessing as mp
from sys import argv
from literate_library import *
print("Birth-Death Sampler 18\n")


##########################################################################
###########                 SIMULATION SETTINGS                 ##########
##########################################################################


class Simulator(object):
	
	def __init__(self):
		
		###SIMULATION SETTINGS###
		self.theoretical_only = True  #if True, the next block is irrelevant since those are stochastic guidelines
		
		self.n_reps = 100 # number of simulations
		# CONSTRAINTS on DATA SIZE (simulations will run until size requirements are met)
		self.s_species=5   # number of starting species
		self.minSP=100     # min standing diversity
		self.maxSP=20000    # max size standing diversity
		self.minEX_SP=100    # minimum number of extinct lineages allowed
		
		
		self.scale=100 #this parameter is used to stretch out the simulation, approximating continuous time
		self.origin = 1968	# amount of time to simulate. specified in TBP
		self.present=2000
		self.time_frame=self.present-self.origin+1
		self.t_as_TBP=False
		

		#LOGISTIC GROWTH PARAMETERS
		self.logistic_params={
		'l_max':.425,
		'm_max':.165,
		'L':23227.11, #max carrying cappacity
		'div_0':188.15, #min carrying capacity
		'k':5.1, #growth rate
		'x0':9, #x value at midpoint
		'nu':1.0 #skew where growth is occuring the most greater than 1 skewed towards beginning, less than 1 skewed towards end.
		#Must be greater than 0
		}
		
		###SUSCEPTIBLE/INFECTED PARAMETERS
		self.si_params={
				'l0':.5,
				'm0':.2,
				'gamma':.7, #birth modulator
				'threshold':.2,
				}
		#AGE-DEPENDENT EXTINCTION PARAMETERS
		self.mean_lifespan=2.7
		self.w_shape=0.5
		self.w_scale = self.mean_lifespan/ gamma_func(1+1/self.w_shape)


		###FIXED VECTOR PARAMETERS
		self.m_vector=np.array([.2]*9 +[.6]*5+[.3]*15+[1.4]+[.2*.9])
		self.m_vector=np.array([.2]*15 +[2] +[.2]*8+[.4]*7)
		self.m_vector=np.array([.208]*31)
		self.l_vector=np.array([.3,.3,.3]+ [.4]*5+ [.6]*7+ [.3]*6+ [.45,.45,.425,.425,.4]+ [.4]*5)

	def __get_logistic(self,t):
		p=self.logistic_params
		t=t/self.scale
		return ( p['div_0'] + p['L']/((1+exp(-p['k']*(t-p['x0'])))**(1/p['nu'])) )


	def __get_dKdT_tSI(self,D_t,K_t):
		D=D_t/self.maxSP
		K=K_t/self.maxSP
		dKdT= (self.si_params['gamma']*D+self.si_params['threshold'])*(1-K)
		dKdT=self.maxSP*dKdT
		return dKdT
			
	
	def __calc_branchlength(self,ts,te,t0,t1):
		return get_br(ts,te,t0,t1)

	def simulate(self,logistic_params=None):
		if logistic_params!=None: self.b_logistic_params=b_params
		#Could aternatively be set up to take fixed rates or different model equations here
		time=np.array(list(range(int(self.time_frame*self.scale))))
		print(time)
		K_vec= np.array(self.__get_logistic(time))
		print(K_vec)
		LOtrue=[0]
		n_extinct=-0
		tries=0
		while len(LOtrue) < self.minSP or len(LOtrue) > self.maxSP or n_extinct < self.minEX_SP:
			if tries>=100: 
				return None
				break
			ts=repeat(self.origin*self.scale,self.s_species)
			te=repeat(0,self.s_species)
			Kt=K_vec[0]
			st_Dt=self.s_species*1.0
			th_Dt=self.s_species*1.0
			
			#EXAMPLE FOR CLADE ADE WHERE EACH LINEAGE HAS CONSTANT EXTINCTION RISK OVER TIME
			#m=repeat(self.si_params['m0']*1.0*self.s_species/K_t,self.s_species)
			
			thb_rates=[]
			thd_rates=[]
			eb_rates=[]
			ed_rates=[]
			th_diversity=[]
			st_diversity=[]
			carrying_capacity=[]
			
			births_cache=0
			deaths_cache=0


			for t in range(int(self.time_frame*self.scale)): #this is to smooth
				alive_indx=(te==0).nonzero()[0] #indices of living lineages
				st_Dt=len(alive_indx)*1.0
				
				#example with diff EQ
				#dK_dt = self.__get_dKdT_tSI(D_t,K_t)
				#K_t= max(self.s_species,K_t+dK_dt/self.scale)
				
				#example with fixed rates or trend
				#smooth_t=int((t/self.scale)-self.root_age)
				#l_t=l_vec[smooth_t]
				#m_t=m_vec[smooth_t]
				
				Kt= K_vec[t]
				st_lt = max(0.0,self.si_params['l0']-(self.si_params['l0']-self.si_params['m0'])*st_Dt/Kt)
				st_mt = max(0.0,self.si_params['m0']+(self.si_params['l0']-self.si_params['m0'])*st_Dt/Kt)
				th_lt = max(0.0,self.si_params['l0']-(self.si_params['l0']-self.si_params['m0'])*th_Dt/Kt)
				th_mt = max(0.0,self.si_params['m0']+(self.si_params['l0']-self.si_params['m0'])*th_Dt/Kt)

				
				#rescale rates: if we are making this time continuous rates get divided by 100				
				st_lt=st_lt/self.scale
				st_mt=st_mt/self.scale
				th_lt=th_lt/self.scale
				th_mt=th_mt/self.scale
				
				th_Dt=th_Dt+(th_Dt*th_lt)-(th_Dt*th_mt)
				
				
				
				###CALCULATING STOCHASTIC BIRTH/DEATH RATES
				ran_vec=np.random.random(len(alive_indx))


				r_sp_indx = (ran_vec < st_lt).nonzero()[0]
				r_ex_indx = np.intersect1d((ran_vec >= st_lt).nonzero()[0], (ran_vec < st_lt+st_mt).nonzero()[0])
				kill_indx=alive_indx[r_ex_indx]

				
				#we'll create a cache that we'll dump at new discrete time bin for emprical birth rates
				births_cache+= len(r_sp_indx)
				deaths_cache+= len(kill_indx)
	
				te[kill_indx]=t #kill em all

				
				
				#birth em
				ts=append(ts,repeat(t,len(r_sp_indx)))
				te=append(te,repeat(0,len(r_sp_indx)))
	
				
				
				if t % self.scale == 0:
					print("t:",t,"thDt:",th_Dt,"stDt:",st_Dt,"thlt",th_lt*self.scale,"thmt",th_mt*self.scale,"stlt",st_lt*self.scale,"K",Kt)
					branch_length=self.__calc_branchlength(ts,te,t-self.scale,t)
					
					
					eb_rates.append(round((births_cache/max(branch_length/self.scale,.00001)),4))
					ed_rates.append(round((deaths_cache/max(branch_length/self.scale,.00001)),4))
					
					thb_rates.append(th_lt*self.scale)
					thd_rates.append(th_mt*self.scale)
					
					th_diversity.append(th_Dt)
					st_diversity.append(st_Dt)
					carrying_capacity.append(Kt)
					
					
					births_cache=0
					deaths_cache=0
	
				
				LOtrue=te
				n_extinct = len(te[te<0])
			if self.theoretical_only==True:
				return (np.array(thb_rates), np.array(thd_rates),
						None, None,
						np.array(th_diversity),None,
						np.array(carrying_capacity),
						None,None)
			tries+=1	
			print("tries:",tries)
			
		ts, te= floor(-array(ts)/self.scale), floor(-(te)/self.scale)
		#if self.t_as_TBP: ts, te = self.present_year-ts, self.present_year-te
		
		#average theoretical rates down to years
		#thb_rates= thb_rates[:(-1*(len(thb_rates)%100))] #truncate to multiple of 100
		#thd_rates= thd_rates[:(-1*(len(thd_rates)%100))]
		#thb_rates = np.round(np.mean(np.array(thb_rates).reshape(-1, int(100)), axis=1),4) #this averages every 100 times to get 
		#thd_rates = np.round(np.mean(np.array(thd_rates).reshape(-1, int(100)), axis=1),4)
		print("TRIES",tries)		
		return (	np.array(thb_rates), np.array(thd_rates),
				np.array(eb_rates), np.array(ed_rates),
				np.array(th_diversity),np.array(st_diversity),
				np.array(carrying_capacity),
				ts,te)

test=Simulator()
attempt=test.simulate()
plt.plot(attempt[2],linewidth=2,linestyle='dashed',color='blue')
plt.plot(attempt[3],linewidth=2,linestyle='dashed',color='red')

'''

plt.plot(attempt[0],linewidth=2,color='blue')
plt.plot(attempt[1],linewidth=2,color='red')
plt.plot(attempt[2]-attempt[3],linewidth=2,linestyle='dashed',color='green')
plt.plot(attempt[4],linewidth=2,linestyle='dashed',color='purple')
plt.plot(attempt[5],linewidth=2,linestyle='dashed',color='orange')


si={'l0':0.525,
	'm0':0.208,
	'gamma':.7 ,
	'threshold':0.18739526337605672,
	}
					
b_scenarios=[]
b1={
		'l0':.55,
		'K':1500, #max carrying cappacity
		'A0':5, #min carrying capacity
		'B':1, #growth rate
		'x0':-33/2.0, #x value at midpoint
		'v':1.0 #skew where growth is occuring the most greater than 1 skewed towards beginning, less than 1 skewed towards end.
		#Must be greater than 0
		}



b2=b1.copy(); b2['K']=1600; b2['l0']=.6; b_scenarios.append(b2)
b2=b1.copy(); b2['K']=1800; b2['l0']=.6; b_scenarios.append(b2)
b2=b1.copy(); b2['K']=2000; b2['l0']=.6; b_scenarios.append(b2)
b2=b1.copy(); b2['K']=2400; b2['l0']=.6; b_scenarios.append(b2)

b2=b1.copy(); b2['K']=1600; b2['l0']=.5; b_scenarios.append(b2)
b2=b1.copy(); b2['K']=1800; b2['l0']=.5; b_scenarios.append(b2)
b2=b1.copy(); b2['K']=2000; b2['l0']=.5; b_scenarios.append(b2)
b2=b1.copy(); b2['K']=2400; b2['l0']=.5; b_scenarios.append(b2)

b2=b1.copy(); b2['K']=1600; b2['l0']=.4; b_scenarios.append(b2)
b2=b1.copy(); b2['K']=1800; b2['l0']=.4; b_scenarios.append(b2)
b2=b1.copy(); b2['K']=2000; b2['l0']=.4; b_scenarios.append(b2)
b2=b1.copy(); b2['K']=2400; b2['l0']=.4; b_scenarios.append(b2)


b2=b1.copy(); b2['x0']=-37; b_scenarios.append(b2)
b2=b1.copy(); b2['x0']=-43; b_scenarios.append(b2)
b2=b1.copy(); b2['x0']=-47; b_scenarios.append(b2)
b2=b1.copy(); b2['x0']=-52; b_scenarios.append(b2)

d={
		'm0':.2, # this is used in
		'K':1.0, #max % of baseline death rate
		'A0':0.6, #min % of baseline death rate
		'B':.5, #growth rate
		'x0':-40/2.0, #x value at midpoint
		'v':1.0 #skew where growth is occuring the most greater than 1 skewed towards beginning, less than 1 skewed towards end.
		#Must be greater than 0
		}




'''
'''
with PdfPages('C:\\Users/bernard/Box Sync/metal_project/13-simulation_trials/test_simulations.pdf') as pdf:
	for b in b_scenarios:	
		plt.suptitle('l0: '+str(b['l0'])+'\tK: '+str(b['K'])+'\tx0:\t'+str(b['x0']))
		for i in range(1,4,2):
			attempt=None
			while attempt==None: attempt=test.simulate(b,d)
			if attempt!=None:
				plt.subplot(2,2,i)
				plt.plot(attempt[0],linewidth=2,color='blue')
				plt.plot(attempt[1],linewidth=2,color='red')
				plt.plot(attempt[2],linewidth=2,linestyle='dashed',color='blue')
				plt.plot(attempt[3],linewidth=2,linestyle='dashed',color='red')
				plt.plot(attempt[3]-attempt[2],linewidth=2,linestyle='dashed',color='green')
				plt.subplot(2,2,i+1)
				plt.plot(attempt[4],linewidth=2,linestyle='dashed',color='purple')
				plt.plot(attempt[5],linewidth=2,linestyle='dashed',color='orange')
				plt.tight_layout()
				
		pdf.savefig()
		plt.close()
		plt.suptitle('l0: '+str(b['l0'])+'\tK: '+str(b['K'])+'\tx0:\t'+str(b['x0']))
		for i in range(1,4,2):
			attempt=test.simulate(b,d)
			if attempt!=None:
				plt.subplot(2,2,i)
				plt.plot(attempt[0],linewidth=2,color='blue')
				plt.plot(attempt[1],linewidth=2,color='red')
				plt.plot(attempt[2],linewidth=2,linestyle='dashed',color='blue')
				plt.plot(attempt[3],linewidth=2,linestyle='dashed',color='red')
				plt.plot(attempt[3]-attempt[2],linewidth=2,linestyle='dashed',color='green')
				plt.subplot(2,2,i+1)
				plt.plot(attempt[4],linewidth=2,linestyle='dashed',color='purple')
				plt.plot(attempt[5],linewidth=2,linestyle='dashed',color='orange')
				plt.tight_layout()
				

		pdf.savefig()
		plt.close()

'''
class ABCMachine(object):
	
	def __init__(self,eb_rates,ex_rates,outfile,nreps=10000):
		#print(eb_rates,ex_rates,outfile)
		self.nreps=nreps
		self.eb_rates=eb_rates
		self.ex_rates=ex_rates
		self.seed=1337
		
		self.outfile=open(outfile,'w')

	   
	            #BIRTH PRIORS
		self.b_priors=OrderedDict()
		self.b_priors['A0']=5 #min carrying capacity
		self.b_priors['K']=uniform(16000) #max carrying cappacity draw uniformyly and then exponentiate in log space
		#fix K
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
		
		self.SI_priors=OrderedDict()
		self.SI_priors['l0']=.525
		self.SI_priors['m0']=.208
		self.SI_priors['gamma']=uniform(0,1) #THIS IS IGNORED IN CRRENT IMPLIMENTATION
		self.SI_priors['threshold']=uniform(0,.9999)

		
		self.outfile.write('\t'.join(self.SI_priors.keys())+'\tsummary_stochastic\tsummary_theoretical\tsb_rates\tsd_rates\tth_rates\tth_rates\n')
	
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


		self.SI_samples=OrderedDict()
		#initial params
		for p in self.SI_priors.keys():
			if type(self.SI_priors[p]) not in [int, float]:
				self.SI_samples[p]=self.SI_priors[p].rvs()
			else: self.SI_samples[p]=self.SI_priors[p]
	'''
	def __sample_prior(self,param='random'):
		p=param
		#any prior except param 1 which is fixed for now
		if param == 'random': p = random.choice(self.SI_priors.keys()[2:]) 
		self.SI_samples[p]=self.SI_priors[p].rvs()
	'''
	def __sample_prior(self,param='random'):
		p=param
		#any prior except param 1 which is fixed for now
		if param == 'random': p = random.choice(self.SI_priors.keys()[2:])
		if p =='threshold': 
			self.SI_samples['threshold']=self.SI_priors['threshold'].rvs()
			self.SI_samples['gamma']=uniform(.0001,1-self.SI_samples['threshold']).rvs()
		else:
			self.SI_samples['gamma']=uniform(.0001,1-self.SI_samples['threshold']).rvs()
		
	def __summary_statistic(self,s_brates,s_xrates):
		eb_rates=self.eb_rates[:len(s_brates)] #truncate to as long as the other one got. May have to devise some penalty.
		ex_rates=self.ex_rates[:len(s_xrates)] #truncate to as long as the other one got. May have to devise some penalty.
		deviations=concatenate([(abs(eb_rates-s_brates)/eb_rates),(abs(ex_rates-s_xrates)/ex_rates)])
		return round(mean(deviations),4)
	

		
	def sample(self, simulator):
		for n in range(self.nreps):
			self.__sample_prior()
			print("KEYS",self.SI_samples.items())
			self.outfile.write('\t'.join(map(str,self.SI_samples.values())))
			results=simulator.simulate(SI_params=self.SI_samples)
			if results!=None:
				summary_stat=self.__summary_statistic(results[0],results[1])
				summary_th=self.__summary_statistic(results[2],results[3])
				self.outfile.write('\t'+str(summary_stat)+'\t'+\
					   str(summary_th) + '\t' + \
					   ','.join(map(str,results[0]))+'\t'+\
					   ','.join(map(str,results[1]))+'\t'+\
					   ','.join(map(str,results[2]))+'\t'+\
					   ','.join(map(str,results[3]))+'\t'+\
					   '\n')
			else:
				self.outfile.write('\tNA\tNA\tNA\tNA\tNA\tNA\n')
			self.outfile.flush()
'''
empirical_sp_rates=np.array([0.289743008359, 0.287367182534,0.284896320833,0.280636840251,0.280341735686,0.280496778232,0.281470432507,0.291538191317,0.322504264968,0.48112876232,0.522630854588,0.524657573667,0.524686910807,0.524664981881,0.517896743275,0.450228134092,0.438813867216,0.438792482559,0.357830674458,0.356775259624,0.356013665723,0.355513398002,0.355477490528,0.350315667557,0.299243748343,0.295284794078,0.273007253616,0.242753236612,0.237542913652,0.237368041342,0.237368041342])
empirical_ex_rates=np.array([0.0339931450196, 0.0327806387538,0.0325335921417,0.032398195428,0.0323478493467,0.0323539536075,0.0324292593697,0.0326491347942,0.033592640723,0.0554936055319,0.0793795493384,0.136416116364,0.148592110113,0.149509204258,0.193116249817,0.206377895916,0.206393874451,0.206383157358,0.206386386,0.206460734731,0.206471218662,0.206500717936,0.207980286224,0.204930951255,0.200808652149,0.197172220377,0.178120670042,0.154238761299,0.138940401632,0.138042528337,0.138042528337])
sim=Simulator()
abc=ABCMachine(eb_rates=empirical_sp_rates,\
			ex_rates=empirical_ex_rates,\
			outfile=argv[1],\
			nreps=int(argv[2]))
abc.sample(sim)
'''

