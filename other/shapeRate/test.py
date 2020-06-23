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
import pandas as pd
#np.set_printoptions(suppress=True)
#np.set_printoptions(precision=3)  
from time import time


def get_log_pmf_beta_discrete(num_bins, a, b):
	bins = np.linspace(0, 1, num_bins + 1)
	cdf_b = scipy.stats.beta.cdf(bins, a, b, loc=0, scale=1)
	return np.log(np.diff(cdf_b))

def calc_lik_BetaBin(data, par, n_bins, indx = -1):
	alpha, beta = par
	if indx == -1:
		lik_vec = get_log_pmf_beta_discrete(n_bins, alpha, beta)
		lik_list = (lik_vec-  np.log(np.sum(np.exp(lik_vec))))* data
	else:
		lik_list = np.zeros(len(alpha))
		i=0
		for a,b in zip(alpha,beta):
			lik_vec = get_log_pmf_beta_discrete(n_bins, a,b)
			lik_list[i] = (lik_vec[indx]-  np.log(np.sum(np.exp(lik_vec))))* data
			i+=1
	return lik_list



data = 5
par =[ np.array([2.,4]), np.array([1.,1]) ]
n_bins=5
indx=2



data = np.array([13,14])
par=[2.,4]
n_bins=2
print(calc_lik_BetaBin(data, par, n_bins))