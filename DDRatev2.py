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
from literate_library import *

np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)  

########################
 #THIS SETUP IS ESSENTIALLY THE SAME FOR ALL SCRIPTS

p = core_arguments()
#ADD EXTRA ARGUMENTS
p.add_argument('-m_birth', type=int, help='0) use const b rates 1) DD birth 2) niche dep DD b', default=2,metavar=2)
p.add_argument('-m_death', type=int, help='-1) fixed d rate 0) use const d rates 1) DD death 2) niche dep DD d', default=2,metavar=2)
p.add_argument('-fix_birth', type=float, help='Fix birth rate (with -m_birth -1)', default=0.1,metavar=0.1)
p.add_argument('-fix_death', type=float, help='Fix death rate (with -m_death -1)', default=0.1,metavar=0.1)


args = p.parse_args()

seed=set_seed(args.seed)

TS,TE,PRESENT,ORIGIN=parse_ts_te(args.d,args.TBP,args.first_year,args.last_year,args.death_jitter)

ORIGIN, PRESENT, N_SPEC, N_EXTI, DT, N_TIME_BINS, TIME_RANGE=create_bins(ORIGIN, PRESENT,TS,TE,args.rm_first_bin)

print(ORIGIN, PRESENT)

B_EMP,D_EMP=print_empirical_rates(N_SPEC,N_EXTI,DT)

#######PUT ADDITIONAL GLOBALS HERE#########
M_BIRTH = args.m_birth
M_DEATH = args.m_death
PRIOR_K0_L = np.max(DT) # scale of Gamma(1,s) prior
SMALL_NUMBER = 0.000000000000001 #used for flooring rates
INIT_BIRTH = args.fix_birth
INIT_DEATH = args.fix_death


########################OTHER FUNCTIONS########


    
#following notation on wikipedia except that we have added a booster for the min
def get_logistic(x,L,k,x0,div_0,nu):
    return( div_0 + L/((1+exp(-k*(x-x0)))**(1/nu)) )

def get_const_K(x,L,div_0):
    return( np.ones(len(x))*(L+div_0) )

def get_brates(rate_f,rate_mul,niche_frac):
    rate_max=rate_f+rate_f*np.exp(rate_mul)
    rate =  rate_max - (rate_max-rate_f)*niche_frac
    rate[rate<=0] = SMALL_NUMBER #no negative birth rates
    return(rate)

def get_drates(rate_f,rate_mul,niche_frac):
    rate_min=rate_f-rate_f*rate_mul
    rate =  rate_min + (rate_f-rate_min)*niche_frac
    rate[rate<=0] = SMALL_NUMBER #no negative birth rates
    return(rate)

def likelihood_function(args):
    [l_f, l_mul,  k, x0,     div_0,   L,  m_mul, nuB, nuD] = args

    if M_BIRTH==0:
        birth_rates = np.ones(N_TIME_BINS)*l_f*np.exp(l_mul)
        niche = np.ones(N_TIME_BINS)
        niche_frac = np.ones(N_TIME_BINS)
    elif M_BIRTH==1:
        niche = get_const_K(TIME_RANGE,L,div_0)
        niche_frac = DT/niche
        birth_rates = get_brates(l_f,l_mul,(niche_frac**nuB))
    elif M_BIRTH==2:
        niche = get_logistic(TIME_RANGE,L,k,x0,div_0,1)
        niche_frac = DT/niche
        birth_rates = get_brates(l_f,l_mul,(niche_frac**nuB))
    birth_lik = np.sum(log(birth_rates)*N_SPEC - birth_rates*DT)
    #print(niche)

    if M_DEATH <=0:    
        death_rates = np.ones(N_TIME_BINS)*l_f
        #niche = np.ones(N_TIME_BINS)
        #niche_frac = np.ones(N_TIME_BINS)
    elif M_DEATH ==1:
        niche = get_const_K(TIME_RANGE,L,div_0)
        niche_frac = DT/niche
        death_rates =  get_drates(l_f,m_mul,(niche_frac**nuD))
    elif M_DEATH==2:
        niche = get_logistic(TIME_RANGE,L,k,x0,div_0,1)
        niche_frac = DT/niche
        death_rates = get_drates(l_f,m_mul,(niche_frac**nuD))
    death_lik = np.sum(log(death_rates)*N_EXTI - death_rates*DT)
    lik = np.array([birth_lik, death_lik])
    # print(niche, M_DEATH)
    # quit()
    

    return [lik, birth_rates, death_rates, niche, niche_frac]


def calc_prior(args):
    #argsA=np.array([l_f,  l_mul,  k, x0,     div_0,   L,  m_mul, nuB, nuD])
    p = prior_gamma(args[0],a=1,s=10,l=0) #l_f
    p += prior_gamma(args[1],a=1,s=1,l=1) #l_mul
    p += prior_beta(args[6],a=1,b=1.2) #M_mul
    p += prior_norm(args[2]) #k
    p += prior_gamma(args[4],a=1,s=PRIOR_K0_L,l=0) #div_0
    p += prior_gamma(args[5],a=1,s=PRIOR_K0_L,l=0) #L
    p += prior_gamma(args[7],a=2,s=.5,l=0) #nuB
    p += prior_gamma(args[8],a=2,s=.5,l=0) #nuD
    if ORIGIN + args[3]>= PRESENT:
        p = -np.inf #if midpoint greater than present: fail
    return p
    
def __main__(parsed_args):    
    
    
    out=""
    if M_BIRTH==0: out += "_LL"
    elif M_BIRTH==1: out += "_LDD"
    elif M_BIRTH==2: out += "_LDDN"
    if M_DEATH<=0: out += "_ML"
    elif M_DEATH==1: out += "_MDD"
    elif M_DEATH==2: out += "_MDDN"
    
    outfile = "%s_%s%s.log" % (os.path.splitext(parsed_args.d)[0], seed, out)
    logfile = open(outfile , "w") 
    wlog=csv.writer(logfile, delimiter='\t')
    head =["it","posterior","likelihood","likelihood_birth","likelihood_death","prior","l_f","l_mul","steepness_k","midpoint_x0",\
    "initCarryingCap","maxCarryingCap","m_mul","nuB","nuD"]
    for i in range(len(DT)): head.append("l_%s" % i)
    for i in range(len(DT)): head.append("m_%s" % i)
    for i in range(len(DT)): head.append("niche_%s" % i)
    for i in range(len(DT)): head.append("nicheFrac_%s" % i)
    head+=["corr_coeff","rsquared","gelman_r2"]
    wlog.writerow(head)
    
    
    
    
    
    L = 20000 # maximum
    k = 1.5 # steepness
    x0 = PRESENT - np.mean([ORIGIN, PRESENT]) # midpoint
    div_0 = 10 # starting carrying capacity
    l_f = 0.5 #birth and death rates at carrying capacity
    if M_DEATH==-1: m_mul = init_death #fixed death rate
    else: m_mul=.99 #multiply by l_f and -1 to get minimum death rate
    if M_BIRTH==-1: l_mul = init_birth #fixed birth rate
    else: l_mul=1.01 #multiply by l_f to get maximum birth rate
    nuB = 1.
    nuD = 1.
    
    
    argsA=np.array([l_f,  l_mul,  k, x0,     div_0,   L,  m_mul, nuB, nuD])
    
    
    #figure out which params to update based on model Note that nu is an extended logistic param which we are not currently using
    
    
    #constant birth and death
    if M_BIRTH==0 and M_DEATH<=0:
        #argsA=             np.array([l_f,l_max,  k,    x0,     div_0,   L,     m_max, nuB, nuD])
        update_multiplier = np.array([1,  1,  0,  0,  0,  0,  0,  0,  0]) #we wont update l_f and just let multipliers figure it out
    elif M_BIRTH==2 or M_DEATH==2:
        #argsA=np.array([l_f,  l_mul,  k, x0,     div_0,   L,  m_mul, nuB, nuD])
        update_multiplier = np.array([1,  1,  1,  0,  1,  1,  0,  1,  1])   
    else:
        #argsA=np.array([l_f,  l_mul,  k, x0,     div_0,   L,  m_mul, nuB, nuD])
        update_multiplier = np.array([1,  1,  0,  0,  0,  1,  0,  1,  1])   

    if M_DEATH== -1: # [DS: I need to fix this]
        #argsA=             np.array([l_max, k, x0,   div_0,   L,     m_max, nu])
        update_multiplier *= np.array([0, 1.,    0, 1,       1,   1,       0 , 1, 0])
    if M_BIRTH== -1: # [DS: I need to fix this]
        #argsA=             np.array([l_max, k, x0,   div_0,   L,     m_max, nu])
        update_multiplier *= np.array([0, 1.,    0, 1,       1,   1,       0 , 0, 0])
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
    while iteration != parsed_args.n:
        args = argsA+0.
        updated_ext = 0
        hastings= 0
        rr = np.random.random(3)
        if rr[1]<0.1 and (M_BIRTH>=1 or M_DEATH>=1):
            res = argsA+0
            if rr[2]<.5:
                res[3] = update_sliding_win(res[3], m=0, M=PRESENT, d=1.5) #update midpoint (the only sliding window proposal)
            else:
                res[6] = update_sliding_win(res[6], m=0, M=1, d=.05)
            if M_DEATH== -1:
                res[2] = update_normal_nobound(res[2], d=0.2) #update slope
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
        if iteration % parsed_args.s==0:
            #print lik,prior, args
            argsO=deepcopy(argsA) #when you copy lists, makes sure you dont change things by reference
            argsO[3] += ORIGIN # right point in time
            argsO[5] += argsO[4] #true max is div_0 + L
            argsO[1] = np.exp(argsO[1])
            print(iteration, likA, argsO) #, args
            
            #compute adequacy stats
            adequacy=calculate_r_squared(B_EMP,D_EMP,birth_rates,death_rates)
            #print(adequacy)
            l= [iteration,likA+priorA, likA,likBirthA,likDeathA, priorA] + list(argsO) + list(birth_rates) + list(death_rates) + list(niche) + list(nicheFrac) + list(adequacy)
            wlog.writerow(l)
            logfile.flush()
            os.fsync(logfile)
            
        iteration += 1

__main__(args)
    
