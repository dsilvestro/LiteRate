from numpy import *
import numpy as np
import scipy.special
np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)  
random.seed(1)
np.random.seed(1)


# FUNCTIONS
def get_age_mu(w_shape,w_scale,age):
	return w_shape/w_scale * (age/w_scale)**(w_shape-1)


def death_age_log_likelihood_vec(s,e,w_shape,w_scale):
	vec = np.linspace(0.01,0.99,10)
	
	# discrete integral
	dT = e-s
	sum_m = sum_m_vec[dT]
	m_death = m_death_vec[dT]
		
	if dT>0: 
		sum_m = np.sum(get_age_mu(w_shape,w_scale,np.arange(e-s-1)+0.5))
	else: 
		sum_m = 0# m_death
	m_death = np.mean(get_age_mu(w_shape,w_scale,(e-s)+vec))
	sum_m += m_death
	
	return np.log(m_death) - sum_m



def death_age_log_likelihood(s,e,w_shape,w_scale):
	vec = np.linspace(0.01,0.99,10)
	
	# discrete integral
	dT = e-s	
	if dT>0: 
		sum_m = np.sum(get_age_mu(w_shape,w_scale,np.arange(e-s-1)+0.5))
	else: 
		sum_m = 0# m_death
	m_death = np.mean(get_age_mu(w_shape,w_scale,(e-s)+vec))
	sum_m += m_death
	
	return np.log(m_death) - sum_m


def death_age_log_likelihood_slow_version(s,e,w_shape,w_scale):
	vec = np.linspace(0.01,0.99,10)
	m_death = np.mean(get_age_mu(w_shape,w_scale,(e-s)+vec))
	
	# discrete integral
	dT = e-s	
	if dT>0:
		sum_m = 0
		for i in range(int(e-s)):
			sum_m += np.mean(get_age_mu(w_shape,w_scale,i+vec))
	else: sum_m = m_death
	
	return np.log(m_death) - sum_m


def birth_log_likelihood(s,e,l):
	# can take a vector of s and e
	return np.sum(np.log(l)*len(s) - np.sum((e-s)*l))


def update_prm(q,d=1.1):
	u = np.random.uniform(0,1)
	l = 2*log(d)
	m = exp(l*(u-.5))
 	new_q = q * m
	return new_q


# GENERATE DATA 
W_shape = .5
mean_longevity = 3.
# get the scale yielding desired longevity
W_scale = mean_longevity/scipy.special.gamma(1+1./W_shape)
n_lineages = 100
# assumes forward time e > s
e = np.random.weibull(W_shape,n_lineages)*W_scale
s = np.zeros(n_lineages)
print "Mean longevity", mean(e-s), "scale:", W_scale

# discretize data and add random uncertainty on exact age of death
e = e.astype(int) #+ np.random.random(len(e))

rescale = 1.
e = e/rescale

# ANALYZE DATA

# init prm values
lA        = .25
w_shapeA  = 1.
w_scaleA  = 2.
# init likelihood
dLik = np.sum([death_age_log_likelihood(s[i],e[i],w_shapeA,w_scaleA) for i in range(n_lineages)])
bLik = birth_log_likelihood(s,e,lA)
likA = dLik+bLik

# max likelihood (crappy) optimizer 
for iteration in range(5001):
	l,w_shape,w_scale = lA,w_shapeA,w_scaleA
	l = update_prm(lA,1.1)
	if np.random.random()>0.5:
		w_shape = update_prm(w_shapeA,1.1)
	else:
		w_scale = abs(np.random.normal(w_scaleA,0.5))
	
	# calc likelihood	
	dTunique = np.unique()
	if dT>0: 
		sum_m = [np.sum(get_age_mu(w_shape,w_scale,np.arange(dTunique[i]-1)+0.5)) for i in]
	else: 
		sum_m = 0# m_death
	
	
	sum_m_vec = np.cumsum(get_age_mu(w_shape,w_scale,np.arange(np.max(e-s)-1)+0.5))
	m_death_vec = np.cumsum(get_age_mu(w_shape,w_scale,np.arange(np.max(e-s)-1)+0.5))
	
	
	dLik = np.sum([death_age_log_likelihood(s[i],e[i],w_shape,w_scale) for i in range(n_lineages)])
	bLik = birth_log_likelihood(s,e,l)
	lik = dLik+bLik
	
	if lik > likA:
		likA = lik
		w_shapeA = w_shape
		w_scaleA = w_scale
		lA       = l
	
	if iteration % 1000 ==0: 
		w_scaleA_rescaled = w_scaleA*rescale
		print iteration, np.array([likA, lA, w_shapeA, w_scaleA_rescaled, w_scaleA_rescaled*scipy.special.gamma(1+1./w_shapeA)])


