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

def death_age_log_likelihood_discrete(s,e,w_shape,w_scale):
	# assumes forward time e > s
	m_death = get_age_mu(w_shape,w_scale,(e-s)+0.5)
	# discrete integral
	sum_m = np.sum(get_age_mu(w_shape,w_scale,np.arange(e-s)+0.5))
	return np.log(m_death) - sum_m


def death_age_log_likelihood_(s,e,w_shape,w_scale):
	if e-s== 0:
		vec = np.array([0.01,0.2,0.5,0.75])
		vec = np.linspace(0.2,0.6,4)
		
		m_death = mean(get_age_mu(w_shape,w_scale,(e-s)+vec))
		# discrete integral
		sum_m = m_death # np.sum(get_age_mu(w_shape,w_scale,np.arange(e-s)+0.5))
		return np.log(m_death) - sum_m
		
		
		# # assumes forward time e > s
		# m_death = get_age_mu(w_shape,w_scale,(e-s)+vec)
		# # discrete integral
		# sum_m1 = np.sum(get_age_mu(w_shape,w_scale,np.arange(e-s)+vec[0]))
		# sum_m2 = np.sum(get_age_mu(w_shape,w_scale,np.arange(e-s)+vec[1]))
		# sum_m3 = np.sum(get_age_mu(w_shape,w_scale,np.arange(e-s)+vec[2]))
		# sum_m4 = np.sum(get_age_mu(w_shape,w_scale,np.arange(e-s)+vec[3]))
		# lik_vec = m_death * np.exp(-np.array([sum_m1,sum_m2,sum_m3,sum_m4]))
		# return log(sum(lik_vec)/4.)
	else: return death_age_log_likelihood_discrete(s,e,w_shape,w_scale)


def death_age_log_likelihood(s,e,w_shape,w_scale):
	vec = np.linspace(0.01,0.99,10)
	
	# discrete integral
	dT = e-s	
	if dT>0: 
		m_death = get_age_mu(w_shape,w_scale,(e-s)+0.5)
		sum_m = np.sum(get_age_mu(w_shape,w_scale,np.arange(e-s)+0.5))
		#_ # first bin
		#_ sum_m = np.mean(get_age_mu(w_shape,w_scale,vec))
		#_ if dT>1:
		#_ 	sum_m += np.sum(get_age_mu(w_shape,w_scale,np.arange(1,e-s)+0.5))
	else: 
		m_death = np.mean(get_age_mu(w_shape,w_scale,(e-s)+vec))
		sum_m = 0# m_death
		sum_m += m_death
	
	return np.log(m_death) - sum_m


def death_age_log_likelihood_______(s,e,w_shape,w_scale):
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



def death_age_log_likelihood__(s,e,w_shape,w_scale):
	# assumes forward time e > s
	m_death = get_age_mu(w_shape,w_scale,(e-s))
	# integral (via numerical integration)
	#spacing = np.linspace(s,e,100)
	#sum_m = np.sum(get_age_mu(w_shape,w_scale,spacing[1:])*spacing[1])
	# integral (analytical form)
	sum_m = (w_scale/e) **(-w_shape)
	#print sum_m, sum_m1, w_shape
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
W_shape = 3.5
mean_longevity = 5
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


