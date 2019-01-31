setwd("/Users/danielesilvestro/Desktop/test_time_series")
# proposal functions
sliding_window <- function(i,wsize=2){
	new_i <- i+(runif(1)-0.5)*wsize
	return(new_i)
}

multiplier_proposal <- function(i,d=1.2){
	u <- runif(1)
	l <- 2*log(d)
	m <- exp(l*(u-.5))
 	ii <- i * m
	hastings_ratio <- log(m)
	return( c(ii, hastings_ratio) )
}

get_mu_t_linear <- function(mu_0,time_year,mu_a){
	return( mu_0 + time_year*mu_a )
}

get_sd_t_linear <- function(sd_0,time_year,sd_a){
	transf_sd = sd_0 + time_year*sd_a
	transf_sd[transf_sd<0] = 0.01
	return( transf_sd )
}

####################################
####      FIT BINOMIAL PDF      ####
####################################
time_variable = cumsum(rnorm(111,0,1))
plot(time_variable,type="l",ylim=c(-1.2*max(abs(time_variable)),1.2*max(abs(time_variable))))

alpha = -0.5
response = 1- exp(-alpha*time_variable)
lines(response,col="red")


####################################
####       FIT NORMAL PDF       ####
####################################

# init output file
logfile <- "mcmc_samples.txt"
# the mean of a gamma distribution is shape/rate so we can write the mean in the output
cat(c("it","post","likelihood","prior","alpha","delta","sigma\n"),file=logfile,sep="\t")


# MCMC settings
n_iterations = 25000
sampling_freq = 100
print_freq = 500


get_mu_t_vector_Normal <- function(time_variable,alpha,mu0=0,delta=0,tau=0){
	# mu0 = initial mean aka intercept (only for WN, not for BM)
	# alpha = slope (correlation factor determining changing mean of normal density)
	# delta = point lag (integer >= 0, i.e. time before present)
	# tau = range lag (moving average): average of the last 0:tau points
	
	# BROWNIAN OPTION (nolag)
	# used to compute the likelihood of x[i] - x[i-1]
	# if no correlation mu_vec = c(0,0,0,0...)
	mu_vec = NULL # no need for a first value because the likelihood is computed starting from the second value
	
	for (i in 1:length(response_data)){
		delta = min(delta,start_time_obs_response) # avoid going where there is no data
		indx = (i-delta - tau):(i-delta)
		indx = indx + start_time_obs_response
		mu_vec[i]= alpha*mean(time_variable[indx])
	}
	# WHITE NOISE OPTION (no lag)
	# mu_vec = mu0 + mu_vec
	return(mu_vec)
	
}


#### SIMULATE DATA
# make up a predictor 
time_variable = cumsum(rnorm(111,0,1))
plot(time_variable,type="l",ylim=c(-1.2*max(abs(time_variable)),1.2*max(abs(time_variable))))

time_variable_diff = diff(time_variable) 
start_time_obs_response = 11 # at which time on the predictor do we start to have observations
				    # This also defines how much back in time you can go with lagged-models	
				    # if =1 they both start at the same time

# Add constant for starting value of variable
# or work on the likelihood of the difference between successive points!

response_data = rep(0,100)
response_data_mean = get_mu_t_vector_Normal(time_variable,alpha=1,mu0=0,delta=5,tau=0)
response_data = rnorm(length(response_data_mean),response_data_mean,sd=0.1)
lines(x= (start_time_obs_response+1):length(time_variable), y= response_data,type="l",col="red")

##### END SIMULATE

# init parameters
alpha_accepted = 0
delta_accepted = 0
sigma_accepted = 1
tau_accepted   = 0

# init likelihood and priors
mu_vector <- get_mu_t_vector_Normal(time_variable,alpha=alpha_accepted,delta=delta_accepted,tau=tau_accepted)		
likelihood_accepted <- sum(dnorm(response_data,mu_vector,sigma_accepted,log=T))	
prior_accepted <- 0


# MCMC loop
for (iteration in 0:n_iterations){
	
	alpha_new <- alpha_accepted
	delta_new <- delta_accepted
	sigma_new <- sigma_accepted
	tau_new   <- tau_accepted
	hasting   <- 0	
	
	r <- runif(1)
	if (r < 0.3){
		# update correlation prm
		alpha_new <- sliding_window(alpha_accepted,0.25)
	}else if (r > 0.6){
		# update lag
		delta_new <- abs(sliding_window(delta_accepted,0.25))
	}else{
		 l <- multiplier_proposal(sigma_accepted,1.2)
		 sigma_new = l[[1]]
		 hasting   = l[[2]]
	}
		
	mu_vector <- get_mu_t_vector_Normal(time_variable,alpha=alpha_new,delta=delta_new,tau=tau_new)		
	likelihood_new <- sum(dnorm(response_data,mu_vector,sigma_new,log=T))	
	prior_new <- 0	
	
	# calculate posterior ratio = likelihood ratio * prior ratio * hastings_ratio
	r <- (likelihood_new-likelihood_accepted) + (prior_new-prior_accepted) + hasting
	# accept or reject
	u = runif(1)
	if (r >= log(u)){
		# if new state accepted, set current parameters to the new ones	
		likelihood_accepted <- likelihood_new
		prior_accepted <- prior_new
		alpha_accepted <- alpha_new
		delta_accepted <- delta_new
		sigma_accepted <- sigma_new
		tau_accepted   <- tau_new
		
	}
	# print to screen
	if (iteration %% print_freq == 0){
		print( c(iteration,likelihood_accepted+prior_accepted,likelihood_accepted))
	}
	# save to file
	if (iteration %% sampling_freq == 0){
		cat(c(iteration,likelihood_accepted+prior_accepted,likelihood_accepted, prior_accepted, alpha_accepted, delta_accepted,sigma_accepted,"\n"),sep="\t",file=logfile,append=T)
	}
}
#points(x= start_time_obs_response:length(time_variable), y= cumsum(c(0,get_mu_t_vector_Normal(time_variable,alpha=alpha_accepted,delta=delta_accepted))),pch=19,col="darkblue")
#
#
#plot(x= start_time_obs_response:length(time_variable), y= c(0,get_mu_t_vector_Normal(time_variable,alpha=alpha_accepted,delta=delta_accepted)),pch=19,col="darkblue")
#
