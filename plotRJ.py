#!/usr/bin/env python 
from numpy import *
import numpy as np
import os,platform,glob,sys
import csv 
import argparse

def calcHPD(data, level=0.95) :
	assert (0 < level < 1)	
	d = list(data)
	d.sort()	
	nData = len(data)
	nIn = int(round(level * nData))
	if nIn < 2 :
		raise RuntimeError("not enough data")	
	i = 0
	r = d[i+nIn-1] - d[i]
	for k in range(len(d) - (nIn - 1)) :
		rk = d[k+nIn-1] - d[k]
		if rk < r :
			r = rk
			i = k
	assert 0 <= i <= i+nIn-1 < len(d)	
	return np.array([d[i], d[i+nIn-1]])


def print_R_vec(name,v):
	new_v=[]
	if len(v)==0: vec= "%s=c()" % (name)
	elif len(v)==1: vec= "%s=c(%s)" % (name,v[0])
	elif len(v)==2: vec= "%s=c(%s,%s)" % (name,v[0],v[1])
	else:
		for j in range(0,len(v)): 
			value=v[j]
			if isnan(v[j]): value="NA"
			new_v.append(value)

		vec="%s=c(%s, " % (name,new_v[0])
		for j in range(1,len(v)-1): vec += "%s," % (new_v[j])
		vec += "%s)"  % (new_v[j+1])
	return vec



def calcBF(threshold,empirical_prior):
	A = exp(threshold/2)*empirical_prior/(1-empirical_prior)
	return A/(A+1)

def get_prior_shift(t_start,t_end,bins_histogram):
	times_of_shift = []
	sampled_K = []
	# Gamma hyper-prior
	G_shape = 2. # currently hard-coded
	G_rate = 1.  # mode at 1
	min_time_frame_size = 1
	iteration=0.
	print "\nComputing empirical priors on rate shifts..."
	for rep in range(100000):
		if rep % 10000 ==0:
			sys.stdout.write(".")
			sys.stdout.flush()	
		try:
			# Poisson prior
			Poi_lambda = np.random.gamma(G_shape,1./G_rate)
			n_rates_temp = np.random.poisson(Poi_lambda,1000)
			n_rates = n_rates_temp[n_rates_temp>0][0]
			shift_times = list(np.random.uniform(t_end-min_time_frame_size,t_start+min_time_frame_size,n_rates-1))
			time_frames = np.sort([t_start, t_end]+shift_times)	
			if min(np.diff(time_frames))<min_time_frame_size:
				pass
			else:
				iteration+=1
				times_of_shift += shift_times
				sampled_K.append(n_rates)
		except(IndexError): pass
	expectedK = np.array(sampled_K)
	prior_s = np.mean(np.histogram(times_of_shift,bins=bins_histogram)[0]/iteration)
	bf2 = calcBF(2,prior_s)
	bf6 = calcBF(6,prior_s)
	print np.array([prior_s,bf2,bf6])
	return [prior_s,bf2,bf6]


def get_marginal_rates(f_name,min_age,max_age,nbins=0,burnin=0.2):
	# returns a list of 5 items:
	# 1. a vector of times (age of each marginal rate)
	# 2-4. mean, min and max marginal rates (95% HPD)
	# 5. a vector of times of rate shift
	f = file(f_name,'U')
	if nbins==0:
		nbins = abs(int(max_age-min_age))
	post_rate=f.readlines()
	if present_year == -1: 
		bins_histogram = np.linspace(min_age,max_age,nbins+1)	
	else:
		bins_histogram = np.linspace(max_age,min_age,nbins+1)	
	marginal_rates_list = []
	times_of_shift = []
	
	if burnin<1: # define burnin as a fraction
		burnin=min(int(burnin*len(post_rate)),int(0.9*len(post_rate)))
	
	for i in range(burnin,len(post_rate)):
		row = np.array(post_rate[i].split()).astype(float)
		
		if len(row)==1: 
			marginal_rates = np.zeros(nbins)+row[0]
		else:
			ind_rates = np.arange(0,int(np.ceil(len(row)/2.)))
			ind_shifts = np.arange(int(np.ceil(len(row)/2.)),len(row))
			rates = row[ind_rates]
			if present_year == -1: 
				shifts = row[ind_shifts]
				h = np.histogram(shifts,bins =bins_histogram)[0][::-1]
				marginal_rates = rates[np.cumsum(h)]
			else: 
				shifts = present_year-row[ind_shifts]
				h = np.histogram(shifts,bins =bins_histogram)[0]
				marginal_rates = rates[np.cumsum(h)][::-1]
			
			#print rates, marginal_rates, shifts,bins_histogram
			#quit()
			times_of_shift += list(shifts)
		
			marginal_rates_list.append(marginal_rates)
	
	marginal_rates_list = np.array(marginal_rates_list)
	mean_rates= np.mean(marginal_rates_list,axis=0)
	min_rates,max_rates=[],[]
	for i in range(nbins):
		hpd = calcHPD(marginal_rates_list[:,i],0.95)
		min_rates += [hpd[0]]
		max_rates += [hpd[1]]
	
	time_frames = bins_histogram-abs(bins_histogram[1]-bins_histogram[0])/2.
	#print rates, marginal_rates, 
	#print shifts, h
	#quit()
	#print time_frames
	#quit()
	time_frames = time_frames[1:]
	#print len(time_frames),len(mean_rates), 
	n_mcmc_samples = len(post_rate)-burnin # number of samples used to normalize frequencies of rate shifts
	return [time_frames,mean_rates,np.array(min_rates),np.array(max_rates),np.array(times_of_shift),n_mcmc_samples]


def get_r_plot(res,col,parameter,min_age,max_age,plot_title,plot_log,run_simulation=1):
	out_str = "\n"
	if present_year == -1: 
		out_str += print_R_vec("\ntime",-res[0])
		minXaxis,maxXaxis= -max_age,-min_age
		time_lab = "BP"
	else:
		out_str += print_R_vec("\ntime",res[0])
		minXaxis,maxXaxis= max_age,min_age
		time_lab = "AD"
	out_str += print_R_vec("\nrate",res[1][::-1])
	out_str += print_R_vec("\nminHPD",res[2][::-1])
	out_str += print_R_vec("\nmaxHPD",res[3][::-1])
	if plot_log==0:
		out_str += "\nplot(time,time,type = 'n', ylim = c(%s, %s), xlim = c(%s,%s), ylab = '%s', xlab = 'Time (%s)',main='%s' )" \
			% (0,1.1*np.nanmax(res[3]),minXaxis,maxXaxis,parameter,time_lab,plot_title) 
		out_str += "\npolygon(c(time, rev(time)), c(maxHPD, rev(minHPD)), col = alpha('%s',0.3), border = NA)" % (col)
		out_str += "\nlines(time,rate, col = '%s', lwd=2)" % (col)
	else:
		out_str += "\nplot(time,time,type = 'n', ylim = c(%s, %s), xlim = c(%s,%s), ylab = 'Log10 %s', xlab = 'Time (%s)',main='%s' )" \
			% (np.nanmin(np.log10(0.9*res[2])),np.nanmax(np.log10(1.1*res[3])),minXaxis,maxXaxis,parameter,time_lab,plot_title) 
		out_str += "\npolygon(c(time, rev(time)), c(log10(maxHPD), rev(log10(minHPD))), col = alpha('%s',0.3), border = NA)" % (col)
		out_str += "\nlines(time,log10(rate), col = '%s', lwd=2)" % (col)
		
	# add barplot rate shifts
	if present_year == -1: bins_histogram = np.linspace(min_age,max_age,len(res[0]))
	else: bins_histogram = np.linspace(max_age,min_age,len(res[0]))
	if len(res[4])>1: # rate shift sampled at least once
		h = np.histogram(res[4],bins =bins_histogram) #,density=1)
	else:
		h = [np.zeros(len(bins_histogram)-1),bins_histogram]
	a = h[1]
	mids = (a-abs(a[1]-a[0])/2.)[1:]
	if present_year == -1: out_str += print_R_vec("\nmids",-mids)
	else: out_str += print_R_vec("\nmids",mids)
	out_str += print_R_vec("\ncounts",h[0]/float(res[5]))
	out_str += "\nplot(mids,counts,type = 'h', xlim = c(%s,%s), ylim=c(0,%s), ylab = 'Frequency of rate shift', xlab = 'Time (%s)',lwd=5,col='%s')" \
	    % (minXaxis,maxXaxis,max(max(h[0]/float(res[5])),0.2),time_lab,col)
	# get BFs
	if run_simulation==1:
		BFs = get_prior_shift(min_age,max_age,bins_histogram)
		out_str += "\nbf2 = %s\nbf6 = %s" % (BFs[1],BFs[2])
	out_str += "\nabline(h=bf2, lty=2)"
	out_str += "\nabline(h=bf6, lty=2)"
	return out_str

def get_K_values(mcmc_tbl,head,col,par,burnin=0.2):
	burnin=min(int(burnin*len(mcmc_tbl)),int(0.9*len(mcmc_tbl)))
	post_tbl = mcmc_tbl[burnin:,:]
	h1 = head.index("K_l")
	h2 = head.index("K_m")
	if par=="l": h = h1
	else: h = h2
	print h
	unique, counts = np.unique(post_tbl[:,h], return_counts=True)
	print unique, counts
	out_str  = print_R_vec("\nunique",unique)
	out_str += print_R_vec("\ncounts",counts)
	out_str += "\nplot(unique,counts,type = 'h', xlim = c(0,%s), ylab = 'Frequency', xlab = 'n. shifts',lwd=5,col='%s')" \
	    % (np.max(post_tbl[:,np.array([h1,h2])])+1,col)
	return out_str
	

def plot_marginal_rates(path_dir,name_tag="",bin_size=1.,burnin=0.2,min_age=0,max_age=0,logT=0):
	direct="%s/*%s*mcmc.log" % (path_dir,name_tag)
	files=glob.glob(direct)
	files=np.sort(files)
	stem_file=files[0]
	wd = "%s" % os.path.dirname(stem_file)
	#print(name_file, wd)
	print "found", len(files), "log files...\n"
	if logT==1: outname = "Log_"
	else: outname = ""
	if max_age>0: outname+= "t%s" % (int(max_age))
	r_str = "\n\npdf(file='%s/%sRTT_plots.pdf',width=12, height=8)\npar(mfrow=c(2,3))\nlibrary(scales)" % (wd,outname)
	for mcmc_file in files:
		if 2>1: #try:
			name_file = os.path.splitext(os.path.basename(mcmc_file))[0]		
			if min_age==0 and max_age==0: # get empirical time range
				tbl=np.loadtxt(mcmc_file, skiprows=1)
				head = next(open(mcmc_file)).split() # should be faster
				if present_year == -1:
					max_age_t = np.mean(tbl[:,head.index("root_age")])
					min_age_t = np.mean(tbl[:,head.index("death_age")])
				else:
					max_age_t = present_year-np.mean(tbl[:,head.index("root_age")])
					min_age_t = present_year-np.mean(tbl[:,head.index("death_age")])
			else:
				min_age_t, max_age_t = min_age, max_age
			nbins = int(abs(max_age_t-min_age_t)/float(bin_size))
			colors = ["#4c4cec","#e34a33"] # sp and ex rate
			# sp file
			r_str += get_K_values(tbl,head,colors[0],"l",burnin=0.2)
			f_name = mcmc_file.replace("mcmc.log","sp_rates.log")
			res = get_marginal_rates(f_name,min_age_t,max_age_t,nbins,burnin=0.2)
			r_str += get_r_plot(res,col=colors[0],parameter="Speciation rate",min_age=min_age_t,max_age=max_age_t,plot_title=name_file,plot_log=logT)
			# ex file
			r_str += get_K_values(tbl,head,colors[1],"m",burnin=0.2)
			f_name = mcmc_file.replace("mcmc.log","ex_rates.log")
			res = get_marginal_rates(f_name,min_age_t,max_age_t,nbins,burnin=0.2)
			r_str += get_r_plot(res,col=colors[1],parameter="Extinction rate",min_age=min_age_t,max_age=max_age_t,plot_title="",plot_log=logT,run_simulation=0)
		#except:
		#	print "Could not read file:", mcmc_file
	r_str += "\n\nn <- dev.off()"
	out="%s/%sRTT_plots.r" % (wd,outname)
	outfile = open(out, "wb") 
	outfile.writelines(r_str)
	outfile.close()
	cmd="cd %s; Rscript %sRTT_plots.r" % (wd,outname)
	print "Plots saved in %s (%sRTT_plots)" % (wd,outname)
	os.system(cmd)




p = argparse.ArgumentParser() #description='<input file>') 
p.add_argument('input_data', metavar='<path to log files>', type=str,help='Input python file - see template',default="")
p.add_argument('-logT', metavar='1', type=int,help='set to 1 to log transform rates',default=0)
p.add_argument('-present_year',    type=int, help='set to > present AD to plot in time AD instead of time BP', default= -1, metavar= -1)

args = p.parse_args()
path_dir_log_files = args.input_data
present_year = args.present_year

plot_marginal_rates(path_dir_log_files,logT=args.logT)
