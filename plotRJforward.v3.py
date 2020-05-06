#!/usr/bin/env python
from numpy import *
import numpy as np
import os,platform,glob,sys
import csv
import argparse
import pandas as pd

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
    print( "\nComputing empirical priors on rate shifts...")
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
    return [prior_s,bf2,bf6]


def get_marginal_rates(f_name,start_age,end_age,nbins=0,burnin=0.2):
    # returns a list of 5 items:
    # 1. a vector of times (age of each marginal rate)
    # 2-4. mean, min and max marginal rates (95% HPD)
    # 5. a vector of times of rate shift
    # 6. the marginal rates (used to compute net diversification rate)
    #f = file(f_name,'U')
    f = open(f_name,'r')
    if nbins==0:
        nbins = abs(int(end_age-start_age))
    post_rate=f.readlines()
    bins_histogram = np.arange(end_age,start_age+1)
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

            shifts = row[ind_shifts]
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
    time_frames = time_frames[1:]
    n_mcmc_samples = len(post_rate)-burnin # number of samples used to normalize frequencies of rate shifts
    return [time_frames,mean_rates,np.array(min_rates),np.array(max_rates),np.array(times_of_shift),n_mcmc_samples,marginal_rates_list]


def get_r_plot(res,col,parameter,min_age,max_age,plot_title,plot_log,run_simulation=1,BFs=None):
    prefix=parameter.split()[0].lower()
    out_str = "\n#%s Plot" % parameter
    if TBP == True:
        out_str += print_R_vec("\ntime",res[0]-min_age)
        minXaxis,maxXaxis= max_age-min_age,min_age-min_age
        time_lab = "BP"
    else:
        out_str += print_R_vec("\ntime",res[0])
        minXaxis,maxXaxis= max_age,min_age
        time_lab = "AD"
    out_str += print_R_vec("\n"+prefix+"_rate",res[1][::-1])
    out_str += print_R_vec("\n"+prefix+"_minHPD",res[2][::-1])
    out_str += print_R_vec("\n"+prefix+"_maxHPD",res[3][::-1])
    if plot_log==0:
        out_str += "\nplot(time,time,type = 'n', ylim = c(%s, %s), xlim = c(%s,%s), ylab = '%s', xlab = 'Time (%s)',main='%s' )" \
                % (0,1.1*np.nanmax(res[3]),minXaxis,maxXaxis,parameter,time_lab,plot_title)
        out_str += "\npolygon(c(time, rev(time)), c({0}_maxHPD, rev({0}_minHPD)), col = alpha('{1}',0.3), border = NA)".format(prefix,col)
        out_str += "\nlines(time,{0}_rate, col = '{1}', lwd=2)".format(prefix,col)
    else:
        out_str += "\nplot(time,time,type = 'n', ylim = c(%s, %s), xlim = c(%s,%s), ylab = 'Log10 %s', xlab = 'Time (%s)',main='%s' )" \
                % (np.nanmin(np.log10(0.9*res[2])),np.nanmax(np.log10(1.1*res[3])),minXaxis,maxXaxis,parameter,time_lab,plot_title)
        out_str += "\npolygon(c(time, rev(time)), c(log10({0}_maxHPD), rev(log10({0}_minHPD))), col = alpha('{1}',0.3), border = NA)".format(prefix,col)
        out_str += "\nlines(time,log10({0}_rate), col = '{1}', lwd=2)".format(prefix,col)

    # add barplot rate shifts
    bins_histogram = np.arange(max_age,min_age+1)
    if len(res[4])>1: # rate shift sampled at least once
        h = np.histogram(res[4],bins =bins_histogram) #,density=1)
    else:
        h = [np.zeros(len(bins_histogram)-1),bins_histogram]
    a = h[1]
    #mids = (a-abs(a[1]-a[0])/2.)[1:]
    out_str+="\n#Frequency of shifts"
    #if TBP==True: out_str += print_R_vec("\nmids",-mids[::-1])
    #else: out_str += print_R_vec("\nmids",mids)
    counts=h[0]/float(res[5])
    out_str += print_R_vec("\n"+prefix+"_counts",counts)
    out_str += "\nplot(time,%s_counts,type = 'h', xlim = c(%s,%s), ylim=c(0,%s), ylab = 'Frequency of rate shift', xlab = 'Time (%s)',lwd=5,col='%s')" \
        % (prefix,minXaxis,maxXaxis,max(max(h[0]/float(res[5])),0.2),time_lab,col)
    # get BFs
    if run_simulation==1:
        BFs = get_prior_shift(min_age,max_age,bins_histogram)
        out_str += "\nbf2 = %s\nbf6 = %s" % (BFs[1],BFs[2])
    out_str += "\nabline(h=bf2, lty=2)"
    out_str += "\nabline(h=bf6, lty=2)"
    counts_bf2 = h[0]/float(res[5])
    counts_bf2[(counts_bf2>=BFs[1]) & (counts_bf2< BFs[2])]=1
    counts_bf6 = h[0]/float(res[5])
    counts_bf6[(counts_bf6>= BFs[2])]=1
    counts_bf2[counts_bf2!=1]=None; counts_bf2=counts_bf2*res[1][::-1]
    counts_bf6[counts_bf6!=1]=None; counts_bf6=counts_bf6*res[1][::-1]
    out_str+=print_R_vec("\n"+prefix+"_BF2",counts_bf2)
    out_str+=print_R_vec("\n"+prefix+"_BF6",counts_bf6)
    return out_str, BFs

def pretty_ggplot(resS,resE,div_log):
    tbl=np.loadtxt(div_log, skiprows=1)
    emp_birth=tbl[:,0]/tbl[:,2]; emp_birth[0]=None
    emp_death=tbl[:,1]/tbl[:,2]; emp_death[0]=None
    
    out_str = "\n\n#PRETTY PLOT\n"
    out_str+="usePackage <- function(p){\nif (!is.element(p, installed.packages()[,1]))\ninstall.packages(p, dep = TRUE)\nrequire(p, character.only = TRUE)}\n"
    out_str+="usePackage('ggplot2')\n"
    out_str += print_R_vec("\nemp_birth",emp_birth)
    out_str += print_R_vec("\nemp_death",emp_death)
    min_Yaxis=min(0,np.nanmin(resS[2]),np.nanmin(resE[2]),np.nanmin(emp_birth),np.nanmin(emp_death)) // 0.1 * 0.1
    max_Yaxis=max(np.nanmax(resS[3]),np.nanmax(resE[3]),np.nanmax(emp_birth),np.nanmax(emp_death))*1.1 // 0.1 * 0.1
    out_str+= "\nrates.dat=data.frame(time,emp_birth,emp_death,birth_rate,birth_minHPD,birth_maxHPD,birth_BF2,birth_BF6,death_minHPD,death_maxHPD,death_BF2,death_BF6)"
    out_str+="\nggplot(rates.dat, aes(time,birth_rate)) +\n\
    geom_line(size=.7,col='blue') +\n\
    geom_line(aes(time,death_rate),col='red',size=.7) +\n \
    scale_color_manual(values = c('blue','red')) +\n\
    scale_x_continuous(breaks=seq(time[1]-time[1]%%-5,time[length(time)]-time[length(time)]%%-5,5),minor_breaks=seq(time[1]-time[1]%%-5,time[length(time)]-time[length(time)]%%-5,1)) +\n\
    scale_y_continuous(breaks=seq({0},{1},.1),limits = c({0},{1})) + \n\
    geom_ribbon(aes_string(ymin=birth_minHPD,ymax=birth_maxHPD,fill=shQuote('red')),alpha=.2,col=NA)+ \n\
    geom_ribbon(aes_string(ymin=death_minHPD,ymax=death_maxHPD,fill=shQuote('blue')),alpha=.2,col=NA)+\n\
    geom_line(aes(time,emp_birth,col='eb'),size=.5,linetype = 'dashed')+\n\
    geom_line(aes(time,emp_death,col='ed'),size=.5,linetype = 'dashed')+\n\
    geom_point(aes(time,birth_BF2), size = 4, alpha=1,col='yellow') +\n\
    geom_point(aes(time,birth_BF6), size = 8, alpha=1,col='yellow') +\n\
    geom_point(aes(time,death_BF2), size = 4, alpha=1,col='yellow') +\n\
    geom_point(aes(time,death_BF6), size = 8, alpha=1,col='yellow') +\n\
    theme(legend.position = 'none')+\n\
    labs(x='Time',y='Rates')\n\n\n".format(min_Yaxis,max_Yaxis)
    
    return out_str

def plot_net_rate(resS,resE,col,min_age,max_age,plot_title,burnin=.2):
    #computes and plots net RATES
    resS_marginal_rate = resS[6]
    resE_marginal_rate = resE[6]
    # in case they have different number of samples
    max_indx = np.min([resS_marginal_rate.shape[0], resE_marginal_rate.shape[0]])
    marginal_rates_list	= resS_marginal_rate[0:max_indx,:] - resE_marginal_rate[0:max_indx,:]
    mean_rates= np.mean(marginal_rates_list,axis=0)
    min_rates,max_rates=[],[]
    nbins = abs(int(max_age-min_age))
    for i in range(nbins):
        hpd = calcHPD(marginal_rates_list[:,i],0.95)
        min_rates += [hpd[0]]
        max_rates += [hpd[1]]

    out_str = "\n#Net Rate"
    
    
    
    if TBP == True:
        #out_str += print_R_vec("\ntime",resS[0]-min_age)
        minXaxis,maxXaxis= max_age-min_age,min_age-min_age
        time_lab = "BP"
    else:
        #out_str += print_R_vec("\ntime",resS[0])
        minXaxis,maxXaxis= max_age,min_age
        time_lab = "AD"
    #right now I don't have support for log, but I think this is less likely to be needed for net rates
    out_str += print_R_vec("\nnet_rate",mean_rates[::-1])
    out_str += print_R_vec("\nnet_minHPD",np.array(min_rates[::-1]))
    out_str += print_R_vec("\nnet_maxHPD",np.array(max_rates[::-1]))
    out_str += "\nplot(time,time,type = 'n', ylim = c(%s, %s), xlim = c(%s,%s), ylab = 'Net Rate', xlab = 'Time (%s)',lwd=2, main='%s', col= '%s' )" \
                    % (min(0,1.1*np.nanmin(min_rates)),1.1*np.nanmax(max_rates),minXaxis,maxXaxis,time_lab,plot_title,col)
    out_str += "\npolygon(c(time, rev(time)), c(net_maxHPD, rev(net_minHPD)), col = alpha('%s',0.3), border = NA)" % (col)
    out_str += "\nlines(time,net_rate, col = '%s', lwd=2)" % (col)
    out_str += "\nabline(h=0,lty=2)\n"

    return out_str

def plot_net_diversity(div_log,resS,col,min_age,max_age,plot_title):
    out_str = "\n#Net Diversity"
    if TBP == True:
        #out_str += print_R_vec("\ntime",resS[0]-min_age)
        minXaxis,maxXaxis= max_age-min_age,min_age-min_age
        time_lab = "BP"
    else:
        #out_str += print_R_vec("\ntime",resS[0])
        minXaxis,maxXaxis= max_age,min_age
        time_lab = "AD"
        #plot net_diversity

    tbl=np.loadtxt(div_log, skiprows=1)
    net_div=tbl[:,2]
    out_str += print_R_vec("\nnet_diversity",net_div)
    out_str += "\nplot(time,net_diversity,type = 'l', ylab = 'Net Diversity', xlab = 'Time (%s)',lwd=2, main='%s', col= '%s' )\n" \
                    % (time_lab,plot_title,col)
    return out_str


def get_K_values(mcmc_tbl,head,col,par,burnin=0.2):
    burnin=min(int(burnin*len(mcmc_tbl)),int(0.9*len(mcmc_tbl)))
    post_tbl = mcmc_tbl[burnin:,:]
    h1 = head.index("K_l")
    h2 = head.index("K_m")
    if par=="l": h = h1
    else: h = h2
    unique, counts = np.unique(post_tbl[:,h], return_counts=True)
    out_str = '\n#Number Shifts'
    out_str += print_R_vec("\nunique",unique)
    out_str += print_R_vec("\ncounts",counts)
    out_str += "\nplot(unique,counts,type = 'h', xlim = c(0,%s), ylab = 'Frequency', xlab = 'n. shifts',lwd=5,col='%s')" \
        % (np.max(post_tbl[:,np.array([h1,h2])])+1,col)
    return out_str

def combine_logs(mcmc_files, wd, burnin_pct):
    #MCMC (w/header)
    mcmc_files=list(mcmc_files)
    total_log=[]
    for file_name in mcmc_files:
        with open(file_name) as f:
            file_log=f.readlines()
            header=file_log[0]
            burnin=int(burnin_pct*len(file_log[1:]))
            total_log+=file_log[burnin+1:]
    with open(wd+'/COMBINED_mcmc.log','w') as o:
        o.write(header)
        it_bool= (header.split('\t')[0]=='it')
        for i,l in enumerate(total_log):
            l=l.split('\t')
            l[0]=str(i)
            o.write('\t'.join(l))
    #SP (no header)
    total_log=[]
    for file_name in mcmc_files:
        file_name=file_name.replace('mcmc.log','sp_rates.log')
        with open(file_name) as f:
            file_log=f.readlines()
            burnin=int(burnin_pct*len(file_log))
            total_log+=file_log[burnin:]
    with open(wd+'/COMBINED_sp_rates.log','w') as o: o.writelines(total_log)
    #EX (no header)
    total_log=[]
    for file_name in mcmc_files:
        file_name=file_name.replace('mcmc.log','ex_rates.log')
        with open(file_name) as f:
            file_log=f.readlines()
            burnin=int(burnin_pct*len(file_log))
            total_log+=file_log[burnin:]
    with open(wd+'/COMBINED_ex_rates.log','w') as o: o.writelines(total_log)
    #DIV
    sp_events=[];ex_events=[];br_length=[]
    for file_name in mcmc_files:
        file_name=file_name.replace('mcmc.log','div.log')
        div=np.loadtxt(file_name,skiprows=1)
        sp_events.append(div[:,0]);ex_events.append(div[:,1]);br_length.append(div[:,2])
    sp_events=np.mean(np.array(sp_events),axis=0);ex_events=np.mean(np.array(ex_events),axis=0);br_length=np.mean(np.array(br_length),axis=0)
    combined_div=pd.DataFrame({'sp_events':sp_events,'ex_events':ex_events,'br_length':br_length})
    combined_div[['sp_events','ex_events','br_length']].to_csv(wd+'/COMBINED_div.log',sep='\t',index=False)

def plot_marginal_rates(path_dir,name_tag="",bin_size=1.,burnin=0.2,min_age=0,max_age=0,logT=0,combine=0):
    #FIRST CLEAR COMBIEND FILES
    direct="%s/*%s*mcmc.log" % (path_dir,name_tag) 
    files=glob.glob(direct)
    files.sort()
    stem_file=files[0]
    stem=stem_file.replace('_mcmc.log','')
    wd = "%s" % os.path.dirname(stem_file)
    #print(name_file, wd)
    if wd+'/COMBINED_mcmc.log' in files: files.remove(wd+'/COMBINED_mcmc.log')
    print( "found", len(files), "log files...\n")
    if logT==1: outname = stem="_log_"
    else: outname = stem
    if max_age>0: outname+= "t%s" % (int(max_age))
    if combine==1:
        print("Combining directory to 1 log...\n")
        combine_logs(files,wd,burnin)
        files=[wd+'/COMBINED_mcmc.log']
        burnin=0
        outname=wd+"/COMBINED"
    r_str = "\n\npdf(file='%s_RTT_plots.pdf',width=12, height=8)\npar(mfrow=c(2,4))\nlibrary(scales)" % (outname)
    for mcmc_file in files:
        if 2>1: #try:
            name_file = os.path.splitext(os.path.basename(mcmc_file))[0]
            if min_age==0 and max_age==0: # get empirical time range
                tbl=np.loadtxt(mcmc_file, skiprows=1)
                head = next(open(mcmc_file)).split() # should be faster
                #if present_year == -1:
                max_age_t = np.mean(tbl[:,head.index("root_age")])
                min_age_t = np.mean(tbl[:,head.index("death_age")])
                #else:
                #       max_age_t = present_year-np.mean(tbl[:,head.index("root_age")])
                #       min_age_t = present_year-np.mean(tbl[:,head.index("death_age")])
            else:
                min_age_t, max_age_t = min_age, max_age
            nbins = int(abs(max_age_t-min_age_t)/float(bin_size))
            colors = ["#4c4cec","#e34a33","#32CD32"] # sp and ex rate
            # sp file
            r_str += "\n###%s###\n" % name_file
            r_str += get_K_values(tbl,head,colors[0],"l",burnin=burnin)

            f_name = mcmc_file.replace("mcmc.log","sp_rates.log")
            resS = get_marginal_rates(f_name,min_age_t,max_age_t,nbins,burnin=burnin)
            spec_str,BFs=get_r_plot(resS,col=colors[0],parameter="Birth rate",min_age=min_age_t,max_age=max_age_t,plot_title=name_file,plot_log=logT)
            r_str+=spec_str
            # ex file

            f_name = mcmc_file.replace("mcmc.log","ex_rates.log")
            resE = get_marginal_rates(f_name,min_age_t,max_age_t,nbins,burnin=burnin)

            #net rate
            r_str += plot_net_rate(resS,resE,col=colors[2],min_age=min_age_t,max_age=max_age_t,plot_title='')

            #ex rates
            r_str += get_K_values(tbl,head,colors[1],"m",burnin=burnin)
            ex_str,_ = get_r_plot(resE,col=colors[1],parameter="Death rate",min_age=min_age_t,max_age=max_age_t,plot_title="",plot_log=logT,run_simulation=0,BFs=BFs)
            r_str+=ex_str

            #net div
            f_name = mcmc_file.replace("mcmc.log","div.log")
            r_str += plot_net_diversity(f_name,resS,col=colors[2],min_age=min_age_t,max_age=max_age_t,plot_title='')
            
            #pretty plot
            r_str += pretty_ggplot(resS,resE,f_name)

            
        #except:
        #       print "Could not read file:", mcmc_file
    r_str += "\n\nn <- dev.off()"
    out="%s_RTT_plots.r" % (outname)
    outfile = open(out, "w")
    outfile.writelines(r_str)
    outfile.close()
    cmd="Rscript %s_RTT_plots.r" % (outname)
    print( "Plots saved in %s (%s_RTT_plots)" % (wd,outname))
    os.system(cmd)




p = argparse.ArgumentParser() #description='<input file>')
p.add_argument('input_data', metavar='<path to log files>', type=str,help='Input python file - see template',default="")
p.add_argument('-combine', metavar='0',type=int,help="Whether log files should be combined before plotting or plotted serially")
p.add_argument('-logT', metavar='1', type=int,help='set to 1 to log transform rates',default=0)
p.add_argument('-burnin', metavar='.2',type=float,help='set as a fraction',default=.2)
p.add_argument('-TBP', help='Default is AD. Include for TBP.', default=False, action='store_true')

args = p.parse_args()
path_dir_log_files = args.input_data
TBP=args.TBP
plot_marginal_rates(path_dir_log_files,logT=args.logT,burnin=args.burnin,combine=args.combine)
