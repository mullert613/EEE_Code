import pandas as pd
import numpy
#import pylab
import unittest
import scipy.stats
import joblib
import MCMC
import BirdCount
import Seasonal_ODE
import warnings

# Changes Made Here, finish this
def get_bloodmeal_sample(datafile,loglikelihood,poly_deg,dof=4,pc=4,maxruns=50000):
	bm = pd.read_csv(datafile,index_col=0)
	#bm=bm.drop("Total",axis=1)
	N=len(bm)
	time = numpy.array([int(x) for x in bm.columns],dtype=float)
	init_guess = numpy.zeros((poly_deg+1)*(N-1))
	loglikargs=(bm.as_matrix(),time)
	#the_0 = MCMC.sample_dist(init_guess,pc,loglikelihood,loglikargs)
	theta_bm,k_bm=MCMC.run_MCMC_convergence(init_guess,loglikelihood,loglikargs=(bm.as_matrix(),time),maxruns=maxruns,pc=4)

	n=len(bm.as_matrix())
	results = theta_bm[0][500::10]
	coeff_mean = numpy.zeros((N-1,poly_deg+1))
	for j in range(poly_deg+1):
		coeff_mean[:,j] = numpy.mean(results[:,j::poly_deg+1],axis=0)
	# a = results[:,0::2]
	# b = results[:,1::2]
	# a0= numpy.percentile(a,5,axis=0)
	# a1= numpy.percentile(a,95,axis=0)
	# amean = numpy.mean(a,axis=0)
	# b0= numpy.percentile(b,5,axis=0)
	# b1= numpy.percentile(b,95,axis=0)
	# bmean = numpy.mean(b,axis=0)

	return(coeff_mean,results)

def get_vector_sample(datafile,loglikelihood,poly_deg,dof=4,pc=4,maxruns=50000):
	vc = pd.read_csv(datafile,index_col=0)
	N = len(vc)
	lbd = 10./numpy.sqrt((poly_deg+1)*(N))
	sigma= numpy.eye((poly_deg+1)*N)
	time = numpy.array([int(x) for x in vc.columns],dtype=float)
	vc_mat = vc.as_matrix()
	dof=4
	pc=4
	init_guess = vc_init_guess(datafile,poly_deg)
	p_vals = numpy.zeros((N,len(time)))
	amean = numpy.zeros(N)
	bmean = numpy.zeros(N)
	#coeff_mean = numpy.zeros((N,poly_deg+1))
	coeff_mean = numpy.zeros((N,poly_deg+1))
	coeff = []
	for i in range(N):
		loglikargs = (vc_mat[i],time)
		theta_vc,k_vc=MCMC.run_MCMC_convergence(init_guess[i,:],loglikelihood,loglikargs,maxruns=maxruns,pc=4,method="Nelder-Mead")
		n=len(vc_mat[i])
		results = theta_vc[0][500::10]
		#coeff = numpy.zeros((len(results),poly_deg+1))
		coeff.append(numpy.zeros((len(results),poly_deg+1)))
		for j in range(poly_deg+1):
			#coeff[:,j,numpy.newaxis] = results[:,j::poly_deg+1]
			coeff[-1][:,j,numpy.newaxis]=results[:,j::poly_deg+1]
		#coeff_mean[i,:] = numpy.mean(coeff,axis=0)
		coeff_mean[i,:] = numpy.mean(coeff[-1],axis=0)
	return(coeff_mean,coeff)

def vc_init_guess(datafile,poly_deg):
	vc = pd.read_csv(datafile,index_col=0)
	#bc = bc.drop("Total",axis=1)
	N = len(vc)
	time = numpy.array([int(x) for x in vc.columns],dtype=float)
	A = numpy.column_stack([time**n for n in reversed(range(poly_deg+1))]) 
	b = numpy.log(vc.as_matrix()+.01)
	x = numpy.linalg.lstsq(A,b.T)
	return(x[0].T)

def bloodmeal_function(coeff,t):  # Currently as defined works for single time value, not vector
	q = numpy.array([numpy.polyval(val,t) for val in coeff])

	logp = numpy.zeros(q.shape)
	
	for j in range(len(q)):
		logp[j] = -numpy.log(numpy.exp(-q[j])+numpy.sum(numpy.exp(q-q[j]),axis=0))
	log_val = numpy.log(1+numpy.sum(numpy.exp(q),0))
	if numpy.ndim(t) == 0:
		logp = numpy.hstack((logp,-log_val))
	else:
		logp = numpy.row_stack((logp,-log_val))
	
	p=numpy.exp(logp)
	return(p)

def vector_coeff(datafile,loglikelihood,poly_deg,maxruns=50000):
	#vc = pd.read_csv(datafile,index_col=0)
	#N = len(vc)
	#time = numpy.array([int(x) for x in vc.columns],dtype=float)
	#vc_mat = vc.as_matrix()
	#coeff = get_vector_sample(datafile,loglikelihood,poly_deg)
	coeff_mean,coeff = get_vector_sample(datafile,loglikelihood,poly_deg,maxruns=maxruns)
	# coeff = numpy.polyfit(time,vc_mat[0,:],2)  # This yields a polynomial fit of 
	#return(coeff[0,:])  # for polyfit, return coeff works fine
	return(coeff_mean[0,:],coeff)

def vector_pop(coeff,t):   #Currently as defined works for single input time
	q = numpy.polyval(coeff.T,t)
	p = numpy.exp(q)
	val = numpy.clip(p,0,numpy.inf)
	return(val)

def vector_derivative(coeff,t):
	q = numpy.polyval(coeff.T,t)
	p = numpy.exp(q)
	val = numpy.where(vector_pop(coeff,t)>0,numpy.polyval(numpy.polyder(coeff),t),0)
	return(val*p)

def vector_in(coeff,t):
	val = numpy.clip(vector_derivative(coeff,t),0,numpy.inf)
	return(val)

def vector_out(coeff,t):
	val = -numpy.clip(vector_derivative(coeff,t),-numpy.inf,0)
	return(val)

def Xi(coeff,t):
	val = numpy.where(vector_pop(coeff,t)>0,numpy.ma.divide(vector_in(coeff,t),vector_pop(coeff,t)),0)
	return(val)

def fun(coeff,t):
	val = numpy.where(vector_pop(coeff,t)>0,numpy.ma.divide(vector_out(coeff,t),vector_pop(coeff,t)),0)
	return(val)

def BloodmealTest(datafile,coeff,poly_deg):
	import pylab
	bm = pd.read_csv(datafile,index_col=0)
	birdnames = pd.read_csv(datafile,index_col=0).index
	#bc = bc.drop("Total",axis=1)
	bm_mat = bm.as_matrix()
	N = len(bm)	
	time = numpy.array([int(x) for x in bm.columns],dtype=float)
	t = numpy.linspace(time[0],time[-1],101)
	p = bloodmeal_function(coeff,t)
	#p=numpy.vstack((numpy.where(numpy.isfinite(numpy.exp(q)),numpy.exp(q)/(1+numpy.sum(numpy.exp(q),0)),1),1/(1+numpy.sum(numpy.exp(q),0))))
	#p = numpy.exp(logp)
	#p = numpy.append(numpy.exp(q)/(1+sum(numpy.exp(q))),1/(1+sum(numpy.exp(q))))

	for i in range(N):
		pylab.subplot(3,3,i+1)
		pylab.title('%s blood meal fit' %(birdnames[i]))
		mat=bm_mat[i].astype(float)/numpy.sum(bm_mat,0)
		pylab.scatter(time,mat)
		pylab.plot(t,p[i,:])
	
	return()

def MosquitoTest(datafile,coeff,poly_deg):
	import pylab
	vc = pd.read_csv(datafile,index_col=0)
	vc_mat = vc.as_matrix()
	time = numpy.array([int(x) for x in vc.columns],dtype=float)
	t = numpy.linspace(time[0],time[-1],101)
	p = vector_pop(coeff,t)
	pylab.title('Vector Fit')
	pylab.scatter(time,vc_mat)
	pylab.plot(t,p)
	return()