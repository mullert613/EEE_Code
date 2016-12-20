import sys
import pandas as pd
import numpy
#import pylab
import unittest
import scipy.stats
import joblib
import MCMC

def get_birdcounts_sample(datafile,loglikelihood,poly_deg,dof=4,pc=4,maxruns=50000):
	bc = pd.read_csv(datafile,index_col=0)
	#bc = bc.drop("Total",axis=1)
	N = len(bc)
	lbd = 10./numpy.sqrt((poly_deg+1)*(N))
	sigma= numpy.eye((poly_deg+1)*N)
	time = numpy.array([int(x) for x in bc.columns])
	bc_mat = bc.as_matrix()
	dof=4
	pc=4
	init_guess = bc_init_guess(datafile,poly_deg)
	p_vals = numpy.zeros((N,len(time)))
	amean = numpy.zeros(N)
	bmean = numpy.zeros(N)
	coeff_mean = numpy.zeros((N,poly_deg+1))
	coeff = []
	for i in range(N):
		loglikargs = (bc_mat[i],time)
		theta_bc,k_bc=MCMC.run_MCMC_convergence(init_guess[i,:],loglikelihood,loglikargs,maxruns=maxruns,pc=4,method="Nelder-Mead")
		n=len(bc_mat[i])
		results = theta_bc[0][500::100]
		#coeff = numpy.zeros((len(results),poly_deg+1))
		coeff.append(numpy.zeros((len(results),poly_deg+1)))
		for j in range(poly_deg+1):
			#coeff[:,j,numpy.newaxis] = results[:,j::poly_deg+1]
			coeff[-1][:,j,numpy.newaxis]=results[:,j::poly_deg+1]
		#coeff_mean[i,:] = numpy.mean(coeff,axis=0)
		coeff_mean[i,:] = numpy.mean(coeff[-1],axis=0)	
	return(coeff_mean,coeff)

def birdcounts_function(coeff_mat,time):  #Currently as defined works for single input time
	q = numpy.polyval(coeff_mat.T,time)
	p = numpy.exp(q)
	return(p)

def bc_init_guess(datafile,poly_deg):
	bc = pd.read_csv(datafile,index_col=0)
	#bc = bc.drop("Total",axis=1)
	N = len(bc)
	time = numpy.array([int(x) for x in bc.columns])
	A = numpy.column_stack([time**n for n in reversed(range(poly_deg+1))]) 
	b = numpy.log(bc.as_matrix()+.01)
	x = numpy.linalg.lstsq(A,b.T)
	return(x[0].T)

def birdcounts_derivative(coeff_mat,time):
	q = numpy.polyval(coeff_mat.T,time)
	p = numpy.exp(q)
	der = numpy.array([numpy.polyder(x) for x in coeff_mat])
	der = numpy.polyval(der.T,time)
	return(der*p)

def xi(coeff_mat,time):			#Returns the positive piece of the birdcount derivative, or 0.
	return(numpy.clip(birdcounts_derivative(coeff_mat,time),0,numpy.inf))

def bc_out(coeff_mat,time):
	val = -numpy.clip(birdcounts_derivative(coeff_mat,time),-numpy.inf,0)
	return(val)

def bc_in(coeff_mat,time):
	val = numpy.clip(birdcounts_derivative(coeff_mat,time),0,numpy.inf)
	return(val)


def Xi(coeff,t):
	val = numpy.where(birdcounts_function(coeff,t)>0,numpy.ma.divide(bc_in(coeff,t),birdcounts_function(coeff,t)),0)
	return(val)


def F(coeff_mat,time):            #Returns the negative piece of the birdcount derivative, or 0.
	return(-numpy.clip(birdcounts_derivative(coeff_mat,time),-numpy.inf,0))	


def BirdcountTest(datafile,coeff_mat,poly_deg):
	import pylab
	bc = pd.read_csv(datafile,index_col=0)
	birdnames = pd.read_csv(datafile,index_col=0).index
	#bc = bc.drop("Total",axis=1)
	bc_mat = bc.as_matrix()
	N = len(bc)	
	time = numpy.array([int(x) for x in bc.columns],dtype=float)
	q = numpy.zeros((N,len(time)))
	for j in range(0,poly_deg+1):
		q += numpy.dot(coeff_mat[:,j,numpy.newaxis],(time**(poly_deg-j))[:,numpy.newaxis].T)
	p=numpy.exp(q)

	for i in range(N):
		pylab.subplot(3,3,i+1)
		pylab.title('%s with fit degree %d' %(birdnames[i],poly_deg))
		mat=numpy.array(bc_mat[i])
		pylab.scatter(time,mat.astype(float))
		pylab.plot(time,p[i,:])
	
	return()

