import pandas as pd
import numpy
import unittest
import scipy.stats
import joblib

# 
def example():   # Simple run,sigma,lbd,theta_0 values to be used in test code
	runs = 10000
	sigma= numpy.eye(1)
	lbd =2.4
	theta_0 = numpy.ones(1)
	return theta_0,sigma,lbd,runs

def get_DIC(results,loglikelihood,datafile):
	data = pd.read_csv(datafile,index_col=0)
	time = numpy.array([int(x) for x in data.columns])
	data_mat = data.as_matrix()
	p = len(results)
	DIC_vals=numpy.zeros(p)
	for j in range(p):
		DIC_vals[j]=DIC(results[j],loglikelihood,data_mat[j],time)
	return(DIC_vals)



def DIC(results,loglikelihood,data,times):
	D_bar = numpy.mean([-2*loglikelihood(row,data,times) for row in results])
	D_theta_bar = -2*loglikelihood(numpy.mean(results,axis=0),data,times)
	return 2*D_bar-D_theta_bar


def svals(*args,**kwargs):  # a function, used in test functions, to return the starting values for the code
	#Use this function to generate the input values
	runs = 10000
	sigma= numpy.eye(1)
	lbd = 2.4
	theta_0 = numpy.ones(1)
	return theta_0,sigma,lbd,runs

def logit(x):	#Used in MCMC
	return numpy.log(x)-numpy.log(1-x)
def invlogit(x):			#Used in MCMC  scipy.expit
	return 1/(1+numpy.exp(-x))

def bin_log_lik(theta_0,n,data):		#Loglikelihood for the binomial r.v.
	p = scipy.special.expit(theta_0)
	return data*numpy.log(p)+(n-data)*numpy.log(1-p)     # This is the given loglikelihood function

def poiss_log_lik(theta_0,data):
	return numpy.sum(data*numpy.log(theta_0)-theta_0)

def lin_poiss_log_lik(theta_0,data,times):
	a = theta_0[0]
	b = theta_0[1]
	#a = theta_0[0::2]
	#b = theta_0[1::2]
	#q = a[:,numpy.newaxis]*times[numpy.newaxis,:]+b[:,numpy.newaxis]
	q = a*times+b  #  log lambda = a*t+b = q
	n = len(data)
	logp = numpy.sum(data*q-numpy.exp(q))
	return logp

def poly_poiss_log_lik(theta_0,data,times):
	q = numpy.polyval(theta_0,times)
	logp = numpy.sum(data*q - numpy.exp(q))
	return logp

def quad_test(theta_0,data,times):
	a = theta_0[0]
	b = theta_0[1]
	c = theta_0[2]
	q = a*times**2+b*times+c
	n = len(data)
	logp = numpy.sum(data*q - numpy.exp(q))
	return logp	

def logtest(x):  #Test LogLikelihood function for the Laplace distribution
	a=0
	b=1
	return -numpy.log(2*b)-numpy.abs(x-a)/b
#The loglikelihood function to be fixed

def loglikelihood(theta_0,bloodmealcounts,bloodmealtimes):  #loglikelihood for the bloodmeals
	a = theta_0[0::2]
	b = theta_0[1::2]
	q = a[:,numpy.newaxis]*bloodmealtimes[numpy.newaxis,:]+b[:,numpy.newaxis] #First axis species, second time
	#p = numpy.vstack((numpy.exp(q)/(1+numpy.sum(numpy.exp(q),0)),1/(1+numpy.sum(numpy.exp(q),0))))	#Multinomial regression reference
	logp = numpy.vstack((q-numpy.log(1+numpy.sum(numpy.exp(q),0)),-numpy.log(1+numpy.sum(numpy.exp(q),0))))
	#return numpy.sum(numpy.sum(p*(bloodmealcounts-numpy.sum(bloodmealcounts,0)),1))  # The original loglikelihood, which we haven't been able to recreate
	return numpy.sum(numpy.sum(logp*(bloodmealcounts)))  #This sums over the likelihoods of each a,b may be worth examining individually

def bm_polynomial_loglikelihood(theta_0,counts,times):  #Proposed updated loglikelihood for bloodmeals
	param_count = len(counts)-1  # For bm, since the last population is determined from the others, is 1 less than length
	poly_deg = len(theta_0)//param_count-1
	coeff = numpy.zeros((param_count,poly_deg+1))
	for j in range(poly_deg+1):
		coeff[:,j] = theta_0[j::poly_deg+1]
	q = numpy.array([numpy.polyval(val,times) for val in coeff])
	#q = numpy.polyval(theta_0,times) # This is incorrect
	logp = numpy.vstack((q-numpy.log(1+numpy.sum(numpy.exp(q),0)),-numpy.log(1+numpy.sum(numpy.exp(q),0))))
	return numpy.sum(numpy.sum(logp*(counts)))

def gauss_log_lik(mu,sig_square,d):
	return -numpy.sum((d-mu)**2/(2*sig_square))	

def NewMCMC(theta_0,sigma,lbd,runs,loglikelihood,loglikargs=()):
    #args will be a list of stored additional parameters, if a keyword is provided will appear in kwargs dictionary
	k=0
	n=len(theta_0)
	theta=numpy.zeros((runs+1,n))
	theta[0]=theta_0
	loglike_0 = loglikelihood(theta_0,*loglikargs)
	for i in range(0,runs):
		theta_1=numpy.random.multivariate_normal(theta_0,sigma*lbd**2)
		loglike_1 = loglikelihood(theta_1,*loglikargs)
		alpha = numpy.exp(numpy.min((0.,loglike_1-loglike_0)))
		accept = numpy.random.binomial(1,alpha)
		if accept==1:
			k+=1
			theta_0=theta_1
			loglike_0=loglike_1
		theta[i+1]=theta_0	
	return theta,k

def Gelman_Rubin_test(chains):		# Updated Loopless version
	M = numpy.float(len(chains))
	n = numpy.float(len(chains[0]))
	sigma_m_bar = 1./n*numpy.sum(chains, axis=1) 	#mean within chains
	s_m = 1./(n-1)*numpy.sum((chains-sigma_m_bar[:,numpy.newaxis,:])**2,axis=1) # variance within chains	
	sigma_bar = 1./M*numpy.sum(sigma_m_bar,axis=0)	#Total mean across chains
	#B= n/(M-1)*numpy.sum((sigma_m_bar - sigma_bar)**2,axis=0)  # n * Variance of the means 
	B= 1./(M-1)*numpy.sum((sigma_m_bar - sigma_bar)**2,axis=0)  # Variance of the means 
	W= 1./M*numpy.sum(s_m,axis=0)			# Mean of the variances
	V_hat= (n-1)/n*W + (M+1)/(M)*B  	# Changed from (M+1)/(M*n)*B after changes to B formulation
	#print(W,B)
	if numpy.max((V_hat,W))<10**(-16):
		R_hat = numpy.ones(len(chains[0][0]))
	else:
		R_hat=numpy.sqrt(V_hat/W)	
	return R_hat

def matrix_sqrt(A):		# Assumes a square matrix A, include a check for positive definiteness (?)
	U, s , V = numpy.linalg.svd(A, full_matrices=True)
	S = numpy.diag(numpy.sqrt(s))
	return(numpy.dot(U,numpy.dot(S,V)))

def sample_dist(theta_0,pc,loglike,loglikargs=(),dof=4,method = 'BFGS'):  # Calculate a sample distribution from student t tests from the mode 
														  # for a number of parallel chains
	
	def flog(x,loglike,*args):
		return -loglike(x,*args)

	val = scipy.optimize.minimize(flog,theta_0,args=(loglike,)+loglikargs,method = method)
	theta_0_par = val.x
	time = loglikargs[1]
	q = numpy.polyval(theta_0_par,time)
	p = numpy.exp(q)
	poly_deg = len(theta_0_par)-1
	if method=="BFGS":
		hess_inv = val.hess_inv   #In this version, .hess returns the hessian inverse, in updated scipy packages it's .hess_inv
	else:	
		hessian = numpy.zeros((poly_deg+1,poly_deg+1))
		for i in range(0,poly_deg+1):
			for j in range(0,poly_deg+1):
				hessian[i][j] = numpy.sum(time**(2*poly_deg-(i+j))*p)
		hess_inv = numpy.linalg.inv(hessian)		

	A = matrix_sqrt(hess_inv)
	t = scipy.stats.t(dof)
	z = t.rvs(size=(len(theta_0_par),pc))/t.std()
	the_0 = (theta_0_par[:,numpy.newaxis] + numpy.dot(A,z)).T
	return(the_0)



def convergence_update(theta,sigma,accept,lbd,upd,counter,p,loglikelihood,loglikargs=()):
	nparam = numpy.shape(theta)[-1]
	value,k = NewMCMC(theta[-1],sigma,lbd,upd,loglikelihood,loglikargs)
	theta = numpy.vstack(( theta[1:] , value ))
	sigma_star = numpy.cov(value.T)
	sigma = p*sigma + (1-p)*sigma_star
	accept = numpy.append(accept,numpy.float(k)/upd)
	if nparam>1:
		alpha = .23
	else:
		alpha = .44
	#lbd *= numpy.exp((accept[-1]-alpha)/1)		#This line is different than the paper
	lbd *= numpy.exp((accept[-1]-alpha)/(numpy.float(counter)/upd+1))		# Original Line
	return(theta,sigma,accept,lbd)	
	
def run_MCMC_convergence(init_guess,loglik,loglikargs=(),maxruns=10000,pc = 4,**kargs):   # kargs include dof
	the_0 = sample_dist(init_guess,pc,loglik,loglikargs = loglikargs,**kargs)
	theta = [the_0[numpy.newaxis,j] for j in range(pc)]			
	upd = 500  #The number of runs for each step before we update the covariance matrix
	p=.25	# Covariance Matrix Weight

	_,nparam = numpy.shape(the_0)
	accept=[ numpy.array([]) for j in range(pc)]
	sigma= [numpy.eye(nparam) for j in range(pc)]
	lbd  = [1/numpy.sqrt(nparam) for j in range(pc)]
	with joblib.Parallel(n_jobs=-1) as parallel:
		counter=0
		while counter < maxruns:
			# for j in range(pc):
			# 	theta[j],sigma[j],accept[j],lbd[j] = convergence_update(theta[j],sigma[j],accept[j],lbd[j],upd,counter,p,loglik,loglikargs)
			output = parallel(
					joblib.delayed(convergence_update)(theta[j],sigma[j],accept[j],lbd[j],upd,counter,p,loglik,loglikargs)
					for j in range(pc))
			counter += upd
			theta,sigma,accept,lbd = zip(*output)
			R_hat = Gelman_Rubin_test(theta)
			#print R_hat	
			#print accept
			if numpy.all(R_hat<1.10):
				break
		else:
			print("Convergence not reached")	
	return(theta,accept)	


def MCMC_output_calculator(theta,counts,time):
	import pylab
	n=len(counts)
	results = theta[0][500::100]
	a = results[:,0::2]
	b = results[:,1::2]
	a0= numpy.percentile(a,5,axis=0)
	a1= numpy.percentile(a,95,axis=0)
	amean = numpy.mean(a,axis=0)
	b0= numpy.percentile(b,5,axis=0)
	b1= numpy.percentile(b,95,axis=0)
	bmean = numpy.mean(b,axis=0)
	q = amean[:,numpy.newaxis]*time[numpy.newaxis,:]+bmean[:,numpy.newaxis] #First axis species, second time
	totq = a[:,:,numpy.newaxis]*time[numpy.newaxis,numpy.newaxis,:]+b[:,:,numpy.newaxis]
	p = numpy.vstack((numpy.exp(q)/(1+numpy.sum(numpy.exp(q),0)),1/(1+numpy.sum(numpy.exp(q),0))))
	mat = numpy.array(counts)
	for i in range(0,n):
		pylab.subplot(3,3,i+1)
		pylab.scatter(time,mat.astype(float)[i,:]/numpy.sum(mat.astype(float),0))
		pylab.plot(time,p[i,:])
	pylab.show()
	return(p)

def count_output_plotter(p_vals,bc_mat,time):
	import pylab
	for i in range(0,len(p_vals)):
		pylab.subplot(3,3,i+1)
		pylab.scatter(time,numpy.array(bc_mat[i]).astype(float))
		pylab.plot(time,p_vals[i,:])
	pylab.show()