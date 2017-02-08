import sys
import pandas as pd
import numpy
#import pylab
import unittest
import scipy.stats
import joblib
import MCMC
import BirdCount
import bloodmeal
import Seasonal_ODE
import pickle
import warnings

#10_10 Notes
# Investigate polynomial fits dependent on bird species (quadratic for some, cubic for others)
# Bloodmeals?
# Can make the order of the polynomial a fittable parameter by the MCMC
# theta = ( --- M1 ---,  --- M2 ---, Model?)  See Gelman book for Model Selection

# 9_23 Notes
# Instead of log transform, consider transform (1+y)

if __name__=='__main__':
	poly_deg = 1
	maxruns = 100000
	#beta1=.07
	bm_file = "Days_BloodMeal.csv"
	bc_file = "Days_BirdCounts.csv"
	#msq_file = "Vector_Data.csv"
	msq_file = "Vector_Data(NoZeros).csv"
	bm_data = pd.read_csv(bm_file,index_col=0)
	bm_time = numpy.array([int(x) for x in bm_data.columns])
	bm_data = bm_data.as_matrix()
	tstart = Seasonal_ODE.time_transform(90) # Setting Start Time to April 1st
	tend = Seasonal_ODE.time_transform(270)
	flag = 0
	write_flag = 1   # If set to 0 will write the results, if set to 1 will not
	if flag==0:  # This section runs the MCMC for the various areas of interest, and stores the results
		mos_coeff,mos_results = bloodmeal.vector_coeff(msq_file,MCMC.poly_poiss_log_lik,poly_deg,maxruns=maxruns)
		bm_coeff_mat,bm_results = bloodmeal.get_bloodmeal_sample(bm_file,MCMC.bm_polynomial_loglikelihood,poly_deg,maxruns=maxruns)
		bc_coeff_mat,bc_results = BirdCount.get_birdcounts_sample(bc_file,MCMC.poly_poiss_log_lik,poly_deg,maxruns=maxruns)
		#BirdCount.BirdcountTest(bc_file,bc_coeff_mat,poly_deg)
		bc_DIC = MCMC.get_DIC(bc_results,MCMC.poly_poiss_log_lik,bc_file)
		bm_DIC = MCMC.DIC(bm_results,MCMC.bm_polynomial_loglikelihood,bm_data,bm_time)
		mos_DIC = MCMC.get_DIC(mos_results,MCMC.poly_poiss_log_lik,msq_file)

		if write_flag ==0:
			with open('Mos_coeff_poly_deg(%d).pkl' %poly_deg, 'wb') as output:
				pickle.dump(mos_coeff,output)

			with open('bloodmeal_coeff_poly_deg(%d).pkl' %poly_deg, 'wb') as output:
				pickle.dump(bm_coeff_mat,output)

			with open('host_coeff_poly_deg(%d).pkl' %poly_deg, 'wb') as output:
				pickle.dump(bc_coeff_mat,output)

			with open('host_coeff_poly_deg(%d)_full_results.pkl' %poly_deg, 'wb') as output:
				pickle.dump(bc_results,output)	

			with open('bloodmeal_coeff_poly_deg(%d)_full_results.pkl' %poly_deg, 'wb') as output:
				pickle.dump(bm_results,output)	
			
			with open('Mos_coeff_poly_deg(%d)_full_results.pkl' %poly_deg, 'wb') as output:
				pickle.dump(mos_results,output)	

			with open('Mos_coeff_poly_deg(%d)_DIC.pkl' %poly_deg, 'wb') as output:
				pickle.dump(mos_DIC,output)			

			with open('host_coeff_poly_deg(%d)_DIC.pkl' %poly_deg, 'wb') as output:
				pickle.dump(bc_DIC,output)	

			with open('bloodmeal_coeff_poly_deg(%d)_DIC.pkl' %poly_deg, 'wb') as output:
				pickle.dump(bm_DIC,output)	

	elif flag==1:  #This section calls the stored data to be used in running the ODE
		import pylab
		bm_coeff_mat =pickle.load(open('bloodmeal_coeff_poly_deg(%d).pkl' %poly_deg,'rb'))
		mos_coeff = pickle.load(open('Mos_coeff_poly_deg(%d).pkl' %poly_deg, 'rb'))
		bc_coeff_mat = pickle.load(open('host_coeff_poly_deg(%d).pkl' %poly_deg,'rb'))
		mos_results = pickle.load(open('Mos_coeff_poly_deg(%d)_full_results.pkl' %poly_deg,'rb'))
		bm_results = pickle.load(open('bloodmeal_coeff_poly_deg(%d)_full_results.pkl' %poly_deg,'rb'))
		bc_results = pickle.load(open('host_coeff_poly_deg(%d)_full_results.pkl' %poly_deg,'rb'))

		rhs_func = Seasonal_ODE.test_rhs
		ODE_flag = 1			#ODE_flag, use 0 is the ODE is using counts, 1 if the ODE is using proportions, 2 if log proportions
		beta_flag = 0  # Set = 0 to run the beta optimization, otherwise will use a stored value of beta
		if beta_flag ==0:  # Is flag supposed to be ODE_flag?  
			#beta1 = scipy.optimize.leastsq(Seasonal_ODE.findbeta,.5,args=(rhs_func,bm_amean,bm_bmean,bc_coeff_mat,mos_coeff,tstart,tend,flag,ODE_flag))
			beta1 = scipy.optimize.minimize(Seasonal_ODE.findbeta,.5,args=(rhs_func,bm_coeff_mat,bc_coeff_mat,mos_coeff,tstart,tend,flag,ODE_flag),method="COBYLA",bounds=[(0,1)],options={"disp":True,"iprint":2,"rhobeg":.25})
			output5 = open('poly_deg(%d)_beta1.pkl' %poly_deg, 'wb')
			pickle.dump(beta1.x,output5)
		else:
			beta1 =     pickle.load(open('poly_deg(%d)_beta1.pkl' %poly_deg,'rb'))
		pylab.show()
		Y = Seasonal_ODE.run_ode(beta1.x,rhs_func,bm_coeff_mat,bc_coeff_mat,mos_coeff,tstart,tend,ODE_flag)
		
		
		if ODE_flag==2:
			Seasonal_ODE.eval_log_results(Y,bm_coeff_mat,bc_coeff_mat,mos_coeff,tstart,tend,bc_file)
		else:
			Seasonal_ODE.eval_ode_results(Y,bm_coeff_mat,bc_coeff_mat,mos_coeff,tstart,tend,bc_file,ODE_flag)
		pylab.show()	

	elif flag==2:  #Statistical Comparison of the Different Poly Degrees, run based off optimal DIC
		import pylab
		poly_deg_vals = [1,2,3,4]  #Can be updated to include different poly degrees
		bc_DIC_holder = numpy.zeros((len(poly_deg_vals),7))
		mos_DIC = numpy.zeros(len(poly_deg_vals))
		bm_DIC = numpy.zeros(len(poly_deg_vals))
		for i in range(len(poly_deg_vals)):
			bc_DIC_holder[i,:] = pickle.load(open('host_coeff_poly_deg(%d)_DIC.pkl' %poly_deg_vals[i],'rb'))
			mos_DIC[i] = pickle.load(open('Mos_coeff_poly_deg(%d)_DIC.pkl' %poly_deg_vals[i],'rb'))
			bm_DIC[i] = pickle.load(open('bloodmeal_coeff_poly_deg(%d)_DIC.pkl' %poly_deg_vals[i],'rb'))
		bc_DIC = numpy.sum(bc_DIC_holder,axis=1)
		bc_poly_deg = poly_deg_vals[numpy.where(bc_DIC==numpy.min(bc_DIC))[0][0]]
		mos_poly_deg = poly_deg_vals[numpy.where(mos_DIC==numpy.min(mos_DIC))[0][0]]
		bm_poly_deg = poly_deg_vals[numpy.where(bm_DIC == numpy.min(bm_DIC))[0][0]]
		#bm_poly_deg = 1  # When we update the code to allow for polynomial bm calc, replace with above line
		bm_coeff_mat =pickle.load(open('bloodmeal_coeff_poly_deg(%d).pkl' %bm_poly_deg,'rb'))
		mos_coeff = pickle.load(open('Mos_coeff_poly_deg(%d).pkl' %mos_poly_deg, 'rb'))
		bc_coeff_mat = pickle.load(open('host_coeff_poly_deg(%d).pkl' %bc_poly_deg,'rb'))

		poly_deg = [bc_poly_deg,bm_poly_deg,mos_poly_deg]
		rhs_func = Seasonal_ODE.test_rhs
		#beta1 = scipy.optimize.leastsq(Seasonal_ODE.findbeta,.5,args=(rhs_func,bm_amean,bm_bmean,bc_coeff_mat,mos_coeff,tstart,tend,flag,ODE_flag))
		ODE_flag = 1
		beta1 = scipy.optimize.minimize(Seasonal_ODE.findbeta,.5,args=(rhs_func,bm_coeff_mat,bc_coeff_mat,mos_coeff,tstart,tend,1,ODE_flag),method="COBYLA",bounds=[(0,1)],options={"disp":True,"iprint":2,"rhobeg":.25})
		output5 = open('poly_deg(%d)_beta1.pkl' % int(''.join(str(i) for i in poly_deg)), 'wb')
		Y = Seasonal_ODE.run_ode(beta1.x,rhs_func,bm_coeff_mat,bc_coeff_mat,mos_coeff,tstart,tend,ODE_flag)
		pickle.dump(beta1.x,output5)	
		Seasonal_ODE.eval_ode_results(Y,bm_coeff_mat,bc_coeff_mat,mos_coeff,tstart,tend,bc_file,ODE_flag)
		pylab.show()







# Lines used for debugging:
		# p = len(bm_amean)+1
		# beta2 = 1
		# gammab = .1*numpy.ones(p)
		# v=.14			# Biting Rate of Vectors on Hosts
		# b=0			# Bird "Recruitment" Rate
		# d=0	
		# dv=.10			# Mosquito Mortality Rate
		# dEEE= 0	
		# T = scipy.linspace(tstart,tend,1001)
		# Sv = .99
		# Iv = .01
		# S0 = .99*numpy.ones(p)
		# I0 = .01*numpy.ones(p)
		# Y0 = numpy.hstack((S0, I0, Sv, Iv))
		# ds,di,dsv,div,lbdv,alpha=Seasonal_ODE.get_ODE_vals(Y0,T,beta1.x,beta2, gammab, v, b, d, dv, dEEE, bc_coeff_mat,bm_amean,bm_bmean,bloodmeal.vector_pop,bloodmeal.vector_derivative,bloodmeal.vector_in,bloodmeal.fun,mos_coeff,p)


