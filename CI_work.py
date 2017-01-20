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
import statsmodels.stats.api as sms

bm_file = "Days_BloodMeal.csv"
bc_file = "Days_BirdCounts.csv"
msq_file = "Vector_Data(NoZeros).csv"
bm_data = pd.read_csv(bm_file,index_col=0)
bm_time = numpy.array([int(x) for x in bm_data.columns])
bm_data = bm_data.as_matrix()
tstart = 90 # Setting Start Time to April 1st
tend = 270

poly_deg = 2
flag = 0

bm_coeff_mat =pickle.load(open('bloodmeal_coeff_poly_deg(%d).pkl' %poly_deg,'rb'))
mos_coeff = pickle.load(open('Mos_coeff_poly_deg(%d).pkl' %poly_deg, 'rb'))
bc_coeff_mat = pickle.load(open('host_coeff_poly_deg(%d).pkl' %poly_deg,'rb'))
mos_results = pickle.load(open('Mos_coeff_poly_deg(%d)_full_results.pkl' %poly_deg,'rb'))
bm_results = pickle.load(open('bloodmeal_coeff_poly_deg(%d)_full_results.pkl' %poly_deg,'rb'))
bc_results = pickle.load(open('host_coeff_poly_deg(%d)_full_results.pkl' %poly_deg,'rb'))

#inputting bm_results works as written, need to input mos_results[0], similarly bc_results[j]
def get_ci(results,poly_deg):
	n = results[0].shape[0]/(poly_deg+1)
	up_ci = numpy.zeros((n,poly_deg+1))
	low_ci = numpy.zeros((n,poly_deg+1))
	for j in range(poly_deg+1):
		val = sms.DescrStatsW(results[:,j::poly_deg+1]).tconfint_mean()
		low_ci[:,j] = val[0] 
		up_ci[:,j] = val[1]
	return(low_ci,up_ci)	

if flag==0:
	# how do we run the beta1's?  Do we run for every possible combination of mos,bm,bc samples?
	beta1 = numpy.zeros(len(bm_results))

if flag==1:
	bm_ci = get_ci(bm_results,poly_deg)
	mos_ci = get_ci(mos_results[0],poly_deg)
	bc_ci = numpy.zeros((2,7,poly_deg+1))
	for j in range(7):
		val = get_ci(bc_results[j],poly_deg)
		bc_ci[0,j,:] = val[0]
		bc_ci[1,j,:] = val[1]

	#plotting the values with the data, and ci values
	bloodmeal.BloodmealTest(bm_file,bm_coeff_mat,poly_deg)
	bloodmeal.BloodmealTest(bm_file,bm_ci[0],poly_deg)
	bloodmeal.BloodmealTest(bm_file,bm_ci[1],poly_deg)

	BirdCount.BirdcountTest(bc_file,bc_coeff_mat,poly_deg)
	BirdCount.BirdcountTest(bc_file,bc_ci[0],poly_deg)
	BirdCount.BirdcountTest(bc_file,bc_ci[1],poly_deg)

	bloodmeal.MosquitoTest(msq_file,mos_coeff,poly_deg)
	bloodmeal.MosquitoTest(msq_file,mos_ci[0],poly_deg)
	bloodmeal.MosquitoTest(msq_file,mos_ci[1],poly_deg)

	#Calculate beta1 for the data, and the CI'sms

def get_beta1(bm_coeff_mat,bc_coeff_mat,mos_coeff):
	beta1 = scipy.optimize.minimize(Seasonal_ODE.findbeta,.5,args=(rhs_func,bm_coeff_mat,bc_coeff_mat,mos_coeff,tstart,tend,1,ODE_flag),method="COBYLA",bounds=[(0,1)],options={"disp":True,"iprint":2,"rhobeg":.25})
	return(beta1.x)
rhs_func = Seasonal_ODE.test_rhs
ODE_flag = 1			#ODE_flag, use 0 is the ODE is using counts, 1 if the ODE is using proportions, 2 if log proportions
beta_flag = 0  # Set = 0 to run the beta optimization, otherwise will use a stored value of beta
beta1 = get_beta1(bm_coeff_mat,bc_coeff_mat,mos_coeff)
beta1_ci=((get_beta1(bm_ci[0],bc_ci[0],mos_ci[0].squeeze()),get_beta1(bm_ci[1],bc_ci[1],mos_ci[1].squeeze())))


def build_bc_mat(results,index,poly_deg):
	a = numpy.zeros((7,poly_deg+1))
	for j in range(7):
		a[j,:] = results[j][index]
	return(a)



nsamples = 100  # This is the number of data points to be run, optimally 1000, until we have the newly parsed data, use 100
mos_val = floor(len(mos_results[0])/nsamples)
bm_val = floor(len(bm_results)/nsamples)
bc_val_holder = numpy.zeros(7)
for j in range(7):
	bc_val_holder[j] = floor(len(bc_results[j])/nsamples)
bc_val = numpy.min(bc_val_holder)
beta1_vals = numpy.zeros(nsamples)
ODE_results = numpy.zeros((nsamples,1001,23))
for j in range(nsamples):
	bm_array = numpy.reshape(numpy.array(bm_results[bm_val*j]),(6,poly_deg+1))
	bc_array = build_bc_mat(bc_results,bc_val*j,poly_deg)
	mos_array = mos_results[0][mos_val*j]
	beta1_vals[j] = get_beta1(bm_array,bc_array,mos_array)
	ODE_results[j] = Seasonal_ODE.run_ode(beta1_vals[j],rhs_func,bm_array,bc_array,mos_array,tstart,tend,ODE_flag)
