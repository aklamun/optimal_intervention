# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 15:48:34 2020

@author: Ariah
"""

import numpy as np
import scipy.linalg as la
from scipy import stats

import GJ_cascades_dense



###############################################################################
def SampleShocks(p, rho, sigma, a, samples):
    n = len(p)
    u = a*np.ones(n)
    cov = sigma**2*( rho*np.ones((n,n)) + (1-rho)*np.eye(n) )
    mvnorm = stats.multivariate_normal(mean=u, cov=cov)
    rvs = mvnorm.rvs(samples)
    return rvs

def setup_simulate(C, Dp, theta, beta, rho, sigma, a, samples):
    lu, piv = la.lu_factor(np.eye(len(C))-C)
    C_hat = calc_C_hat(C)
    fv = precompute_fv(f_GJ, lu, piv, C_hat, beta)
    rvs = SampleShocks(Dp, rho, sigma, a, samples)
    return lu, piv, C_hat, fv, rvs

def run_simulate(lu, piv, C_hat, fv, rvs, C, Dp, theta, beta, samples, b_frac, b_num):
    b_array = np.linspace(0, np.sum(Dp)/b_frac, num=b_num)
    y_fracs_e = np.zeros(b_num)
    y_fracs_tot = np.zeros((samples, b_num))
    
    i=0
    for ran in rvs:
        Dp_prime = np.multiply(Dp, 1+ran)
        tilde_theta, T = TransformThresh(lu, piv, C_hat, Dp_prime, theta, beta)
        x_dict, S_size_dict = DiscountFrac_batch(fv, f_GJ, b_array, C, C_hat, beta, lu, piv, tilde_theta, T)
        y = np.array([S_size_dict[b]/len(T) for b in b_array])
        y_fracs_e += y
        y_fracs_tot[i,:] = y
        
        i += 1
    
    return y_fracs_e/samples, y_fracs_tot, b_array









###############################################################################
###############################################################################
'''Set up the network'''
year = 2014
path = 'C:/Users/Ariah/Box Sync/!-Research/2019/Code/cascade_sensitivity/wiot_model'
with open(path + '/wiot_rmt{}'.format(year), 'rb') as f:
    df_C, Dp, theta, beta = pickle.load(f)
    C = df_C.values
    np.fill_diagonal(C, 0)
value_added = np.array([beta[i]/0.1 if beta[i]>=0 else 0 for i in range(len(C))])
beta = 0.1*value_added
C_hat = calc_C_hat(C)
lu, piv = la.lu_factor(np.eye(len(C))-C)
v,y = solve_GJ_factor(C_hat, lu, piv, Dp, np.zeros(len(C)), np.zeros(len(C)))
theta = v - value_added
theta = [theta[i] if theta[i]>=0 else 0 for i in range(len(C))]


###############################################################################
'''run the simulation'''
rho = 0.3
sigma = 0.15
a = -0.1
samples = 10
b_frac = 100
b_num = 10000
lu, piv, C_hat, fv, rvs = setup_simulate(C, Dp, theta, beta, rho, sigma, a, samples)
y_fracs_e, y_fracs_tot, b_array = run_simulate(lu, piv, C_hat, fv, rvs, C, Dp, theta, beta, samples, b_frac, b_num)


#plot
plt.plot(b_array, y_fracs)
plt.plot(b_array, np.percentile(y_fracs_tot, 75, axis=0) )
plt.plot(b_array, np.percentile(y_fracs_tot, 25, axis=0) )
plt.title('Expected Fraction of Defaults Prevented', fontsize=14)
plt.ylabel('Estimated |S|/|T|', fontsize=14)
plt.xlabel('Budget b', fontsize=14)
plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
plt.savefig('plot.eps')
plt.show()


