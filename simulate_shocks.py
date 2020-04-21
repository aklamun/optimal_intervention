# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 15:48:34 2020

@author: Ariah
"""

import numpy as np
import scipy.linalg as la
from scipy import stats
import time

import GJ_cascades_dense as cascades



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
    C_hat = cascades.calc_C_hat(C)
    fv = cascades.precompute_fv(cascades.f_GJ, lu, piv, C_hat, beta)
    rvs = SampleShocks(Dp, rho, sigma, a, samples)
    return lu, piv, C_hat, fv, rvs

def run_simulate(lu, piv, C_hat, fv, rvs, C, Dp, theta, beta, samples, b_frac, b_num):
    b_array = np.linspace(0, np.sum(Dp)/b_frac, num=b_num)
    S_data = np.zeros((samples, b_num))
    T_data = np.zeros(samples)
    
    i=0
    for ran in rvs:
        Dp_prime = np.multiply(Dp, 1+ran)
        tilde_theta, Ind_T = cascades.TransformThresh(lu, piv, C_hat, Dp_prime, theta, beta)
        start_time = time.time()
        x_dict, S_size_dict = cascades.DiscountFrac_batch(fv, cascades.f_GJ, b_array, C, C_hat, beta, lu, piv, tilde_theta, Ind_T, Dp_prime, theta)
        y = np.array([S_size_dict[b] for b in b_array])
        S_data[i,:] = y
        T_data[i] = np.sum(Ind_T)
        
        print("{} simulation complete, {} initial failures, {} seconds, |T|={}, |S|={}".format(i,np.sum(Ind_T), time.time()-start_time, np.sum(Ind_T), np.sum(S_data[i,-1])))
        i += 1
    
    return S_data, T_data, b_array









