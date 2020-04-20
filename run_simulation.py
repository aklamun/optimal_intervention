# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 03:39:29 2020

@author: Ariah
"""

import numpy as np
import scipy.linalg as la
import pickle
import matplotlib.pyplot as plt
#import multiprocessing

import GJ_cascades_dense as cascades
import simulate_shocks as simulate

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
C_hat = cascades.calc_C_hat(C)
lu, piv = la.lu_factor(np.eye(len(C))-C)
v,y = cascades.solve_GJ_factor(C_hat, lu, piv, Dp, np.zeros(len(C)), np.zeros(len(C)))
theta = v - value_added
theta = [theta[i] if theta[i]>=0 else 0 for i in range(len(C))]


###############################################################################
'''run the simulation'''
rho = 0.5
sigma = 0.15
a = -0.2
samples = 5000
b_frac = 100
b_num = 5000
lu, piv, C_hat, fv, rvs = simulate.setup_simulate(C, Dp, theta, beta, rho, sigma, a, samples)
S_data, T_data, b_array = simulate.run_simulate(lu, piv, C_hat, fv, rvs, C, Dp, theta, beta, samples, b_frac, b_num)

'''
#plot
plt.plot(b_array, y_fracs_e)
plt.plot(b_array, np.percentile(y_fracs_tot, 90, axis=0) )
plt.plot(b_array, np.percentile(y_fracs_tot, 10, axis=0) )
plt.title('Expected Fraction of Defaults Prevented', fontsize=14)
plt.ylabel('Estimated |S|/|T|', fontsize=14)
plt.xlabel('Budget b', fontsize=14)
plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
plt.savefig('plot.eps')
plt.show()
'''