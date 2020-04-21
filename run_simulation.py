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
import interpret_data as interpret

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
rho = 0.6
sigma = 0.15
a = -0.3
samples = 5000
b_frac = 100
b_num = 5000
lu, piv, C_hat, fv, rvs = simulate.setup_simulate(C, Dp, theta, beta, rho, sigma, a, samples)
rvs = np.maximum(rvs, -1)
S_data, T_data, b_array = simulate.run_simulate(lu, piv, C_hat, fv, rvs, C, Dp, theta, beta, samples, b_frac, b_num)


with open('fs_2014', 'wb') as f:
    pickle.dump([rvs, S_data, T_data, b_array, (rho, sigma, a, samples, b_frac, b_num)],f)



#plot expected values and percentiles
n = len(C)
plt.plot(b_array/np.sum(Dp), np.mean(S_data, axis=0)/n)
plt.plot(b_array/np.sum(Dp), np.percentile(S_data, 90, axis=0)/n )
plt.plot(b_array/np.sum(Dp), np.percentile(S_data, 10, axis=0)/n )
plt.axhline(y=np.mean(T_data)/n)
plt.title('Expected Fraction of Defaults Prevented', fontsize=14)
plt.ylabel('Estimated E|S| / |U|', fontsize=14)
plt.xlabel('b / |Dp|', fontsize=14)
plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
plt.savefig('plot.eps')
plt.show()


q = 100
S_tvar, T_tvar = interpret.tvar(S_data, T_data, q)
n = len(C)
plt.plot( b_array/np.sum(Dp), (np.mean(T_data) - np.mean(S_data, axis=0))/n, label='Expected')
plt.plot( b_array/np.sum(Dp), (T_tvar - S_tvar)/n, label='TVaR(q=10)')
plt.title('Expected % of Firms Defaulting', fontsize=14)
plt.ylabel('Estimated E|T \ S| / |U|', fontsize=14)
plt.xlabel('b / |Dp|', fontsize=14)
plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
plt.legend(loc='upper right', fontsize=14)
plt.savefig('plot.eps')
plt.show()





