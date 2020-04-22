# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 03:39:29 2020

@author: Ariah
"""

import numpy as np
import scipy.linalg as la
import pickle
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.ticker as mtick
import seaborn as sns
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









###############################################################################
###############################################################################
###############################################################################
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

###############################################################################
#plot conditional expected defaults
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


###############################################################################
#plot 2d histogram with contour lines

fig, ax = plt.subplots()
hist = plt.hist2d(b_uniarray, A_uniarray, bins=300, cmap=cm.gray, norm=mcolors.PowerNorm(0.3))
CS = plt.contour(hist[1][1:], hist[2][1:], np.transpose(hist[0]), levels=80, cmap='flag')

# make a colorbar for the contour lines
CB = fig.colorbar(CS, shrink=0.9, extend='both', format='%.1e')

# We can still add a colorbar for the image, too.
CBI = fig.colorbar(hist[3], orientation='vertical', shrink=0.9, format='%.0e')

plt.title('Histogram % Firms Defaulting vs. Intervention Budget', fontsize=18)
plt.ylabel('% Firms Defaulting', fontsize=18)
plt.xlabel('Budget % of Total Assets |Dp|', fontsize=18)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=1))
ax.tick_params(axis='both', which='major', labelsize=18)

fig.set_size_inches(12, 7, forward=True)

plt.savefig('simulation_hist_contour.eps')
plt.show()

###############################################################################
#plot 1d histograms

fig, ax = plt.subplots()
sns.distplot(S_data[:,-1]/len(C), kde=False)
plt.title('Defaults Averted from 1% Intervention', fontsize=18)
plt.ylabel('Count', fontsize=18)
plt.xlabel('Defaults Averted (% Total Firms)', fontsize=18)
ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=1))
ax.tick_params(axis='both', which='major', labelsize=14)
ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
fig.set_size_inches(6.5, 5, forward=True)
plt.savefig('simulation_hist_1d_diff.pdf')
plt.show()

fig, ax = plt.subplots()
ax1 = sns.distplot(T_data/len(C), kde=False, label='No Intervention')
ax2 = sns.distplot((T_data - S_data[:,-1])/len(C), kde=False, label='1% Intervention')
plt.title('Histogram % Firms Defaulting', fontsize=18)
plt.ylabel('Count', fontsize=18)
plt.xlabel('% Firms Defaulting', fontsize=18)
ax1.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=1))
ax1.tick_params(axis='both', which='major', labelsize=14)
ax1.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
plt.legend(loc='upper right', fontsize=14)
fig.set_size_inches(7.5, 5, forward=True)
plt.savefig('simulation_hists_1d.pdf')
plt.show()





