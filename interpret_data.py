# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 11:57:05 2020

@author: Ariah
"""

import numpy as np

def tvar(S_data, T_data, q):
    '''output experimental tail value at risk for quantile q
    q = % worst scenario to size of T (e.g., 5% worst scenarios), larger T is worse
    '''
    (n,m) = S_data.shape
    q_level = np.percentile(T_data, 100-q)
    inds = [i for i in range(len(T_data)) if T_data[i] >= q_level]
    
    
    S_qvals = np.zeros((len(inds), m))
    i = 0
    for ind in inds:
        S_qvals[i,:] = S_data[ind,:]
        i += 1
    
    return np.mean(S_qvals, axis=0), np.mean([T_data[i] for i in inds])


'''
1. get q percentile of len(T)-len(S_final)
2. get all S rvs samples (rows) that fall above/below the q, percentile
3. take average over these indices of S_data

also do same but using q quantile of T initial failures
'''