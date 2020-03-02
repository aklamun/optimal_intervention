# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 19:10:45 2020

@author: Ariah
"""

import numpy as np
import scipy.linalg as la
import pickle


def calc_C_hat(C):
    col_sums = np.sum(C,axis=0)
    vec = 1. - col_sums
    return np.diag(vec)

###############################################################################
def solve_GJ(C, Dp, theta, beta):
    '''solve G-J system
    returns v, y (default indicator)'''
    n = len(C)
    C_hat = calc_C_hat(C)
    (y_0, y_1, b) = ([0 for i in range(n)],[0 for i in range(n)],[0 for i in range(n)])
    t=0
    
    #factor I-C and store
    lu, piv = la.lu_factor(np.eye(n)-C)
    
    while y_0 != y_1 or t==0:
        t += 1
        y_0 = y_1[:]
        V = la.lu_solve((lu,piv), Dp-b)
        v = np.dot(C_hat,V)
        y_1 = [1 if v[i]<theta[i] else 0 for i in range(n)]
        b = np.multiply(beta,y_1)
    return v, y_1


###############################################################################
def f_GJ(S, lu, piv, C_hat, beta):
    '''influence propagation in GJ model
    S = influence set of nodes in GJ model
    lu, piv = la.lu_factor of I-C in GJ model'''
    n = len(C)
    y = [1 if i in S else 0 for i in range(n)]
    z = la.lu_solve((lu,piv), np.multiply(beta,y))
    return np.dot(C_hat,z)

def precompute_fv(f, lu, piv, C_hat, beta):
    '''pre-compute f([v]) for each node v
    used later in Gamma_neg'''
    n = len(C_hat)
    fv = {}
    for v in range(n):
        fv[v] = f([v], lu, piv, C_hat, beta)
    return fv

def Gamma_neg(v,A,f, lu, piv, C_hat, beta, fv):
    '''\Gamma^-(v,A) = total sum of weight of edges from node v to set A'''
    n = len(C_hat)
    y = np.array([1 if u in A else 0 for u in range(n)])
    return np.sum(np.multiply(y,fv[v]))

def DiscountFrac(f,b,C, C_hat, beta, lu, piv, theta_exp, theta_err, T, s):
    '''heuristic algorithm for fractional influence maximization
    f = influence propagation function
    b = budget
    (C,C_hat,beta) define GJ system, (lu, piv) are la.lu_factor of I-C
    theta_exp = expected threshold shortfall (threshold - node value)
    theta_err = +/- error range for threshold shortfall
    T = initial set of node failures (set for which theta_exp + theta_err > 0)
    '''
    n = len(C)
    (x_0,x_1) = (np.zeros(n),np.zeros(n))
    S = []
    fv = precompute_fv(f, lu, piv, C_hat, beta)
    while np.sum(x_1) < b and len(S) < n:
        #print(np.sum(x_1),len(S))
        q = {}
        A = list(set(range(n))-set(S))
        for v in A:
            q[v] = Gamma_neg(v, A, f, lu, piv, C_hat, beta, fv)
        x_0 = np.array(x_1, copy=True)
        u = max(q, key=q.get)
        fS = f(S, lu, piv, C_hat, beta)
        u_frac_inf = max(theta_exp[u] + theta_err[u] - fS[u],0)
        if u_frac_inf == 0:
            #in this case, don't consider other nodes that are already passed threshold
            for vv in A:
                if fS[vv] > theta_exp + theta_err:
                    S.append(vv)
            S = list(set(S))
        else:
            x_1[u] = u_frac_inf
            S.append(u)
    if np.sum(x_1) <= b:
        return x_1
    else:
        return x_0


year = 2014
path = 'C:/Users/Ariah/Box Sync/!-Research/2019/Code/cascade_sensitivity/wiot_model'
with open(path + '/wiot_rmt{}'.format(year), 'rb') as f:
    df_C, Dp, theta, beta = pickle.load(f)
    C = df_C.values
    np.fill_diagonal(C, 0)