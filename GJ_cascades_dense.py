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
    
    #factor I-C and store
    lu, piv = la.lu_factor(np.eye(n)-C)
    v, y = solve_GJ_factor(C_hat, lu, piv, Dp, theta, beta)
    return v, y


def solve_GJ_factor(C_hat, lu, piv, Dp, theta, beta):
    '''lu,piv are LU factors of I-C'''
    n = len(C)
    (y_0, y_1, b) = ([0 for i in range(n)],[0 for i in range(n)],[0 for i in range(n)])
    t=0
    
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
    lu, piv = la.lu_factor of I-C in GJ model
    Note: this doesn't reduce for self-contribution of nodes in S b/c influence on nodes in S ignored anyway'''
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

def DiscountFrac_step(fv, S, x_1, f, b, C, C_hat, beta, lu, piv, theta_exp, T):
    q = {}
    A = list(set(T)-set(S))
    
    fS = f(S, lu, piv, C_hat, beta)
    influence_cost = theta_exp - fS
    
    #check for nodes that are already passed threshold
    repeat = True
    while repeat == True:
        change = False
        for vv in A:
            if influence_cost[vv] <= 0:
                change = True
                S.append(vv)
            if change == True:
                S = list(set(S))
                A = list(set(T)-set(S))
                fS = f(S, lu, piv, C_hat, beta)
                influence_cost = theta_exp - fS
            else:
                repeat = False
    
    for v in A:
        #q[v] = Gamma_neg(v, A, f, lu, piv, C_hat, beta, fv)     #influence of v on set A
        if influence_cost[v] > 0:
            q[v] = Gamma_neg(v, A, f, lu, piv, C_hat, beta, fv) / influence_cost[v]     #influence on set A / cost to activate v
        else:
            q[v] = np.inf
    
    #pick the max entry and add to seed set and payments x
    u = max(q, key=q.get)
    u_frac_inf = max(influence_cost[u],0)
    x_1[u] = u_frac_inf
    S.append(u)
    return x_1, S

def DiscountFrac_opt(f, b, C, C_hat, beta, lu, piv, theta_exp, T):
    '''heuristic algorithm for fractional influence maximization
    f = influence propagation function
    b = budget
    (C,C_hat,beta) define GJ system, (lu, piv) are la.lu_factor of I-C
    theta_exp = expected threshold shortfall (threshold - node value)
    theta_err = +/- error range for threshold shortfall
    T = initial set of node failures (set for which theta_exp + theta_err > 0)
    outputs the optimization estimate for budget b
    '''
    n = len(C)
    (x_0,x_1) = (np.zeros(n),np.zeros(n))
    S = []
    fv = precompute_fv(f, lu, piv, C_hat, beta)
    while np.sum(x_1) < b and len(S) < len(T):
        #print(np.sum(x_1),len(S))
        x_0 = np.array(x_1, copy=True)
        x_1, S = DiscountFrac_step(fv, S, x_1, f, b, C, C_hat, beta, lu, piv, theta_exp, T)
        
    if np.sum(x_1) <= b:
        return x_1
    else:
        return x_0


def DiscountFrac_batch(fv, f, b_array, C, C_hat, beta, lu, piv, theta_exp, T):
    '''heuristic algorithm for fractional influence maximization
    fv = output of precompute_fv (a slow step)
    f = influence propagation function
    b_array = array of budgets to consider, ordered least to greatest
    (C,C_hat,beta) define GJ system, (lu, piv) are la.lu_factor of I-C
    theta_exp = expected threshold shortfall (threshold - node value)
    theta_err = +/- error range for threshold shortfall
    T = initial set of node failures (set for which theta_exp + theta_err > 0)
    outputs the optimization estimates for all budgets in b_array
    '''
    x_dict = {}
    
    n = len(C)
    (x_0,x_1) = (np.zeros(n),np.zeros(n))
    S = []
    for b in b_array:
        while np.sum(x_1) < b and len(S) < len(T):
            x_0 = np.array(x_1, copy=True)
            x_1, S = DiscountFrac_step(fv, S, x_1, f, b, C, C_hat, beta, lu, piv, theta_exp, T)
            
        if np.sum(x_1) <= b:
            x_dict[b] = np.array(x_1, copy=True)
        else:
            x_dict[b] = np.array(x_0, copy=True)
    
    return x_dict



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

tilde_theta, T = TransformThresh(lu,piv,C_hat,Dp*0.9,theta,beta)
b = np.sum(tilde_theta)
#x = DiscountFrac_opt(f_GJ,b,C, C_hat, beta, lu, piv, tilde_theta, T)
#vi,yi = solve_GJ_factor(C_hat, lu, piv, Dp*0.9, theta, beta)
#vf,yf = solve_GJ_factor(C_hat, lu, piv, Dp*0.9, theta-x, beta)

b_array = np.linspace(0, np.sum(Dp)/2, num=1000)
b_array = np.linspace(0, b, num=1000)
#b_array = np.array([0, b, b*2])
#fv = precompute_fv(f_GJ, lu, piv, C_hat, beta)
x_dict = DiscountFrac_batch(fv, f_GJ, b_array, C, C_hat, beta, lu, piv, tilde_theta, T)
x_sums = [np.sum(x_dict[b_array[i]]) for i in range(len(b_array))]

