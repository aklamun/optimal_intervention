# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 19:10:45 2020

@author: Ariah
"""

import numpy as np
import scipy.linalg as la


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
    n = len(C_hat)
    (y_0, y_1, b) = ([0 for i in range(n)],[0 for i in range(n)],[0 for i in range(n)])
    t=0
    
    while y_0 != y_1 or t==0:
        t += 1
        y_0 = y_1[:]
        V = la.lu_solve((lu,piv), Dp-b)
        v = np.dot(C_hat,V)
        y_1 = [1 if v[i]<theta[i] else 0 for i in range(n)]
        b = np.multiply(beta,y_1)
    return v, np.array(y_1, dtype=np.uint8)

def TransformThresh(lu, piv, C_hat, Dp, theta, beta):
    '''returns tilde_theta, nonzero entries are thresholds of nodes relevent to influence max problem'''
    v, Ind_T = solve_GJ_factor(C_hat, lu, piv, Dp, theta, beta)
    tilde_theta = np.zeros(len(Dp))
    
    C_hat_inv = np.diag( 1/np.diag(C_hat) )
    th = np.dot( C_hat_inv, theta)
    V = la.lu_solve((lu,piv), Dp - np.multiply(beta, Ind_T))
    
    for u in np.nonzero(Ind_T):
        Ind_u = np.zeros(len(Dp))
        Ind_u[u] = 1
        diff = la.lu_solve((lu,piv), np.multiply(beta,Ind_u))
        tilde_theta[u] = th[u] - V[u] - diff[u]

    return tilde_theta, Ind_T


###############################################################################
def f_GJ(Ind_S, lu, piv, C_hat, beta):
    '''influence propagation in GJ model
    Ind_S = influence set of nodes in GJ model, indicated by 1 entries
    lu, piv = la.lu_factor of I-C in GJ model
    Note: this doesn't reduce for self-contribution of nodes in S b/c influence on nodes in S ignored anyway'''
    z = la.lu_solve((lu,piv), np.multiply(beta, Ind_S))
    return z

def precompute_fv(f, lu, piv, C_hat, beta):
    '''pre-compute f([v]) for each node v
    used later in Gamma_neg'''
    n = len(C_hat)
    fv = {}
    for v in range(n):
        fv[v] = f([v], lu, piv, C_hat, beta)
    return fv

def Gamma_neg(v, Ind_A, f, lu, piv, C_hat, beta, fv):
    '''\Gamma^-(v,A) = total sum of weight of edges from node v to set A'''
    #Ind_A = np.array([1 if u in A and u != v else 0 for u in range(n)]) #this takes too long to construct, only really need to construct once, pass as arg
    Ind_A[v] = 0
    calc = np.sum(np.multiply(Ind_A, fv[v]))
    Ind_A[v] = 1
    return calc

def DiscountFrac_step(fv, Ind_S, x_1, f, b, C, C_hat, beta, lu, piv, theta_exp, Ind_T):
    q = {}
    Ind_A = Ind_T - Ind_S   #remaining set to influence
    
    fS = f(Ind_S, lu, piv, C_hat, beta)
    influence_cost = theta_exp - fS
    
    for v in np.nonzero(Ind_A)[0]:
        #q[v] = Gamma_neg(v, Ind_A, f, lu, piv, C_hat, beta, fv)     #influence of v on set A
        if influence_cost[v] > 0:
            q[v] = Gamma_neg(v, Ind_A, f, lu, piv, C_hat, beta, fv) / influence_cost[v]     #influence on set A / cost to activate v
        else:
            q[v] = np.inf
    
    #pick the max entry and add to seed set and payments x
    u = max(q, key=q.get)
    u_frac_inf = max(influence_cost[u],0)
    x_1[u] = u_frac_inf
    Ind_S[u] = 1
    
    fS = f(Ind_S, lu, piv, C_hat, beta)
    influence_cost = theta_exp - fS
    Ind_A = Ind_T - Ind_S
    #check for nodes that are already passed threshold
    repeat = True
    while repeat == True:
        change = False
        if np.sum(Ind_A) == 0:
            repeat = False
        for vv in np.nonzero(Ind_A)[0]:
            if influence_cost[vv] <= 0:
                change = True
                Ind_S[vv] = 1
            if change == True:
                Ind_A = Ind_T - Ind_S
                fS = f(Ind_S, lu, piv, C_hat, beta)
                influence_cost = theta_exp - fS
            else:
                repeat = False
    
    return x_1, Ind_S

def DiscountFrac_opt(f, b, C, C_hat, beta, lu, piv, theta_exp, Ind_T):
    '''heuristic algorithm for fractional influence maximization
    f = influence propagation function
    b = budget
    (C,C_hat,beta) define GJ system, (lu, piv) are la.lu_factor of I-C
    theta_exp = expected threshold shortfall (threshold - node value)
    theta_err = +/- error range for threshold shortfall
    Ind_T = initial set of node failures (set for which theta_exp + theta_err > 0), indicated by 1 entries
    outputs the optimization estimate for budget b, end influenced set S
    '''
    n = len(C_hat)
    (x_0,x_1) = (np.zeros(n),np.zeros(n))
    Ind_S = np.zeros(n, dtype=np.uint8)
    fv = precompute_fv(f, lu, piv, C_hat, beta)
    count=0
    while np.sum(x_1) < b and np.sum(Ind_S) < np.sum(Ind_T):
        #print(np.sum(x_1),len(S))
        x_0 = np.array(x_1, copy=True)
        Ind_S_0 = np.array(Ind_S, copy=True)
        x_1, Ind_S = DiscountFrac_step(fv, Ind_S, x_1, f, b, C, C_hat, beta, lu, piv, theta_exp, Ind_T)
        
        print(count)
        count += 1
        
    if np.sum(x_1) <= b:
        return x_1, np.nonzero(Ind_S)
    else:
        return x_0, np.nonzero(Ind_S_0)


def DiscountFrac_batch(fv, f, b_array, C, C_hat, beta, lu, piv, theta_exp, Ind_T):
    '''heuristic algorithm for fractional influence maximization
    fv = output of precompute_fv (a slow step)
    f = influence propagation function
    b_array = array of budgets to consider, ordered least to greatest
    (C,C_hat,beta) define GJ system, (lu, piv) are la.lu_factor of I-C
    theta_exp = expected threshold shortfall (threshold - node value)
    theta_err = +/- error range for threshold shortfall
    Ind_T = initial set of node failures (set for which theta_exp + theta_err > 0), indicated by 1 entries
    outputs the optimization estimates for all budgets in b_array, size of end influenced sets S
    '''
    x_dict = {}
    S_size_dict = {}
    
    n = len(C_hat)
    (x_0,x_1) = (np.zeros(n),np.zeros(n))
    Ind_S = np.zeros(n, dtype=np.uint8)
    for b in b_array:
        while np.sum(x_1) < b and np.sum(Ind_S) < np.sum(Ind_T):
            x_0 = np.array(x_1, copy=True)
            Ind_S_0 = np.array(Ind_S, copy=True)
            x_1, Ind_S = DiscountFrac_step(fv, Ind_S, x_1, f, b, C, C_hat, beta, lu, piv, theta_exp, Ind_T)
            
        if np.sum(x_1) <= b:
            x_dict[b] = np.array(x_1, copy=True)
            S_size_dict[b] = np.sum(Ind_S)
        else:
            x_dict[b] = np.array(x_0, copy=True)
            S_size_dict[b] = np.sum(Ind_S_0)
    
    return x_dict, S_size_dict




'''
tilde_theta, Ind_T = TransformThresh(lu,piv,C_hat,Dp*0.9,theta,beta)
b = np.sum(tilde_theta)
#x, S = DiscountFrac_opt(f_GJ,b,C, C_hat, beta, lu, piv, tilde_theta, Ind_T)
#vi,yi = solve_GJ_factor(C_hat, lu, piv, Dp*0.9, theta, beta)
#vf,yf = solve_GJ_factor(C_hat, lu, piv, Dp*0.9, theta-x, beta)

b_array = np.linspace(0, np.sum(Dp)/2, num=1000)
b_array = np.linspace(0, b, num=1000)
#b_array = np.array([0, b, b*2])
#fv = precompute_fv(f_GJ, lu, piv, C_hat, beta)
x_dict, S_size_dict = DiscountFrac_batch(fv, f_GJ, b_array, C, C_hat, beta, lu, piv, tilde_theta, Ind_T)
x_sums = [np.sum(x_dict[b_array[i]]) for i in range(len(b_array))]
'''
