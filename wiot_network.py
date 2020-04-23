# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 18:10:08 2019

@author: aak228
"""

import pandas as pd
import numpy as np
import os
import networkx as nx

#first, need to convert WIOT xlsb files to xlsx

#fill in directory as appropriate
#directory = "..."
#data_path = 'wiot_data/'
#os.chdir(directory)

def get_data(year, data_path='wiot_data/'):
    if int(year) not in list(range(2000,2015)):
        raise Exception("No data for year {}".format(year))
    
    #Get table
    fname = 'WIOT{}_Nov16_ROW.xlsx'.format(year)
    df = pd.read_excel(data_path + fname, header=None, index_col=None, )
    df = df.drop([0])
    
    '''get the column data'''
    df_cols = df[1:4]
    df_cols = df_cols.drop([0,1,2,3],axis=1)
    cols = []
    for col in df_cols.columns:
        key = str(df_cols[col][4]) + "_" + str(df_cols[col][2])
        cols += [key]
    df_cols.columns = cols
    df_cols.index = ['category','description','country']
    df_cols = df_cols.transpose()
    
    '''get the index data'''
    df_inds = df.loc[:, 0:2]
    df_inds = df_inds.drop([1,2,3,4,5])
    inds = []
    for ind in df_inds.index:
        key = str(df_inds[2][ind]) + "_" + str(df_inds[0][ind])
        inds += [key]
    df_inds.index = inds
    df_inds.columns = ['category','description','country']
    
    '''format dataframe'''
    df = df.drop([0,1,2,3], axis=1)
    df = df.drop([1,2,3,4,5])
    df.columns = df_cols.index
    df.index = df_inds.index
    
    return df, df_cols, df_inds

def interpret_data(df, df_inds, rm_t=True):
    df_ = pd.DataFrame(index=df_inds.index[:-8], columns=df_inds.index[:-8]) #-8 entry is 'ROW_U'
    df_.loc[:'ROW_U',:'ROW_U'] = df.loc[:'ROW_U',:'ROW_U']
    df_ = df_.fillna(0)
    
    #invert coordinates of negative entries
    dfpos = df_[df_>0]
    dfpos = dfpos.fillna(0)
    dfneg = df_[df_<0]
    dfneg = dfneg.fillna(0)
    df_ = dfpos - dfneg.transpose()
    
    #scale columns to 1 (including value added)
    divisor1 = df_.sum(axis=0)
    divisor2 = df.loc['TOT_VA',:'ROW_U']
    divisor2 = divisor2.where(divisor2>0,-divisor2) #note reverse a few negative signs (like receiving subsidy)
    divisor = divisor1 + divisor2
    divisor = divisor.where(divisor>0) #replace 0s in divisor by nan
    df_ = df_.div(divisor,axis=1)
    df_ = df_.fillna(0)
    
    if rm_t == True:
        rm_inds = df_inds[df_inds.index.isin(df_.index)]
        rm_inds = rm_inds.loc[rm_inds['category']=='T']
        df2_ = df_.loc[(~df_.index.isin(rm_inds.index))]
        df2_ = df2_.drop(rm_inds.index,axis=1)
    else:
        df2_ = df_
    
    return df2_

def create_GJ_system(year):
    df, df_cols, df_inds = get_data(year)
    df_C = interpret_data(df, df_inds) #note: need to remove diagnals later to make C
    
    Dp = df.loc['TOT_GO',:'ROW_U']
    theta = Dp - df.loc['TOT_VA',:'ROW_U']*0.5
    beta = 0.1*df.loc['TOT_VA',:'ROW_U']
    Dp = Dp[Dp.index.isin(df_C.index)]
    theta = theta[theta.index.isin(df_C.index)]
    beta = beta[beta.index.isin(df_C.index)]
    return df_C, Dp, theta, beta, df_inds

def create_graph(year, name):
    df_C, Dp, theta, beta, df_inds = create_GJ_system(year)
    
    G = nx.from_numpy_matrix(df_C.values, parallel_edges=True, create_using=nx.MultiDiGraph())
    label_mapping = {idx: val for idx, val in enumerate(df_C.columns)}
    G = nx.relabel_nodes(G, label_mapping)
    node_dict = {}
    for nd in df_inds.index[:-1]:
        node_dict[nd] = df_inds['country'][nd] + "-" + df_inds['description'][nd]
    nx.set_node_attributes(G, name='label', values=node_dict)
    #nx.write_gexf(G, '{}{}.gexf'.format(name,year))
    return G, Dp, theta, beta

#year = 2014
#df, df_cols, df_inds = get_data(year)
#G, Dp, theta, beta = create_graph(year, 'wiot_graph')
