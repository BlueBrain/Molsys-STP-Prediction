import pandas as pd
import numpy as np
import sklearn as sk
from sklearn import linear_model
from sklearn import preprocessing

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC

import torch
from torch import nn
from torch.optim import Adam
from torch.autograd import Variable 

import matplotlib.pyplot as plt
import time

import importlib
import stpdb # scripts for work with STP database
importlib.reload(stpdb)

#from IPython.display import display, Latex
from IPython.display import display, Markdown, Latex


'''
Scripts for training and testing genes->STP regression and classification models
'''


def fit_classes_tree(X2,y2,X2_cl,cli,model_type='lasso',nmin = 20, alpha=1, l1_ratio=0.5,
                     n_iter=300, alpha_1=1e-06, alpha_2=1e-06,lambda_1=1e-06, lambda_2=1e-06):
    '''
       hierarchical linear regression model 
    '''
    
    model = []
    support =[]
    ncl1 = np.max(X2_cl[:,cli[0]])+1 
    if len(cli)>1:
        ncl2 = np.max(X2_cl[:,cli[1]])+1 
    else:
        ncl2 = 1
            
    for i1 in range(ncl1):
        model1 = []
        support1 = []
        for i2 in range(ncl2):
            if ncl2==1:
                is_in_cl = (X2_cl[:,cli[0]]==i1)
            else:
                is_in_cl = (X2_cl[:,cli[0]]==i1)&(X2_cl[:,cli[1]]==i2)
            #           support2 = [np.sum(is_in_cl),
            #           np.nonzero((X2_cl[:,cli[0]]==i1))[0],
            #           np.nonzero((X2_cl[:,cli[1]]==i2))[0]]
            support2 = np.sum(is_in_cl)
            
            support1 = support1 + [support2]
            regr = 0
            if support2>nmin:
                y_train = y2[is_in_cl,:]
                X_train = X2[is_in_cl,:]
                if model_type=='ridge':
                    regr = sk.linear_model.Ridge(alpha=0.50, fit_intercept=True, normalize=False,
                                                 copy_X=True, max_iter=None, tol=0.001, solver='auto',
                                                 random_state=None)
                
                elif model_type=='lasso':
                    regr = sk.linear_model.Lasso(alpha=alpha, fit_intercept=True, normalize=False,
                                                 precompute=False, copy_X=True,
                                                 max_iter=1000, tol=0.0001, warm_start=False, positive=False, 
                                                 random_state=None, selection='cyclic')
                elif model_type=='elastic_net':   
                    regr = sk.linear_model.ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=True,
                                    normalize=False,
                                    max_iter=1000, copy_X=True, tol=0.0001, warm_start=False,
                                    random_state=None, selection='cyclic')
                elif model_type=='ARDRegression':  
                    regr = sk.linear_model.ARDRegression(n_iter=300, tol=0.001, alpha_1=alpha_1, alpha_2=alpha_2,
                                                lambda_1=lambda_1, lambda_2=lambda_2, compute_score=False, 
                                                threshold_lambda=10000.0, fit_intercept=True,
                                                normalize=False, copy_X=True, verbose=False)
                    y_train = y_train.ravel()
                elif model_type=='HuberRegressor': 
                    regr = sk.linear_model.HuberRegressor(epsilon=1.35, max_iter=n_iter,
                                                           alpha=alpha, warm_start=False,
                                                           fit_intercept=True, tol=1e-04)
                    
                    y_train = y_train.ravel()   

                else:
                    print(" model types supported: ridge, lasso, elastic_net")
                

                regr.fit(X_train, y_train)
                #y_pred = regr.predict(X_test[:,nannot:]) 


            model1 = model1 + [regr]
            
        support = support + [np.array(support1)]
        model = model + [np.array(model1)]    
    return np.array(model), np.array(support)  

def predict_classes_tree(model,X2,X2_cl,cli,nout=1,nmin = 20):

    '''
       predict using hierarcical linear regression model
    '''
    ncl1 = np.max(X2_cl[:,cli[0]])+1 
    if len(cli)>1:
        ncl2 = np.max(X2_cl[:,cli[1]])+1 
    else:
        ncl2 = 1
    y_pred = np.zeros((X2.shape[0],nout))
    for i1 in range(ncl1):
        #print(i1," f1")
        if len(model)>i1:
            #print(i1," f2")
            model1 = model[i1]
            for i2 in range(ncl2):
  
                if ncl2==1:
                    is_in_cl = (X2_cl[:,cli[0]]==i1)
                else:
                    is_in_cl = (X2_cl[:,cli[0]]==i1)&(X2_cl[:,cli[1]]==i2)

                support2 = np.sum(is_in_cl)
                
                #print(i1,i2," f3 ",support2)
                #regr = 0
                if support2>nmin:
                    #print(i1,i2," f3 ",support2)
                    if len(model1)>i2:
                        #print(i1,i2," f4")
                        regr = model1[i2]
                        #regr = sk.linear_model.MultiTaskElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=True,
                        #                    normalize=False,
                        #                    max_iter=1000, copy_X=True, tol=0.0001, warm_start=False,
                        #                    random_state=None, selection='cyclic')
                        #regr.fit(X_train, y_train)
                        if regr!=0:
                            X_test = X2[is_in_cl,:]
                            #print(X_test.shape, nout, regr.predict(X_test).reshape((-1,nout)).shape)
                            y_pred[is_in_cl,:] = regr.predict(X_test).reshape((-1,nout)) 
    
    return y_pred     


def cv_random_groups(n_splits, groups_index, part, N_bootstraps = 100, level='l1',do_random=1 ):
    #  part % random syn-types level2 cv 
    #  part = 10 # % of removed samples
    #  n_splits=10
    '''
       create datasets for cross validation removing of
       randomly selected genetic cell types of selected 
       taxonomical level (aka stp types, clusters, subclasses etc.)
       
    '''
    
    test_indexes = []
    names = []
    #cla_n = [cla_cell_type_l2]

    #groups_index = X_train0.loc[:,cla_n]
    #groups_index=groups_index.iloc[0:X_train0.shape[0]:Dn,:].values
    all_gr = set(groups_index.reshape((-1,)))
    all_gr = all_gr.difference(set([0]))
    all_gr = list(all_gr)

    #dn_syn_type = 100
    nall = groups_index.shape[0]
    all_index = np.arange(nall)
    all_index2 = np.arange(int(nall/N_bootstraps))
    #Dn3_out = int(nall*part/N_bootstraps)
    Dn3_out = int(nall*part/100) # its 100% !!!
    idx0 = all_gr  #np.arange(len(all_gr))
    #idx1 = np.arange(Dn2)
    for icv in range(n_splits):
        if do_random==1:
            idx0_out = np.random.choice(idx0, size=len(all_gr), replace=False, p=None)
        else:
            idx0_out = np.array([idx0[np.mod(icv,len(all_gr))]])
        test_index = np.zeros((0,))
        idx0_out2 = np.zeros((0,))
        for io in idx0_out:
            #train_index  = train_index + icv*dn + np.arange(dn)
            idx1 = np.sort(all_index[np.any(groups_index==io,axis=1)])
            if len(idx1) + len(test_index)<Dn3_out:
                test_index  = np.append(test_index, idx1)
                idx0_out2 = np.append(idx0_out2, io)
            else:
                #idx1 = np.random.choice(idx1, size=Dn3_out-len(test_index), replace=False, p=None)
                dDn3_out = Dn3_out-len(test_index)
                Dn4_out = int(dDn3_out/N_bootstraps)+1
                #idx2 = np.random.choice(all_index2, size=Dn4_out, replace=False, p=None)
                all_index2 = (idx1/N_bootstraps).astype(int)
                all_index3 = np.array(list(set(all_index2)))
                idx3 = np.sort(np.random.choice(all_index3, size=Dn4_out, replace=False, p=None))
                for io3 in idx3:
                    idx4 = np.sort(idx1[all_index2 ==io3])
                    if len(idx4) + len(test_index)<Dn3_out:
                        test_index = np.append(test_index, idx4)
                    else:    
                        idx4 = idx4[:Dn3_out-len(test_index)]
                        test_index  = np.append(test_index, idx4)
                        break
                idx0_out2 = np.append(idx0_out2,io)
                break

        test_indexes = test_indexes + [test_index]   
        names = names + ['syn_types_'+level+'_'+str(icv)+'  '+np.array2string(idx0_out2)]
        #df_cv.loc[icv,'train_index']=[train_index]
        #df_cv.loc[icv,'name']='syn_types_l0_'+str(icv)+'  '+np.array2string(idx0_out)

    #df_cv1  = pd.DataFrame([], columns = ['test_index', 'name'])
    df_cv1  = pd.DataFrame([test_indexes, names]).T
    df_cv1.columns = ['test_index', 'name']
    #return df_cv1
    return df_cv1


def load_genes_and_stp(fname, fname_columns=None):
    
    '''
    Load gene expression combined with STP data
    '''
    
    if fname_columns is None:
        columns  = pd.read_hdf(fname,key='column_names')
        key=None
    else:    
        columns  = pd.read_hdf(fname_columns)
        key='data'
    ge_columns      = columns.loc[:,'ge_columns_train'].values[0]
    annot_columns   = columns.loc[:,'annot_columns_train'].values[0]
    stp_columns     = columns.loc[:,'stp_columns_train'].values[0]
    classes_columns = columns.loc[:,'classes_columns_train'].values[0]
    #print(X_train.columns)
    X_train = pd.read_hdf(fname,key=key)
    
    return X_train, ge_columns, annot_columns, stp_columns, classes_columns


def add_class_hierarchy(new_syntype_columns,X_train0,classes_columns_train):
    ''' 
    Add classes to genes_STP dataframe
    '''
    for g in new_syntype_columns.index:
        df_new = X_train0.loc[:,new_syntype_columns.loc[g,'columns_X0']].copy()
        g20 = new_syntype_columns.loc[g,'columns_X0'][0]
        for g2 in new_syntype_columns.loc[g,'columns_X0']:
            df_new.loc[df_new.loc[:,g2]!=0, g20] = g2
        df_new_g = df_new.loc[:,g20].copy()   
        df_new_g.name = new_syntype_columns.loc[g,'group']
        X_train0 = pd.concat([X_train0, df_new_g],axis=1)
        classes_columns_train = classes_columns_train + [df_new_g.name]
    
    return X_train0, classes_columns_train

# a set of regularized regression functions, pytorch : ridge, uMDL, uMDL for mixed model



def torchRidge(y, X, lambdas):
    '''Solve ridge regression for a vector of regularization values.
    The multiple solutions are obtained efficiently using SVD or the Woodbury identity.
    With X (n_samples x ng) representing the feature matrix, and y ( n_samples x n_out) the outcomes,
    the ridge solution is given by (Woodbury)
        w = (X@X' + diag(lambdas))^{-1}@X@y
    or (SVD)
        w = V@(S**2 + diag(lambdas))^{-1}@S@U.t()@y
        for svd(X): X = U@S@V.t() 
    Arguments:
        X: n_samples x ng tensor of features (where m is the number of samples);            
        y: tensor ( n_samples x n_out) of outcomes.
        lambdas: tensor ng x 1  of initial regularization values for each feature.
    '''
    U, S, V = torch.svd(tr_X2train, some=True)
    S = S.reshape(-1,1)
    Uty = U.t() @ tr_y2train
    a = (S * Uty)/(S**2 +lambdas[0:S.shape[0]])
    w = V @ a

    if 0:
        # Setup.
        _, S, V = torch.svd(X)
        e = S ** 2
        p = X @ y
        Q = X @ V
        r = Q.t() @ p  # (X@V)'@X@y

        w = (1.0 / lambdas) * (p - Q @ (r / (e + lambdas)))

    return w



def npRidge(y, X, lambda_grid):
    '''Solve ridge ridgression in numpy
    '''
    #n = X.shape[0]
    [U, d, V] = np.linalg.svd(X, full_matrices=False) #fsvd = fast.svd(X)
    V = V.transpose()
    d = d.reshape(-1,1)
    Uty = U.transpose() @ y
    #k = len(lambda_grid)
    #dx = d.shape[0]
    #d2 = np.tile((d**2),(1,k)).transpose().reshape(1,-1)
    #div = d2 + np.repeat(lambda_grid, np.repeat(dx, k)) #d^2 + rep(lambda_grid, rep(dx, k))
    a = (d * Uty)/(d**2 +lambda_grid)
    #a = np.tile(np.squeeze(d * Uty),k)/div #drop(d * Uty)/div
    #a = a.reshape(k, dx).transpose() #dim(a) = c(dx, k)
    Betas = V @ a
    return Betas

def gen_random_HLM(M,n_samples):
    #M = {'W': W, 'Oi': Oi, 'R': R, 'RI': RI, 'Ki': Ki, 'Sig': Sig }
    Tr = M['Tree']
    Oi = M['Oi']
    R = M['R']
    RI = M['RI']
    Ki = M['Ki']
    W = M['W']
    Sig = M['Sig']
    nI = len(Oi)
    #n_samples = M['n_samples']

    # assign leaf (input) cluster for each sample datapoint
    zi = np.random.choice(a=np.arange(nI),size=(n_samples, 1),p=RI).ravel() #astype('int') 

    # assign output cluster for each sample datapoint
    zo = np.concatenate([np.random.choice(a=Oi[i],size=1,p=R[i]) for i in zi]).ravel()

    # for l in range(Tr.shape[0]):
    #    i_c = n_c-l-1
    #    nd = Tr[i_c] # select node
    #    lndc=len(nd['c'])
    #    if lndc==0:
    #        nd['x_data'] = np.random.randn(n_dim,1)*nd['sx'] + nd['x'] # node x data


    # sample training data : x[:,o] and y[:,o]=x[:,o]@w[o] + noise
    x = np.zeros((n_samples, n_dim))
    y = np.zeros((n_samples, n_out))
    # for ic in range(nI): # sample only leaf clusters
    # #for ic in range(n_c): # sample all cluster levels 
    #   ic_zi = zi==ic
    #   zoi = zo[ic_zi]
    #   n_i = np.sum(ic_zi) 

    #   # # make x for i-cluster
    #   # Tr[ic]['x_data'] = np.random.randn(n_dim,n_i)*np.tile(Tr[ic]['sx'],(1,n_i)) + np.tile(Tr[ic]['x'],(1,n_i)) # node x data
    #   # #xi = np.random.randn(n_i, n_dim)*np.random.rand(n_i,1) #np.tile(np.random.rand(n_i,1)) - should be cluster and gene specific
    #   # x[ic_zi,:] = Tr[ic]['x_data'].copy()

    #   # make x,y for each o|i-cluster
    #   xi = np.zeros((n_i, n_dim))
    #   yi = np.zeros((n_i, n_out))
    #   for o in Oi[ic]:   # make y for each o|i
    #       o_zoi = zoi==o
    #       n_oi = np.sum(o_zoi)

    #       # make x for o|i-cluster
    #       x_oi = np.random.randn(n_oi,n_dim)*np.tile(Tr[o]['sx'],(n_oi,1)) + np.tile(Tr[o]['x'],(n_oi,1)) # node x data    
    #       # Tr[o]['x_data'].append(.   np.random.randn(n_oi,n_dim)*np.tile(Tr[o]['sx'],(n_oi,1)) + np.tile(Tr[o]['x'],(n_oi,1))    )# node x data    
    #       #x[ic_zi,:] = Tr[ic]['x_data'].copy()
    #       xi[o_zoi,:] = x_oi.copy()

    #       # if o==4:
    #       #   print('ni',sum(ic_zi))
    #       #   print('n_oi',sum(ic_zi))
    #       #   print(np.mean(np.abs(x_oi)))
    #       #   print(zoi)
    #       #   fgfgffg

    #       # make x for o|i-cluster
    #       y_oi = xi[o_zoi,:]@W[o] + Sig[o]*np.random.randn(n_oi, n_out) # ??? - o clusters are not "real" - just different way to structure data? - if it can be implemented in a different way? how to do right nested x? 
    #       # y_oi should be a sum of all parent levels influences? - do on the level of parameters definition - w[o] = sum(pi*w_parents_influences), same for x?
    #       yi[o_zoi,:] = y_oi.copy()



    #   x[ic_zi,:] = xi.copy()
    #   y[ic_zi,:] = yi.copy()  
    for o in range(n_c):   # make y for each o|i
        o_zo = zo==o
        n_o = np.sum(o_zo)

        # make x for o|i-cluster
        x_o = np.random.randn(n_o,n_dim)*np.tile(Tr[o]['sx'],(n_o,1)) + np.tile(Tr[o]['x'],(n_o,1)) # node x data    
        # Tr[o]['x_data'].append(.   np.random.randn(n_oi,n_dim)*np.tile(Tr[o]['sx'],(n_oi,1)) + np.tile(Tr[o]['x'],(n_oi,1))    )# node x data    
        #x[ic_zi,:] = Tr[ic]['x_data'].copy()
        x[o_zo,:] = x_o.copy()

        # if o==4:
        #   print('ni',sum(ic_zi))
        #   print('n_oi',sum(ic_zi))
        #   print(np.mean(np.abs(x_oi)))
        #   print(zoi)
        #   fgfgffg

        # make x for o|i-cluster
        y_o = x[o_zo,:]@W[o] + Sig[o]*np.random.randn(n_o, n_out) # ??? - o clusters are not "real" - just different way to structure data? - if it can be implemented in a different way? how to do right nested x? 
        # y_oi should be a sum of all parent levels influences? - do on the level of parameters definition - w[o] = sum(pi*w_parents_influences), same for x?
        y[o_zo,:] = y_o.copy()




    return y, x, zo, zi


def torchuMDL(y,X,lambdas_init,sigmap_init, alpha_lambda=1e-8,beta_lambda=1e4,alpha_sigma=1e-4,beta_sigma=1e2,par={}):
    ''' uMDL for ridge ridgression for a search of optimal regularisation for each feature (https://arxiv.org/pdf/1804.09904.pdf).
    The multiple solutions are obtained efficiently using SVD.
    With X (n_samples x ng) representing the feature matrix, and y ( n_samples x n_out) the outcomes,
    the ridge solution is given by
        w = V@(S**2 + diag(lambdas))^{-1}@S@U.t()@y
        for svd(X): X = U@S@V.t() 
    use uMDL algorithm based on a bound for LNM for search of regularization parameters and sigma,
    use multistart to search for the global optimimum.     
    Arguments:
        X: n_samples x ng tensor of features (where m is the number of samples);            
        y: tensor ( n_samples x n_out) of outcomes.
        lambdas: tensor ng x 1  of initial regularization values for each feature.
    '''
    nit=1
    if 'nit' in par:
        nit=par['nit']
     
    relTol=1e-10
    if 'relTol' in par:    
        relTol=par['relTol']**2
    
    if False:
        alpha_lambda = par['alpha_lambda']
        beta_lambda = par['beta_lambda']

        alpha_sigma = par['alpha_sigma']
        beta_sigma = par['beta_sigma']
    #dl=1e10
    #iter=0

    n_dim = X.shape[1]
    n_samples = X.shape[0]
    n_out = y.shape[1]

    #print(lambdas_init)
    #lambdas_min = torch.zeros_like(lambdas_init[:,[0]]) #.copy()
    #sigmap_min = torch.zeros_like(sigmap_init) #.copy()
    wp_min = y.new_zeros([n_dim, n_out]) #.copy()
    
    sigmap_init = torch.reshape(sigmap_init, [1,-1])
    #print('sigmap 4 o ',sigmap_init)
    #print('sigmap 4 o len',sigmap_init.shape)
    lambdas_min = lambdas_init.clone() #lambdas_min.copy_(lambdas) #lambdas.copy()
    if sigmap_init.shape[0]<=1: # different sigma for each output dimension
        sigmap_init = y.new_zeros([1,n_out])+sigmap_init
    sigmap_min = sigmap_init.clone() #sigmap_min.copy_(sigmap) #sigmap.copy()
    
    

    pi = 3.1415927410125732 
    eps = torch.finfo(torch.float).eps

    #C = torch.sum(X*X).t()
    C = torch.sum(X*X,0, keepdim=True).t() + eps  # VERY IMPORTANT eps !!!!
    L_min = torch.tensor(float('inf'))
#     for i in range(lambdas_init.shape[1]):  # multistart search for "global" minimum    #??????????
#         lambdas = lambdas_init[:,[i]] +eps #??????????
    lambdas = lambdas_init[:,[0]] +eps    
        
    #sigmap = sigmap_init[:,i]
    sigmap = sigmap_init
     


    iter=0
    cond = True #(iter<nit) or (dl>relTol)
    Ls=[]
    while cond:
        # # e-step
        # U, S, V = torch.svd(X, some=True)
        # S = S.reshape(-1,1)
        # Uty = U.t() @ y
        # a = (S * Uty)/(S**2 +lambdas[0:S.shape[0]])
        # wp = V @ a
        # # m-step
        # lambdas0 = lambdas
        # lambdas = 0.5*C*(torch.sqrt(1 + sigmap**2/(C*wp**2)) - 1)

        # # L
        # L = (y-X@w)**2/(2*sigmap**2)


        # wp = argmin( dy2 + g) by wp | lambdas, sigmap
        # 
        U, S, V = torch.svd(X, some=True)
        S = S.reshape(-1,1)
        Uty = U.t() @ y
        a = (S * Uty)/(S**2 +lambdas[0:S.shape[0]])
        wp = V @ a
        #print('wp',wp.size())
        #print(V.size())
        #print(y.size())
        #print(lambdas.size())
        #print('S',S.size())
        
        #  lambdas = argmin( g + Zl )  by lambdas | wp, sigmap
        # 
        lambdas0 = lambdas
        lambdas = C/2*(torch.sqrt(1 + 4/(C*torch.sum((wp/sigmap)**2,1, keepdim=True))) - 1) # 4*!  AHTUNG - CHECK wp/sigmap !!!!
        lambdas[lambdas<alpha_lambda]=alpha_lambda
        lambdas[lambdas>beta_lambda]=beta_lambda

        #dy2 = torch.sum((y - X@wp)**2)
        #g   = torch.sum(lambdas*torch.sum(wp**2,1, keepdim=True))
        #dy2 = torch.sum((y - X@wp)**2)
        #g   = torch.sum(lambdas*torch.sum(wp**2,1, keepdim=True))
        dy2 = torch.sum((y - X@wp)**2,0, keepdim=True)
        g   = torch.sum(lambdas*wp**2,0, keepdim=True)
        
        #sigmap2 = (dy2 + g)/(n_samples*n_out) # sigmap = argmin( dy2 + g )  by sigmap | lambdas, wp
        sigmap2 = (dy2 + g)/(n_samples) # multi sigmap
        sigmap = torch.sqrt(sigmap2)
        #print('sigma : ',sigmap)
#         if sigmap<alpha_sigma:
#             sigmap = alpha_sigma
#         if sigmap>beta_sigma:
#             sigmap = beta_sigma
        sigmap[sigmap<alpha_sigma]=alpha_sigma
        sigmap[sigmap>beta_sigma]=beta_sigma    
            

        Zl =  1/2*torch.sum(torch.log(1 + C/lambdas))   
        #L =  (dy2 + g)/(2*sigmap**2) + n_samples/2*torch.log(2*pi*sigmap**2) +1/2*torch.sum(torch.log((lambdas+C)/lambdas))
        
        #L =  (dy2 + g)/(2*sigmap**2) + n_samples*n_out/2*torch.log(2*pi*sigmap**2)  + Zl
        L =  torch.sum( (dy2 + g)/(2*sigmap**2) + n_samples/2*torch.log(2*pi*sigmap**2) ) + Zl
        
        #if L<0:
        #    print('sigmap :', sigmap)
        #    print('Zlya :',1/2*torch.sum(torch.log(1 + C/lambdas)))
        #    print('g :', g)
        #    print('log_2_pi_sig2 :',n_samples*n_out/2*torch.log(2*pi*sigmap**2))

        #     dl = torch.median((lambdas- lambdas0)**2 )/torch.median(lambdas**2)
        #     print('i ',iter,' , dl median ',dl)

        dl = torch.median((lambdas- lambdas0)**2 )/torch.median(lambdas**2)
        
        
        
        iter=iter+1
        cond = (iter<nit) or (dl>relTol) #or (dL>relTolL)

        # if iter==0:
        #     print('i ',iter,' , dl median ',dl)
        
        # AHTUNG!!! Its unclear if last two if's are needed
        if (iter==1):
            Lmin=L
        if L<Lmin:  
            Lmin = L
            lambdas_min = lambdas.clone() #lambdas_min.copy_(lambdas) #lambdas.copy()
            sigmap_min = sigmap.clone() #sigmap_min.copy_(sigmap) #sigmap.copy()
            wp_min = wp.clone() #wp_min.copy_(wp)
            
            
        Ls = Ls + [L]
    
    dy2_ = torch.sum(dy2/(2*sigmap**2) + n_samples*n_out/2*torch.log(2*pi*sigmap**2)) #, 1,  keepdim=True)
    g_ = torch.sum(g/(2*sigmap**2))
    return wp_min, lambdas_min, sigmap_min, Lmin, Ls, dy2_, g_, Zl





def predictDNML( X, n_out, Wp, Sigmap=[], Wp_x=[], Sigmap_x=[], p_z=[], X_cl=None, cl_zo=0, cl_zi=-1, Loxi=None, OxI=None, par={}):
    # Loxi -  LL_uMDL for each cluster merge based on training data
        
    #Zo_jump, _ = E_Step(tr_y2train,tr_X2train, Wp,Lambdas,Sigmap,par)
    n_samples = X.shape[0]
    y = X.new_zeros([n_samples, n_out])
    
    is_lzo = False
    if ('dn_jumps' in par):
        if (par['dn_jumps']>0):
            do_sklearn_zo_prediction = False
            if do_sklearn_zo_prediction:
                m = par['clustering_model']
                zo = m.predict(X) #m.predict(X_cl)?
            else:
                #y = X.new_zeros([n_samples,n_out])
                par['zi'] = X_cl[:,cl_zi]
                zo, Lzo = E_Step(None,X,Wp,Sigmap, Wp_x=Wp_x, Sigmap_x=Sigmap_x, p_z=p_z, par=par)
                is_lzo = True
                #print('Lzo size ',Lzo.shape)
                #print('zo ',zo.shape)
        else:  
            zo = X_cl[:,cl_zo]
    else:        
        zo = X_cl[:,cl_zo]
        
    no_zi = True

    # find all nonfitted clusters
    is_wp = (zo[0:len(Wp)]>-1) # True
    for o in range(len(Wp)): #o_clusters: #range(n_c):
        if Wp[o]==None: #torch.sum(is_o)>0: #check!!!
            is_wp[o]=False
            
    # make zo from  fitted o only       
    for i in range(zo.shape[0]):
        if is_wp[zo[i]]==False:
            if is_lzo:
                zo[i]=torch.argsort(Lzo[i,is_wp])[0]
            else:    
                if no_zi:
                    zi = X_cl[:,cl_zi]
                    no_zi = False
                zoi=torch.argsort(OxI[zi[i]])  
                zo[i]=torch.min(zoi)
                            
    o_clusters = zo.unique()
    #n_c = o_clusters.numel() #len(Wp)
    #n_out = Wp[0].shape[1]
    #n_samples = X.shape[0]
    #y=Variable(torch.from_numpy(np.zero((n_samples ,n_out)).astype('float32')).to(cuda0))
    #y = X.new_zeros([n_samples, n_out])

    for o in o_clusters: #range(n_c):
        if Wp[o]!=None: # temporary - needed only for contol of very bad cases
            is_o = zo==o
            y[is_o,:] = X[is_o,:]@Wp[o]
        
#         if Wp[o]!=None: #torch.sum(is_o)>0: #check!!!
#             #print('X',X.shape)
#             #print('X_cl',X_cl.shape)
#             #print('Wpo',Wp[o].shape)
#             y[is_o,:] = X[is_o,:]@Wp[o]
#         else:
#             if False:
#                 if no_zi:
#                     zi = X_cl[:,cl_zi]
#                     no_zi = False
#                 i2 = zi[is_o].unique()
#                 #o2 = torch.argmin(Loxi[i2[0],:]) # !!! ASSUMPTION - o_clusters with unknown Wp[o] corresponds to leaf clusters
#                 Oi2 = np.nonzero(OxI[i2[0],:])
#                 Oi2 = Oi2[Oi2>o] # !!! ASSUMPTION - bigger (higher level) clusters have bigger index
#                 for o2 in Oi2: # !!! SHOULD BE MODIFIED - search for better regulirized fit level is needed (using Loxi)
#                     if Wp[o2]!=None:
#                         break
                        
                    

          #y[is_o,:] = X[is_o,:]@Wp[o2] # !!! ASSUMPTION - there is always a root cluster,   # ACHTUNG!! - UNCOMMENT THIS!!!

    #y = torch.concatenate([X[o]@Wp[o] for o in range(len(Wp))])  
    return y
 

def C_Z(n_c=0,n_samples=0): #(OxI,n_samples):

    return 0



def fit_NML_zo(Zo, Zi=None, par={}):
    n_c = par['n_c']
    n_I = par['n_I']
    cuda0 = par['device']
    n_samples = Zo.shape[0]
#     if 'OxI' in par:
#         OxI = par['OxI']
#     else:
#         OxI = torch.zeros([n_I,n_c])
    eps = torch.finfo(torch.float).eps
    Ziz=[]
    zi=[]
    L_z_i=0
    L_i=0
    if Zi != None:
        for izi in range(n_I):
            n_i = torch.sum(Zi==izi)
            zi = zi + [n_i/n_samples]
            ziz=[]
            for izz in range(n_c):
                n_z_i = torch.sum(Zo[Zi==izi]==izz)
                L_z_i =  L_z_i - n_z_i*torch.log(n_z_i/n_i +eps)
                ziz = ziz + [n_z_i/n_i]
            
            C_z_i = C_Z(n_c=n_c,n_samples=n_i)                    
            L_z_i = L_z_i + C_z_i                      
                                 
            Ziz = Ziz + [ziz]
            L_i =  L_i -n_i*torch.log(n_i/n_samples +eps)   
                         
        C_i = C_Z(n_c=n_I,n_samples=n_samples)                    
        L_i = L_i + C_i 
                      
        par['p_i']=torch.tensor(zi, device=cuda0) 
        par['p_z_i']=torch.tensor(Ziz, device=cuda0)
        
        

    zz=[]
    L_z = 0
    for izz in range(n_c):
        n_z = torch.sum(Zo==izz)
        L_z = L_z -n_z*torch.log(n_z/n_samples +eps)
        zz = zz + [n_z/n_samples]
    par['p_z']=torch.tensor(zz,device=cuda0)
    
                               
    C_z = C_Z(n_c=n_c,n_samples=n_samples)                           
    L_z = L_z + C_z
                               
    return L_z, zz, zi, Ziz, L_i, L_z_i, par                           

def E_Step(y,X,Wp,Sigmap, Wp_x=None, Sigmap_x=None, p_z=None, par={}):
        '''
        
        Expectation step for z - cluster assignment sampling of data sample (x,y,i)
        cases: z,i model : p(z|y,x,i) = p1(y|z,x,i)*p2(x|z,i)*p3(z|i)*p4(i)/sum by z of p(z,y,x,i)   - NOT IMPLEMENTED YET!!!
               z   model : p(z|y,x,i) = p1(y|z,x)*p2(x|z)*p3(z)/sum by z of p(z,y,x)
        parameters of p1:4 are estimated on M-step using decomposed NML + uMDL(for p1,2) approach        

        '''
        print_this = False
        if 'print_this' in par.keys():
            print_this = par['print_this']
            
        #Zo_jump, _ = E_Step(tr_y2train,tr_X2train, Wp,Sigmap,par=par)
        cuda0 = par['device']
        if y!=None:
            n_out = y.shape[1]
        n_samples = X.shape[0]    
        n_c=len(Wp)
        
        #G = par['G']
        #Zls = par['Zls']
        is_px = False  # if p(x|z) model is provided
        #if 'Wpx' in par:
        if Wp_x != None:
            #Wpx = par['Wpx']
            #log_pz = par['log_pz'] # log_p(z)
            is_px = True
            p_z = torch.tensor([1/n_c]*n_c ,dtype=torch.float, device=cuda0)
            if 'p_z' in par:
                p_z = par['p_z']
                is_pz_0=p_z==0
                p_z = p_z*n_samples
                p_z[is_pz_0]=1/torch.sum(is_pz_0)
                p_z = p_z/torch.sum(p_z)
            #print('log_p_z ',torch.log(p_z))
        
        pi = 3.1415927410125732 
        eps = torch.finfo(torch.float).eps
        inf = torch.finfo(torch.float).max
        
        
        L = torch.zeros([n_samples, n_c], dtype=torch.float, device=cuda0) + inf
        #sigmap = Sigmap[0]
        if  'Estep_resticted_i2o' in par:
            if par['Estep_resticted_i2o']:
                i2o_mask=par['OxI']==0
                zi = par['zi']                    
        
        for o in range(n_c):

                        
            wp=Wp[o]
            if wp!=None:
                #print('size sigmap',len(Sigmap))
                sigmap = Sigmap[o]
                #lambdas = Lambdas[o]
                
                # log_p(y|x,z)

                ##dy2_ = dy2/(2*sigmap**2) + n_samples*n_out/2*torch.log(2*pi*sigmap**2)
                ##g_ = g/(2*sigmap**2)
                #g_ = G[o]
                #Zl = Zls[o]

                #dy2 = torch.sum((y - X@wp)**2, 1)
                #g   = torch.sum(lambdas*torch.sum(wp**2,1, keepdim=True))
                #Zl =  1/2*torch.sum(torch.log(1 + C/lambdas))   
                if y!=None:
                    #L[:,o] =  (dy2 + g)/(2*sigmap**2) + n_samples*n_out/2*torch.log(2*pi*sigmap**2)   #+ Zl
                    L[:,o]  =  torch.sum( (y - X@wp)**2/(2*sigmap**2 +eps) + 1*1/2*torch.log(2*pi*sigmap**2 +eps), 1 ) #+ g_ #+Zl
                else:
                    L[:,o]  = 0
                #print('Ly ',torch.sort(L[:,o])[0:10])
                # add log_p(x|z)
                if is_px:

                    wp_x=Wp_x[o]
                    sigmap_x=Sigmap_x[o]
                    #x2 = X@wpx
                    Lxo = torch.sum( (X@wp_x)**2/(2*sigmap_x**2 +eps) + 1*1/2*torch.log(2*pi*sigmap_x**2 +eps), 1 )  
                    #print('Lx ',torch.sort(Lxo)[0:10])
                    #print('Lz ',p_z[o])
                    L[:,o]  =  L[:,o] +  Lxo  +    torch.log(p_z[o]+eps) # -1/2*det((2*pi)**n*wpx.T@wpx)
                    
                    if  'Estep_resticted_i2o' in par:
                        if par['Estep_resticted_i2o']:  
                            L[i2o_mask[zi,o],o] = inf

        #print('sigma : ',torch.tensor(Sigmap))
        
        if par['sample_Estep']:
            Lz_xy= L + torch.logsumexp(-L,1)
            mL,imL = torch.min(Lz_xy,1,keepdim=True)
            Lz_xy = Lz_xy - mL
            m = torch.distributions.categorical.Categorical(logits=-Lz_xy)
            Zo_jump = m.sample()
        else:  # maximum posterior zo 
            logsum_L = torch.logsumexp(-L,1,keepdim=True)
            if print_this:
                print('L ',L.shape)
                print('logsum_L ',logsum_L.shape)
            Lz_xy= L + logsum_L
            mL,imL = torch.min(Lz_xy,1,keepdim=False)
            Zo_jump = imL
        
        if print_this:
            print('Zo_jump : ', Zo_jump)
            #print(L.shape)
            #print('L : ', L[0:400:20,0:10])
            #print('sigma : ',torch.tensor(Sigmap))
            #print('G : ',G)
            #print('Zls : ',Zls)
        
        return Zo_jump, L

def fit_DNML_fixed_zo(tr_y2train,tr_X2train,Zo,Lambdas, Sigmap, par):

    # fit_DNML_fixed_zo(tr_y2train,tr_X2train,zio,lambdas, sigmap, par)
    # Wp2, Lambdas2, Sigmap2, Lo2 = fit_DNML_fixed_zo(tr_y2train,tr_X2train,zo_opt,Lambdas, Sigmap, par)
    do_uMDL = par['do_uMDL'] #True
    cuda0 = par['device']
    print_this = False
    if 'print_this' in par.keys():
        print_this = par['print_this']
    
    if 'do_x_z_model' in par:
        '''
        fit p(x|z) using uMDL for its estimate as Lumdl(0|x,z; theta = [sigmax, lyambdax, wx] )
        '''
        do_x_z_model = par['do_x_z_model'] 
        alpha_lambda = par['alpha_lambda_x']
        beta_lambda = par['beta_lambda_x']
        alpha_sigma = par['alpha_sigma_x']
        beta_sigma = par['beta_sigma_x']
        n_out_x = par['n_out_x']
        
    else:
        par['do_x_z_model']  = False
        alpha_lambda = par['alpha_lambda']
        beta_lambda = par['beta_lambda']
        alpha_sigma = par['alpha_sigma']
        beta_sigma = par['beta_sigma']
        

    #1 new w, lambda 
    n_dim = tr_X2train.shape[1]
    n_c = par['n_c']
    
    Lambdas2 = torch.zeros([n_dim, n_c], dtype=torch.float, device=cuda0)
    #Lambdas2 = []
    #torch_wo = torch.zeros([n_dim, 1], dtype=torch.float, device=cuda0)

    Temp2 = []
    Wp2 = []
    Sigmap2 = []
    Lo2=0
    LLo2=torch.zeros([n_c, 4], dtype=torch.float, device=cuda0)
    LLso2=[]
    multi_lam = len(Sigmap)>1
    if multi_lam==False:
        lambdas = Lambdas
        sigmap = Sigmap[0]
        #print('sigmap 1 ',sigmap)
    for o in range(n_c):
        if multi_lam:
            lambdas = Lambdas[:,[o]]
            sigmap = Sigmap[o] # must be torch array 1xn_out
        #print('sigmap 2 o ',sigmap)
        #Co = torch.sum(Ci[i4o[o]]).t()
        #C = torch.sum(tr_X2train*tr_X2train).t()
        n_samples = tr_X2train.shape[0]
        #wp, lambdas2, sigmap2, L = torchDNML(tr_y2train[zio_opt[o],:], tr_X2train[zio_opt[o],:], tr_X2train_cl[zio_opt[o],:], lambdas, sigmap, par)
        if len(Zo)>1:
            is_zo = Zo[o]
            #lzo = len(is_zo)
            lzo = torch.sum(is_zo)
        else:
            #print(Zo[0])
            is_zo = Zo[0]==o
            lzo = torch.sum(is_zo)
            
#         is_zo = Zo[0]==o
#         lzo = torch.sum(is_zo)    
            
        #print('lzo : ',o,torch.sum(Zo[0]==o),lzo,sigmap)
        
        if lzo>0:#zio_opt_l[o].shape[0]>0:
            #do_uMDL = True
            if do_uMDL:
                #print('sigmap 3 o ',sigmap)
                wp, lambdas2, sigmap2, L, Ls, dy2, g, Zl = torchuMDL(tr_y2train[is_zo,:], tr_X2train[is_zo,:], lambdas, sigmap,
                                                                     alpha_lambda=alpha_lambda,beta_lambda=beta_lambda,alpha_sigma=alpha_sigma,beta_sigma=beta_sigma, par=par)
                
#                 print('sigmap ',sigmap)
#                 print('sigmap2 ',sigmap2)
#                 print('L ',L)
#                 print('dy2 ',dy2)
#                 print('g ',g)
#                 print('Zl ',Zl)
                #print('vv0 ',o,sigmap2,sigmap)
                #np_y=tr_y2train[is_zo,:].detach().cpu().numpy() # æħŧ↓nŋ!!
                #np_x=tr_X2train[is_zo,:].detach().cpu().numpy() # æħŧ↓nŋ!!

                #temp = {'y':np_y, 'x':np_x, 'iszo':is_zo, 'wp':wp} # æħŧ↓nŋ!!
                temp = {} #{'y':[], 'x':[], 'iszo':is_zo, 'wp':wp}
            else:    
                #wp = torchRidge(tr_y2train[is_zo,:], tr_X2train[is_zo,:], lambdas)
                np_y=tr_y2train[is_zo,:].detach().cpu().numpy()
                np_x=tr_X2train[is_zo,:].detach().cpu().numpy()
                l1_ratio=0.01
                alpha = 0.5
                mode_l= sk.linear_model.ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=True,
                                        normalize=False,
                                        max_iter=1000, copy_X=True, tol=0.0001, warm_start=False,
                                        random_state=None, selection='cyclic').fit(np_x,np_y)
                wp = np.copy(mode_l.coef_ )
                wp = torch.from_numpy(wp).to(cuda0) #astype('long')
                wp = torch.reshape(wp,[wp.shape[0],1])
                lambdas2, sigmap2, L = lambdas, sigmap, 0
                temp = {} # {'y':np_y, 'x':np_x, 'iszo':is_zo, 'wp':wp}
                #ggg
        else: 
            #wp=torch_wo
            wp = None #[]   
            #lambdas2=[]
            lambdas2 = lambdas
            sigmap2=sigmap
            L, dy2, g, Zl, Ls  = 0, 0, 0, 0, 0

            temp = {} # {'y':[], 'x':[], 'iszo':is_zo, 'wp':wp}

        Wp2 = Wp2 + [wp]
        #Lambdas2 = Lambdas2 + [lambdas2]

        Temp2 = Temp2 + [temp]

        Lambdas2[:,[o]] = lambdas2
        #print('vv1 ',o,sigmap2)
        Sigmap2 = Sigmap2 + [sigmap2]
        Lo2 = Lo2 + L 
        LLo2[o,0], LLo2[o,1], LLo2[o,2], LLo2[o,3] = L, dy2, g, Zl
        
        
        #LLso2 = LLso2 + [Ls] 
    return Wp2, Lambdas2, Sigmap2, Lo2, Temp2, LLo2 #LLo2 #,LLso2#

def fit_DNML(tr_y2train,tr_X2train,tr_X2train_cl,M,lambdas, sigmap, par):
    #par = {'nit': nit, 'relTol': it_stop, 'alpha_sigma': alpha_sigma, 'beta_sigma': beta_sigma,
    #             'alpha_lambda': alpha_lambda, 'beta_lambda': beta_lambda, 'n_jumps': n_jumps, 'dn_jumps': dn_jumps, 'device':cuda0 }
    #print(lambdas.shape)
    cuda0 = par['device'] 
    n_jumps = par['n_jumps']
    dn_jumps = par['dn_jumps']
    n_iter_umdl = par['nit']
    
    if 'do_simplified_tree' in par.keys():
        do_simplified_tree = par['do_simplified_tree']
    else:
        do_simplified_tree = False    
        
    print_this = False
    if 'print_this' in par.keys():
        print_this = par['print_this']    

    Oi = M['Oi']
    nI = len(Oi)
    n_c = max([np.max(oic) for oic in Oi ])+1 #torch.max(tr_X2train_cl[:,0])+1
    par['n_c'] = n_c
    par['n_I'] = nI

    #print('n_c ',n_c)


    #OxI = M['OxI']
    OxI = torch.zeros([nI, n_c], dtype = torch.float, device=cuda0)  # OxI[i,o]=1 if O(i) contains o
    for i2 in range(nI):
        OxI[i2,Oi[i2]]=1 
    par['OxI'] = OxI 
        
    #nI = OxI.shape[0]
    #n_c = OxI.shape[1]
    n_dim = tr_X2train.shape[1]
    n_samples = tr_X2train.shape[0]
    n_out = tr_y2train.shape[1]
    #par['n_samples'] =  n_samples               

    eps = torch.finfo(torch.float).eps 
    #ng = X2train.shape[1]
    #lambdas = Variable(torch.from_numpy((lyambda0*np.ones((ng,1))).astype('float32')).to(cuda0)) #0.5*torch.tensor(np.ones(ng,1))
    #sigmap  = Variable(torch.from_numpy(np.array(sig).astype('float32')).to(cuda0))

    pi      = 3.1415927410125732  #Variable(torch.from_numpy(np.array(np.pi).astype('float32')).to(cuda0))
    zi = tr_X2train_cl[:,-1] # input clusters for each sample x-vector ---- AHTUNG!!! - ASSIGNMENT!!! 
    par['zi'] = zi
    
    i4o = []   # set of possible i-clusters for each o-cluster
    zio = []    # set of possible samples for each o-cluster
    for o in range(n_c):
        npi4o = torch.nonzero(OxI[:,o]==1) #[0] 
        i4o = i4o + [npi4o] # convert to pytorch variable (np.uint8)
        #(a[..., None] == b).any(-1).nonzero()
        npzio = (zi[..., None]==npi4o.flatten()).any(-1)
        zio = zio +[ npzio ]



    # uMDL for L_nml(y|x,zio)   ~ regression y|x,z 
    #Wp, Lambdas, Sigmap, Lo0, Temp2, LLso2 = fit_DNML_fixed_zo(tr_y2train, tr_X2train, zio, lambdas, [sigmap], par)
    #Wp, Lambdas, Sigmap, Lo0, Temp2, LLo2 = fit_DNML_fixed_zo(tr_y2train, tr_X2train, zio, lambdas, [sigmap], par)
    if print_this:
        print('sigmap ',sigmap)
    Wp, Lambdas, Sigmap, Lo0, Temp2, LLo2 = fit_DNML_fixed_zo(tr_y2train, tr_X2train, zio, lambdas, [sigmap], par)
    
                    
    # uMDL for L_nml(x|zio)  ~ x~N(0,Cov(z))
    #print('dn_jumps ',dn_jumps)
    if (dn_jumps>0)&(do_simplified_tree==False):
        n_out_x=n_out
        lambdas_x = lambdas
        sigmap_x = sigmap
        if 'n_out_x' in par:
            n_out_x=par['n_out_x']
        tr_y2train_x = torch.zeros((n_samples,n_out_x),dtype=torch.float, device=cuda0)  
        Wp_x, Lambdas_x, Sigmap_x, Lo0_x, Temp2_x, LLo2_x = fit_DNML_fixed_zo(tr_y2train_x, tr_X2train, zio, lambdas_x, [sigmap_x], par)
        if print_this:
            #print('Wp_x l',len(Wp_x))
            #print('n_c ',n_c)
            print('Lo0_x ',Lo0_x)
    
    if print_this:
        print('Lo0_y ',Lo0)
        #print('Ls')
        #for o in range(n_c):
        #    print('o : ',LLso2[o])
    
    
    do_statistics_of_LL_for_merging = False
    if do_statistics_of_LL_for_merging: 
        # trace of X*X for input clusters
        Ci = torch.zeros((nI,n_dim),dtype=torch.float, device=cuda0)  
        for i_c in range(nI):
            Ci[i_c,:] = torch.sum(tr_X2train[zi==i_c,:]*tr_X2train[zi==i_c,:],0) #.t()            
        
        # numbers of samples for input clusters            
        ns_i = tr_y2train.new_zeros([nI,1])
        for i in range(nI):
            ns_i[i] =  torch.sum(zi==i)            
                    
        # statisticas of LL components for different i-o combinations of samples - needed for cluster merging zo search
        LL = tr_y2train.new_zeros([nI,n_c])
        LL_oxi = tr_y2train.new_zeros([nI,n_c])
        G = tr_y2train.new_zeros([n_c,1])
        ns_o = tr_y2train.new_zeros([n_c,1])
        for i in range(nI):
            #n_i =  torch.sum(zi==i)
            for o in range(n_c):
                if Wp[o]!=None:
                    dy2 = torch.sum((tr_y2train[zi==i,:] - tr_X2train[zi==i,:]@Wp[o])**2) # assign all input data from i cluster to maximal o-cluster model 
                    if i==0:
                        ns_o[o] = torch.sum(ns_i[OxI[:,[o]]==1])
                        G[o]   = torch.sum(Lambdas[:,[o]]*torch.sum(Wp[o]**2,1, keepdim=True))/(2*Sigmap[o]**2) ### !!!12.5.2022 - WAS ERROR : not devided by sigma**2
                        #G[o]   = torch.sum(Lambdas[o]*torch.sum(Wp[o]**2,1, keepdim=True))
                    #LL[i,o] =  (dy2 + g)/(2*Sigmap[o]**2)+ G[o] + 1/2*n_out*n_i*torch.log(2*pi*Sigmap[o]**2)  #+1/2*torch.sum(torch.log((lambdas+C)/lambdas))
                    LL[i,o] =  dy2/(2*Sigmap[o]**2) + 1/2*n_out*ns_i[i]*torch.log(2*pi*Sigmap[o]**2)  #+1/2*torch.sum(torch.log((lambdas+C)/lambdas)) # g doesn't needed?
                    if OxI[i,o]==1:
                        LL_oxi[i,o] = LL[i,o]
                    
                    
            #else:
            #    LL[i,o] =  0

    # make Ci/lambdas
    #Cio = [[] for o in range(n_c)]
    #for o in range(n_c):
    #  for i in range(nI):
    #    Cio[o] = Cio[o] + [Ci[i]/Lambdas[o]]       

    #Search for best assignments of n[i|o] :
    # 1 start from all n[i|o] in i-clusters
    # 2 test sequential clusters merge - find best n[i|o] configuration
    # 3 test random search of n[i|o] around best configuration +- dn_jumps
    #nio =  torch.from_numpy(np.zeros((nI,n_jumps)).astype('long')).to(cuda0)         # nio[i,jump]=o if n(i|o) = 1 for jump  
    trOi = [ torch.from_numpy(Oi[i2].astype('long')).to(cuda0) for i2 in range(nI) ] # torch version of O(i) sets
    a_nI = torch.arange(nI, dtype = torch.long, device=cuda0)
    a_n_dim3 = torch.from_numpy(np.arange(n_dim).astype('long')).repeat(nI).to(cuda0)
    
    nio_min = torch.zeros( nI, dtype = torch.long, device=cuda0)
    nio3 = torch.zeros( nI, dtype = torch.long, device=cuda0)  #torch.zeros_like( nio[:,0])
    #    try assign all n[i|o] to leaves clusters
    for i2 in range(nI): 
        nio3[i2] = trOi[i2][0] ### !!!!!!! - ASSUMPTION: trOi[i2][0] must corresponds to leaves!!!
    
    if do_statistics_of_LL_for_merging:
        Ci3 = Ci.reshape([-1,])     
        Coi3 = torch.zeros([n_dim,n_c], dtype = torch.float, device=cuda0)
        idx3 = [a_n_dim3, nio3.repeat_interleave(n_dim)]
        Coi3 = Coi3.index_put_(idx3, Ci3, accumulate=True)
    
    #Lmin = torch.sum(LL[[a_nI,nio3]])  + 1/2*torch.sum(torch.log(1+Coi3))  
    
    #Lmin = torch.sum(LL[[a_nI,nio3]])  + torch.sum(G[torch.unique(nio3)]) +1/2*torch.sum(torch.sum(torch.log(1+Coi3/(Lambdas+eps)),0)) 
    o_slected = torch.unique(nio3) 
    L_z=0
#     for o in o_slected:
#         ns_o = torch.sum(zi==o)
#         L_z =  L_z -ns_o*torch.log(ns_o/n_samples + eps)
#     L_z = L_z + C_Z(n_c,n_samples)        
    Lmin = torch.sum(LLo2[o_slected,0]) 
    if (dn_jumps>0)&(do_simplified_tree==False):
        Lmin = Lmin + torch.sum(LLo2_x[o_slected,0]) +L_z
    
    if print_this:
        print('o_slected ',o_slected)
        print('Lmin ',Lmin)
        print('nio_min ', nio3)
    
    #Lmin = torch.sum(LL[[a_nI,nio3]])   +1/2*torch.sum(torch.sum(torch.log(1+Coi3/(Lambdas+eps)),0)) # G[o] should be added for all o in the model - will not influence result !
    
    nio_min = nio3.clone()
    #print(nio3)
    #print(1/2*torch.sum(torch.log(1+Coi3),0), Lmin)
    
    if do_simplified_tree==False:
        
        do_merge_clusters = False
        if do_merge_clusters:
            #     try to sequentially merge leaves i_clusters in to bigger o-clusters and select best configuration
            Zl2=torch.zeros([n_dim,n_c], dtype = torch.float, device=cuda0)
            for o in range(nI, n_c): # test merging of o-clusters - !!!!!!!!! ASSUMPTION - first nI o-clusters - leaves  
                Coi3 = torch.zeros([n_dim,n_c], dtype = torch.float, device=cuda0)
                nio3[OxI[:,o]==1] = o  # merge all i_clusters into o_cluster - !!!!!!!!! ASSUMPTION - OxI describes a sequence of merging from all o-clusters == leaves to top o-cluster  # 12.5.2022 START FROM CHECKING THIS!!!!
                            #  CRITIQUES of clusters merging. : 1) 
                #                       1) not really check all possible sequences of merging ?
                #                       2) check nio3 changing
                #                       3) 

                idx3 = [a_n_dim3, nio3.repeat_interleave(n_dim)]
                Coi3 = Coi3.index_put_(idx3, Ci3, accumulate=True)

                #L = torch.sum(LL[[a_nI,nio3]])  + 1/2*torch.sum(torch.log(1+Coi3))
                #L3 = torch.sum(LL[[a_nI,nio3]]) + torch.sum(G[torch.unique(nio3)])+ 1/2*torch.sum(torch.log(1+Coi3/(Lambdas+eps))) # optimal G for zero data is small (lyambda=beta_lyambda)



                o_slected = torch.unique(nio3) 
                L_z = -torch.sum(ns_o[o_slected]*torch.log(ns_o[o_slected]/n_samples))
                L = torch.sum(LLo2[o_slected,0]) #+ L_z


                #L2 = torch.sum(LL_oxi[OxI[:,o]==1,o]) + 1/2*torch.sum(torch.sum(torch.log(1+Co[:,torch.unique(nio3)]/(Lambdas+eps)),0))

                #  CRITIQUES on L EQ. : 1) Lambdas and Wp not corresponds to real Lambdas of obtained clusters but rather to maximal o-clusters - Z(lyambda) -
                #                          fake!!! - refit of uMDL for current clusters may be needed - no , all o-clusters will be maximal
                #                       2) no L_nml(Z) term!!!!!
                #                       3) check Coi3 term!!!
    #             print('o : ',o)
    #             print('Lmin : ',Lmin)
    #             print('Cz : ',Cz)
    #             print('L : ',L)
    #             print('ns_o ',ns_o.T)
    #             print('a_nI ',a_nI)
    #             print('nio3 ',nio3)
    #             print('o_selected : ', o_slected)
    #             print('L_o',LLo2[o_slected,0])
    #             print('dy2_o',LLo2[o_slected,1])
    #             print('g_o',LLo2[o_slected,2])
    #             print('Zl_o',LLo2[o_slected,3])
    #             print('L_dy2 : ',torch.sum(LL[[a_nI,nio3]]))
    #             print('L_Zl : ',1/2*(torch.sum(torch.log(1+Coi3/(Lambdas+eps)),0)))
    #             print('Coi3 : ',torch.sum(torch.abs(Coi3),0))
    #             print('Ci3 ',torch.abs(Ci3))
    #             print('OxI :',OxI)
    #             print('LL : ',LL_oxi)
    #             print('LL[[a_nI,nio3]] : ',LL[[a_nI,nio3]])

                #print(Lmin, L)
                if L<Lmin:
                    if print_this:
                        print('L start vs new : ' ,Lmin,L)
                        print('nio3 start vs new : ' ,nio_min,nio3)

                    Lmin = L  
                    nio_min = nio3.clone()

                else:  
                    nio3 = nio_min.clone()
                

        Wp_m, Lambdas_m, Sigmap_m = Wp, Lambdas, Sigmap
        if (dn_jumps>0)&(do_simplified_tree==False):
            Wp_m_x, Lambdas_m_x, Sigmap_m_x = Wp_x, Lambdas_x, Sigmap_x
        # random jumps near current best configuration (nio_min +- dn_jumps)
        # ACTUNG!!! - currently it doesn't work - w can not be fixed and should be reoptimized for each jump
        # (with fixed lambda it become not optimal but stil will be ok)
        # current w gives an estimate from above for g , as well as current lambda - for logZ - if LL estimate still selected (is less than Lmin)- it will be better after w,lambda reestimation 
        # plane random jumps : if Ki_init == Ki
        #print('do_DNML_torch 0')
        if dn_jumps>0: 
            # do Ki_init and nio_2_init for jumping around  nio_min +- dn_jumps
            #nio =  torch.from_numpy(np.zeros((nI,n_jumps)).astype('long')).to(cuda0)         # nio[i,jump]=o if n(i|o) = 1 for jump  
#             nio_2_min = torch.zeros(nI, dtype = torch.long, device=cuda0)
#             nio_2_init = torch.zeros(nI, dtype = torch.long, device=cuda0)
#             Ki_init = torch.zeros(nI, dtype = torch.long, device=cuda0)
#             for i2 in range(nI):
#                 nio_2_min[i2] = torch.nonzero(nio_min[i2]==trOi[i2])[0]     # position index of nio_min in O(i) set
#                 nio_2_init[i2] = torch.max(torch.tensor([nio_2_min[i2] - dn_jumps , 0]))   # position of lower sampling bound inside O(i) set
#                 Ki_init[i2] = torch.min(torch.tensor([nio_2_min[i2] + dn_jumps, trOi[i2].numel()])) - nio_2_init[i2] # range of sampling for each O(i) set : from nio_2_init[i2] to nio_2_init[i2]+KI_init[i2]

            # number of jumps without repeats
#             n_jumps = min([n_jumps,torch.prod(torch.tensor(Ki_init))])

            # z_n_c = torch.from_numpy(np.zeros(n_jumps).astype('long')).to(cuda0)
            # #a_n_jumps = torch.from_numpy(np.arange(n_jumps).astype('long')).to(cuda0)
            # a_n_jumps = torch.arange(n_jumps, dtype = torch.long, device=cuda0)
            # a_n_jumps2 = a_n_jumps.repeat_interleave(n_dim)
            # a_n_dim = torch.from_numpy(np.arange(n_dim).astype('long')).repeat(n_jumps).to(cuda0)

            # Ci2 = Ci.t().repeat([n_jumps,1])
            # Coi2 = torch.zeros([n_dim,n_c,n_jumps], dtype = torch.float, device=cuda0)

            # noi2 = torch.zeros([n_c,nI,n_jumps], dtype = torch.long, device=cuda0)

            # generate n_jumps random n(i|o) samples
#             n_max = torch.prod(torch.tensor(Ki_init)) # number of all possible samples
#             nio = torch.zeros([nI,n_jumps], dtype = torch.long, device=cuda0)
#             nio_2 = torch.multinomial(torch.ones(n_max,device=cuda0), n_jumps) # sample randomly without replace
#             nio2 = torch.zeros([nI,n_jumps], dtype = torch.long, device=cuda0)
#             for i2 in range(nI):  # transform sample number to sample vector
#                 nio2[i2,:] =  torch.fmod(nio_2, Ki_init[i2])
#                 nio[i2,:] =  trOi[i2][torch.fmod(nio_2, Ki_init[i2])+nio_2_init[i2]]
#                 nio_2 = (nio_2-nio2[i2,:])//Ki_init[i2]
#             #print(nio)

            # # collect sum(Ci) for each o-cluster
            # for i2 in range(nI):
            #   idx = [nio[i2,:] , z_n_c+i2, a_n_jumps ] 
            #   idx2 = [a_n_dim, nio[i2,:].repeat_interleave(n_dim) ,  a_n_jumps2 ] 
            #   noi2[idx ] = 1
            #   Coi2 = Coi2.index_put_(idx2, Ci2[:,i2], accumulate=True) 

            # #eps = torch.finfo(torch.float).eps
            # Coi2 = Coi2*(1/(Lambdas+eps)).reshape([n_dim,-1,1]).repeat([1,1,n_jumps])

            Ldnml_best = Lmin #torch.min(L)
            nio_best = nio_min.clone()
            
            #Wp_, Lambdas_, Sigmap_ = Wp.clone(), Lambdas.clone(), Sigmap.clone()

            #Zo_jump = torch.zeros_like(tr_X2train_cl[:,0])
#             nio = torch.zeros([nI,n_jumps], dtype = torch.long, device=cuda0)
#             zii = [zi==i2 for i2 in range(nI)]
            par['sample_Estep']=False
            for i_jump in range(n_jumps):
                
                # E step
                #par['G']=LLo2[:,2]
                #par['Zls']=LLo2[:,3]
                
                #print('Sigmap',Sigmap)
                #if i_jump>2:
                #    par['sample']=True
                #print(Wp)
                Zo_jump, _ = E_Step(tr_y2train, tr_X2train, Wp, Sigmap, Wp_x=Wp_x, Sigmap_x=Sigmap_x, par=par)
                
                # M step
                # HERE REESTIMATION OF Wp, G and LL IS NEEDED!!! (if no reestimation of G - "full" bigger clusters almost always will be better?)
                #for i2 in range(nI):  
                #    Zo_jump[zii[i2]]=nio[i2,i_jump]
                
                if print_this:
                    print('i_jump : ',i_jump)
                    print('Ldnml_best : ',Ldnml_best)

                par['nit']=5
                # new parameters of y|x,z model
                Wp_, Lambdas_, Sigmap_, Lo2_y, Temp2, LLo2 = fit_DNML_fixed_zo(tr_y2train,tr_X2train,[Zo_jump],Lambdas, Sigmap, par)
                
                # new parameters of x|z model
                Wp_x_, Lambdas_x_, Sigmap_x_, Lo2_x, Temp2_x, LLo2_x = fit_DNML_fixed_zo(tr_y2train_x,tr_X2train,[Zo_jump],Lambdas_x, Sigmap_x, par)
                
                # new p(z) and p(z|i)
                L_z, _, _, _, _, _, par = fit_NML_zo(Zo_jump,par=par) # fit_NML_zo(Zo,Zi=Zi,par=par)
                                           
                # final Decomposed NML measure:
                
                Lo2 = Lo2_y  +  Lo2_x  +  L_z
                

                if print_this:
                    #print('Sigmap_',Sigmap_)
                    #print('Sigmap',Sigmap)
                    print('Lo2 : ',Lo2)    
                    print('nio_best statistics : ',torch.tensor(par['p_z']))
                    print('Lo2 y : ',Lo2_y) 
                    print('Lo2 x: ',Lo2_x) 
                    print('Lo2 z: ',L_z) 


                # # estimate Ldnml for each n(i|o) configuration
                # #L = torch.sum(LL.gather(1, nio),0) + 1/2*torch.sum(torch.sum(torch.log(1+Coi2),0),0)
                # L = torch.sum(LL.gather(1, nio),0) + 1/2*torch.sum(torch.sum(torch.log(1+Coi2),0),0)

                # # select best n(i|o) configuration
                # j_best = torch.argmin(L)
                # Ldnml_best = L[j_best] #torch.min(L)
                # nio_best = nio[:,j_best]
                if Ldnml_best > Lo2:
                    nio_best = Zo_jump.clone()
                    Ldnml_best = Lo2
                    #Wp, Lambdas, Sigmap = Wp_.clone(), Lambdas_.clone(), Sigmap_.clone()
                    for o in range(n_c):
                        if Wp_[o] is None:
                            Wp_[o] = Wp[o]
                            Sigmap_[o] = Sigmap[o]
                            Lambdas_[o] = Lambdas[o]           
                    
                    Wp, Lambdas, Sigmap = Wp_, Lambdas_, Sigmap_ # AHTUNG!!! problem of empty clusters
                    Wp_x, Lambdas_x, Sigmap_x = Wp_x_, Lambdas_x_, Sigmap_x_ # AHTUNG!!! problem of empty clusters
                    #print('Sigmap',Sigmap)

                #print('nio_best',nio_best)

                Ldnml_best_jumps=Ldnml_best
                nio_best_jumps = nio_best
                if Ldnml_best>Lmin:
                    nio_best = nio_min.clone()
                    Ldnml_best = Lmin
                    Wp, Lambdas, Sigmap = Wp_m, Lambdas_m, Sigmap_m
                    Wp_x, Lambdas_x, Sigmap_x = Wp_m_x, Lambdas_m_x, Sigmap_m_x
        
        
            # M step : train tr_X2train_cl, tr_X2train -> Z model 
            do_this = False
            if do_this:
                m_clustering='kNN' #'random forest'
                if m_clustering=='random forest':
                    m = sk.ensemble.RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2,
                                                     min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='sqrt', 
                                                     max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, 
                                                     oob_score=False, n_jobs=None, random_state=None, verbose=0, 
                                                     warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None)
                if m_clustering=='kNN':
                    m = sk.ensemble.BaggingClassifier(sk.neighbors.KNeighborsClassifier(),
                                 max_samples=0.5, max_features=0.5)
                #Tensor.data.cpu().numpy()   
                m = m.fit(tr_X2train.data.cpu().numpy(), nio_best.data.cpu().numpy())
                par['clustering_model']=m
            
                do_cv_test=True
                if do_cv_test:
                    print('cv test of X -> Z predictor')
                    # scoring='precision_micro' , scoring='f1_micro', scoring='accuracy' ['precision_samples', 'accuracy']
                    scores = sk.model_selection.cross_validate(m, tr_X2train.data.cpu().numpy(), y=nio_best.data.cpu().numpy(), groups=None, 
                                                               scoring=['f1_micro', 'accuracy','recall_micro','precision_micro',], cv=10,
                                                           n_jobs=None, verbose=0, fit_params=None, pre_dispatch='2*n_jobs',
                                                           return_train_score=True, return_estimator=False)
                    print(scores)
                    df_scores=pd.DataFrame(scores).to_hdf('sores_x_to_z_model.hdf', 'scores')
            
        else:
            Ldnml_best_jumps=Lmin
            nio_best_jumps = nio_min.clone()
            nio_best = nio_min.clone()
            Ldnml_best = Lmin
    else:
        Ldnml_best_jumps=Lmin
        nio_best_jumps = nio_min.clone()
        nio_best = nio_min.clone()
        Ldnml_best = Lmin
    #print('nio_min',nio_min)

    #print('Lmin',Lmin)
    #print('Ldnml_best_jumps',Ldnml_best_jumps)        

    #print('do_DNML_torch 1')

    # assign new o-cluster identity for each data sample
    zo_opt = torch.zeros_like(tr_X2train_cl[:,0])
    for i2 in range(nI):  
        zo_opt[zi==i2]=nio_best[i2]

    #zio_opt_l = [torch.nonzero(zo_opt==o)[0] for o in range(n_c)]  


    # find complete L_DNML = LL + g + Z + sum(Cx,o) + Cz(K) 
    if (dn_jumps>0)&(do_simplified_tree==False):
        # #1 new w, lambda 
        #par['nit']=n_iter
        # Wp, Lambdas, Sigmap, Lo2, Temp2, _, dy2_g
        Wp2, Lambdas2, Sigmap2, Lo2_y, Temp2, _ = fit_DNML_fixed_zo(tr_y2train,tr_X2train,[zo_opt],lambdas, [sigmap], par)
        Wp2_x, Lambdas2_x, Sigmap2_x, Lo2_x, Temp2_x, _ = fit_DNML_fixed_zo(tr_y2train_x,tr_X2train,[zo_opt],lambdas_x, [sigmap_x], par)

        #2 add + sum(Cx,o) + Cz(K) , Cx=0 for p(o|i)={0,1} 
        #Cz = C_Z(n_c=n_c, n_samples=n_samples) #(OxI,n_samples)        
        L_z, _, _, _, _, _, par  = fit_NML_zo(zo_opt, par=par)            
        Lout = Lo2_y +Lo2_x + L_z # repair Lo2!!!!!!!!!!!!!!!!!!!!!!!!!!!!! - Z should be different

        Out = {'Wp2': Wp2, 'Lambdas2': Lambdas2,'Sigmap2':Sigmap2, 'zo_optimal': zo_opt, 'L_optimal':Lout, 'nio_optimal':nio_best,
              'Ldnml_best_jumps':Ldnml_best_jumps,  'OxI':OxI,
              'Ldnml_best_merge':Lmin, 'nio_best_merge':nio_min, 'Wp': Wp, 'Lambdas': Lambdas,'Sigmap':Sigmap, 'L_z': L_z, 'p_z':par['p_z'],
              'Wp_x': Wp_x, 'Lambdas_x': Lambdas_x,'Sigmap_x':Sigmap_x}
    else:    
        #2 add + sum(Cx,o) + Cz(K) , Cx=0 for p(o|i)={0,1} 
        #Cz = C_Z(OxI,n_samples)
        L_z, _, _, _, _, _, par  = fit_NML_zo(zo_opt, par=par)              
        Lout = Lmin + L_z # repair Lo2!!!!!!!!!!!!!!!!!!!!!!!!!!!!! - Z should different

        Out = {'Wp2': Wp, 'Lambdas2': Lambdas,'Sigmap2':Sigmap, 'zo_optimal': zo_opt, 'L_optimal':Lout, 'nio_optimal':nio_best,
        'Ldnml_best_jumps':Ldnml_best_jumps, 'OxI':OxI, 'zi': zi,
        'Ldnml_best_merge':Lmin, 'nio_best_merge':nio_min,
        'Wp': [], 'Lambdas': [],'Sigmap':[], 'L_z': L_z, 'p_z':par['p_z'], 'Temp2':Temp2}
    # to do: 
    # best nio structure? - EM estimator
    # jumps?
    # restrict min(nio) 
    # Z update?
    # C_Z function
    # prediction function
    # test uMDL regression function
    # test DNML regression

    return Out, par


################################################################################
#######
####
def fit_UMDL(tr_y2train,tr_X2train,tr_X2train_cl,lambdas, sigmap, par, cli=0):
    #par = {'nit': nit, 'relTol': it_stop, 'alpha_sigma': alpha_sigma, 'beta_sigma': beta_sigma,
    #             'alpha_lambda': alpha_lambda, 'beta_lambda': beta_lambda, 'n_jumps': n_jumps, 'dn_jumps': dn_jumps, 'device':cuda0 }
    cuda0 = par['device'] 

    n_c = torch.max(tr_X2train_cl[:,cli])+1
    n_dim = tr_X2train.shape[1]
    n_samples = tr_X2train.shape[0]
    n_out = tr_y2train.shape[1]

    zo = tr_X2train_cl[:,cli]

    # find uMDL ridge regression weights for each output cluster o (q: for several n(o|i) sets ?)
    #Lambdas = []
    Lambdas = torch.zeros([n_dim, n_c], dtype=torch.float, device=cuda0)
    Wp = []
    Sigmap = []
    Lo0=[]

    #print(n_c) 

    for o in range(n_c):
        #Co = torch.sum(Ci[i4o[o]]).t()
        #C = torch.sum(tr_X2train*tr_X2train).t()
        n_samples = tr_X2train.shape[0]
        #wp, lambdas2, sigmap2, L = torchDNML(tr_y2train[zio[o],:], tr_X2train[zio[o],:], tr_X2train_cl[zio[o],:], lambdas, sigmap, par)
        wp, lambdas2, sigmap2, L = torchuMDL(tr_y2train[zo==o,:], tr_X2train[zo==o,:], lambdas, sigmap, par=par)

        Wp = Wp + [wp]
        #Lambdas = Lambdas + [lambdas2]
        Lambdas[:,[o]] = lambdas2
        Sigmap = Sigmap + [sigmap2]
        Lo0 = Lo0 + [L]

    Out = {'Wp': Wp, 'Lambdas': Lambdas,'Sigmap':Sigmap,  'Lo':Lo0}
    return Out


def data_preprocessing(X_train0, N_bootstraps=1, Dn=1, stp_n=[], classes_columns_train=[], ge_columns_train=[],annot_columns_train=[],
                       stp_columns_train=[], do_filter_gene_set = False, do_normalize = 1, do_log_y = 1,
                       do_remove_scvi_latent_factors=False, do_remove_non_scvi_latent_factors=False,df_ge_names_filter=[],lf_scvi_names=[],
                       remove_st=[],d_log=0.3, gene_set_names='', probab_dict=None):
    
    '''
    Selection and preprocessing of genes_STP data before model training
    '''
    #Dn = 1
    #N_bootstraps = 100 #200
    Dn2 = int(N_bootstraps/Dn) # 100 - should be a number of bootstraps per synapse type !!!

    # 0ll

    ####
    ####   SELECT X
    ####
    cla_n = classes_columns_train #['ex_inh']
    cla_n2 = pd.Series(cla_n)

    #ge_n = imp50.index[0:25].tolist()
    #ge_n = lf_scvi_names #ge_columns_train2

#     df_ge_names_iRF50 = pd.read_excel(d4+'iRF_found_50_best_genes.xlsx',header=None)
#     ipost = df_ge_names_iRF50.iloc[:,0].str.contains('post_')
#     ge_names_irf = df_ge_names_iRF50.iloc[:,0]
#     ge_names_irf.loc[ipost] = 'post__' + ge_names_irf.str.split('post_',expand=True).loc[:,1].loc[ipost].copy() 

    ge_columns_train2 = pd.Series(ge_columns_train)
    # ACHTUNG!!! # ACHTUNG!!! # ACHTUNG!!! # ACHTUNG!!! # ACHTUNG!!! # ACHTUNG!!! # ACHTUNG!!!
    do_filter_gene_set = False
    do_remove_scvi_latent_factors=False
    do_remove_non_scvi_latent_factors=False
    ge_columns_train2 =ge_columns_train2.loc[ge_columns_train2.isin(['samples_pre', 'samples_post'])==False]
    if do_filter_gene_set==True:
        
        #d5 = '/content/drive/My Drive/Colab Notebooks/'
        #df_ge_names_filter = pd.read_excel(d5+'gene_set_names.xlsx',header=None).loc[1:,1]
        df_ge_names_filter = pd.read_excel(gene_set_names,header=None).loc[1:,1]
        #ge_n = df_ge_names_iRF50.iloc[:,0].values #ge_columns_train2 #ge_columns_train #imp50.index[0:25].tolist()
        
        print('size of filter gene set ',df_ge_names_filter.shape)
        ge_columns_train2 = ge_columns_train2.loc[ge_columns_train2.isin(df_ge_names_filter)]
    if do_remove_scvi_latent_factors==True:
        ge_columns_train2 =ge_columns_train2.loc[ge_columns_train2.isin(lf_scvi_names)==False].values  # ACHTUNG!!! remove scvi-latent facrors !!!!!
    if do_remove_non_scvi_latent_factors==True:
        ge_columns_train2 =ge_columns_train2.loc[ge_columns_train2.isin(lf_scvi_names)==True].values  # ACHTUNG!!! remove all feature with except scvi-latent facrors !!!!!
    print('size of final gene set ',ge_columns_train2.shape)
    # ACHTUNG!!! # ACHTUNG!!! # ACHTUNG!!! # ACHTUNG!!! # ACHTUNG!!! # ACHTUNG!!! # ACHTUNG!!!

    

    ge_n = ge_columns_train2

    #X2 = X_train0.loc[:,annot_columns_train + ge_n + cla_n]
    # pure cortex 
    #X3 = X_train0.iloc[0:5300,: ]
    # pure hipp
    #X3 = X_train0.iloc[5300:,: ]

    all_samples = np.ones(X_train0.shape[0])
    for st in remove_st:
        ii = st*N_bootstraps + np.arange(N_bootstraps)
        all_samples[ii] = 0
    X3 = X_train0.iloc[all_samples==1,:]
    Dn3 = int(X3.shape[0]/N_bootstraps) # number of synapse_types
    
    if probab_dict is None:
        probab_y2 = np.array([])
        y2_syntp  = np.array([])
    else:
        probab_y2 = probab_dict['probab_y2']
        y2_syntp  = probab_dict['y2_syntp']
        y2_syntp = y2_syntp[all_samples==1,:]
        all_samples2 = np.ones(probab_y2.shape[0])
        all_samples2[remove_st] = 0
        probab_y2 = probab_y2[all_samples2==1,:]
    
    # 100ll
    X2    = X3.loc[:,ge_n ]
    X2_cl = X3.loc[:,cla_n ]
    X2_an = X3.loc[:,annot_columns_train ]
    #i_cl = np.nonzero(X2.columns.isin(cla_n))[0]
    nannot = len(annot_columns_train)
    lge_n  = len(ge_n)
    if len(cla_n)>0:
        i_cl = np.nonzero(X2.columns.isin(cla_n))[0] - nannot
        #X2.loc[:,cla_n] = X2.loc[:,cla_n]+1

    X2=X2.iloc[0:X_train0.shape[0]:Dn,:].values
    X2_cl=X2_cl.iloc[0:X_train0.shape[0]:Dn,:].values
    X2_an=X2_an.iloc[0:X_train0.shape[0]:Dn,:].values


    #stp_n = ['A2_20Hz','A3_20Hz','A4_20Hz','A5_20Hz','A2_50Hz','A3_50Hz','A4_50Hz','A5_50Hz']
    #stp_n = ['A5_20Hz']

    iy0=np.nonzero(np.array(stp_columns_train)==stp_n[0])[0]
    #  'A2_20Hz',
    #  'A5_20Hz',
    #  'A250_20Hz',
    #  'A1000_20Hz',
    #  'A2_50Hz',
    #  'A2_10Hz',
    #  'A5_50Hz',
    #  'A5_10Hz',
    y2 =  X3.loc[:,stp_n].iloc[0:X_train0.shape[0]:Dn,:].values


    #do_normalize = 1
    #do_log_y = 1
    #from sklearn import preprocessing
    if do_log_y==1:
        #d_log=0.3
        y2 = np.log(y2.astype(float)+d_log)

    if do_normalize==1:

        scale_y2 = np.std(y2[:,:].astype(float),axis=0)
        X2[:,:] = preprocessing.scale(X2[:,:])
        y2[:,:] = y2.astype(float)/scale_y2
        mean_y2 = np.mean(y2[:,:] , axis=0)
        y2[:,:] = y2[:,:] - mean_y2
        # X2[:,:] = preprocessing.scale(X2[:,:])
        # y2[:,:] = preprocessing.scale(y2[:,:])
    
#     #mod_index = modf.index #[3] # modf.index #[0, 3, 9, 11, 13, 15]
#     sts = pd.DataFrame(X3.iloc[0:X3.shape[0]:Dn2,:].iloc[:,0:nannot],columns = annot_columns_train)
#     sts = sts['cell_type2_pre'].map(str)+'_'+  sts['layer_pre'].map(str)+' -> '+  sts['cell_type2_post'].map(str)+'_'+sts['layer_post'].map(str)
#     #sts.to_excel('temp.xlsx')
    sts = []
    if probab_dict is None:
        y2_syntp_sub = np.array([])
    else:    
        y2_syntp_sub = y2_syntp[0:X_train0.shape[0]:Dn,:]
        
    preprocessing_ = {'do_log_y':do_log_y, 'do_normalize':do_normalize,
                      'd_log':d_log, 'scale_y2':scale_y2, 'mean_y2':mean_y2,
                      'probab_y2':probab_y2, 'y2_syntp':y2_syntp, 'y2_syntp_sub':y2_syntp_sub, 'Dn2':Dn2, 'Dn3':Dn3}
    
    return  X2, y2, X2_cl, X2_an, sts, preprocessing_, cla_n2  


def train_and_test_regression_models(X2, y2, X2_cl, X2_an, H_Models, preprocessing_, ncv = 10, stp_n=[], sts=[], cla_n2=[], cuda0=None):
    '''
    Train and test regression model for genes->STP
    '''
    t1 =time.time() 
    
    do_log_y = preprocessing_['do_log_y']
    do_normalize = preprocessing_['do_normalize']
    if do_log_y==1:
        d_log = preprocessing_['d_log']
        
    if do_normalize==1:    
        scale_y2 = preprocessing_['scale_y2']
        mean_y2 = preprocessing_['mean_y2']
#     #mod_index = modf.index #[3] # modf.index #[0, 3, 9, 11, 13, 15]
#     sts = pd.DataFrame(X3.iloc[0:X3.shape[0]:Dn2,:].iloc[:,0:nannot],columns = annot_columns_train)
#     sts = sts['cell_type2_pre'].map(str)+'_'+  sts['layer_pre'].map(str)+' -> '+  sts['cell_type2_post'].map(str)+'_'+sts['layer_post'].map(str)
#     #sts.to_excel('temp.xlsx')
    Y_pred = []
    Y_pred0 = []
    Samples_test = []
    OUT = []
    for i,mdn in enumerate(H_Models.loc[:,'name']):

        # HM_name = 'Ms1c2'
        md = H_Models.loc[H_Models.loc[:,'name']==mdn, :]


        #md = modf.loc[i,:]
        #mdn = md['name']
        print(mdn)


        if type(md['structure'].values[0][0])==str:
            cli = cla_n2.index[cla_n2.isin(md['structure'].values[0])].values
            ncli = 1
        else:
            cli=[]
            cli2=md['structure'].values[0]
            ncli = len(cli2)
            for iii in range(len(cli2)):
                cli = cli + [cla_n2.index[cla_n2.isin(cli2[iii])].values]

        #mdcl2 = np.char.strip(np.array(str.split(md['classes_post'],',')))
        #mdcl1 = np.char.strip(np.array(str.split(md['classes'],','))) 

     # 300ll   
        X3_cl = np.copy(X2_cl)
        if ncli==1:
            cli5 = [cli]
        else:
            cli5 = cli

        print(set(X2_cl[:,cli5[0][0]]))
        if len(cli)>1:
            print(set(X2_cl[:,cli5[0][1]]))
        else:
            print('all')

        for iii in range(ncli): 
            cli6 = cli5[iii]
            for icl, cli2 in enumerate(cli6):
                if icl==0:
                    n2n=pd.DataFrame(list(set(X2_cl[:,cli2]))).reset_index().set_index(0)
                    X3_cl[:,cli2] = n2n.loc[X2_cl[:,cli2]].values.ravel()
                    cli20 = cli2
                else: 
                    # convert class-names to numbers for cli2
                    n2n=pd.DataFrame(list(set(X2_cl[:,cli2]))).reset_index().set_index(0)
                    X4_cl = n2n.loc[X2_cl[:,cli2]].values.ravel()

                    # combine cli2 and cli20 classes
                    X4_cl = X4_cl + n2n.shape[0]*X3_cl[:,cli20]
                    n2n = pd.DataFrame(list(set(X4_cl))).reset_index().set_index(0)
                    X3_cl[:,cli2] = n2n.loc[X4_cl].values.ravel()
                    cli20=cli2
        #### cli??        
        y_pred = np.zeros((0,y2.shape[1]))
        y_pred0 = np.zeros((y2.shape[0],y2.shape[1]))


        #ncv = 10
        #ncv = 10
        #n_samp_cv = np.rint(X2.shape[0]/ncv)
        #samples_all = np.arange(X2.shape[0])
        r2cv  = np.zeros(ncv+1)
        r3cv  = np.zeros(ncv+1)
        r4cv  = np.zeros(ncv+1)
        nonzs = np.array([])
        Lj=[]
        # DNML
        SSEy_dnml_tra = []
        SSEy_dnml_gen = []
        y_pred_dnml = []
        for icv in range(ncv+1): #range(ncv):  # cross-validation cycle
            if icv<ncv-1:
                samples_test = (np.arange(n_samp_cv) + icv*n_samp_cv).astype(int)
                samples_train = np.delete(np.copy(samples_all),samples_test)
            else:  
                n_samp_cv2 = int(n_samp_cv/2)
                samples_test = (np.arange(n_samp_cv2 ) + (ncv-1)*n_samp_cv+(icv-ncv+1)*n_samp_cv2  ).astype(int)
                samples_train = np.delete(np.copy(samples_all),samples_test)
            X2train, y2train, X2train_cl = X2[samples_train,:], y2[samples_train,:], X3_cl[samples_train,:]
            X2test, y2test, X2test_cl = X2[samples_test,:], y2[samples_test,:], X3_cl[samples_test,:]


            if (mdn!='BHLM')&(mdn!='HLM'):

                alpha_1  = 1e-6
                alpha_2  = 1e-6
                lambda_1 = 1e-6
                lambda_2 = 1e-6
                threshold_lambda=10000.0,


                #model, support = fit_classes_tree(X2train,y2train,X2train_cl,cli,model_type='HuberRegressor',
                #                                  nmin = 0,n_iter=100, alpha=1)
                #model, support = fit_classes_tree(X2train,y2train,X2train_cl,cli,model_type='ARDRegression',
                #                                  nmin = 0,n_iter=300, alpha_1=1e-06, alpha_2=1e-06,
                #                                  lambda_1=1e-06, lambda_2=1e-06)
                model, support = fit_classes_tree(X2train,y2train,X2train_cl,cli,model_type='elastic_net',
                                                   nmin = 0, alpha=0.5, l1_ratio=0.01)
                #model, support = fit_classes_tree(X2train,y2train,X2train_cl,cli,model_type='ridge',nmin = 0, alpha=1)
                # model, support = fit_classes_tree(X2train,y2train,X2train_cl,cli,model_type='ARDRegression',
                #                                    nmin = 0, alpha_1=alpha_1, alpha_2=alpha_2,
                #                                    lambda_1=lambda_1, lambda_2=lambda_2)

                y_pred_i = predict_classes_tree(model,X2test,X2test_cl,cli,nout=y2.shape[1],nmin = 0)
                for iy in range(y2.shape[1]):   # n_out = y2.shape[1]
                    nonz = y_pred_i[:,iy]!=0
                    y_pred_i[nonz==False,iy] = np.mean(y2[:,iy])
                y_pred = np.concatenate([y_pred, y_pred_i],axis=0)

            #elif (mdn=='BHLM'):

            if mdn=='HLM':
                if icv==0:
                    X4_cl_df = pd.DataFrame(X3_cl) #X3_cl_df.copy()
                    #X4_cl_df.loc[:,cli[1][1]] = X4_cl_df.loc[:,cli[1][1]]+X4_cl_df.loc[:,cli[2][1]].max()+1
                    X4_cl_df.loc[:,cli[0][1]] = X4_cl_df.loc[:,cli[0][1]]+X4_cl_df.loc[:,cli[1][1]].max()+1
                    X4_cl_df = X4_cl_df.loc[:,[cli[0][1],cli[1][1]]] #,cli[2][1]]]
                    X4_cl = X4_cl_df.values

                    #X4_cl = X4_cl # 1 synapse type model 
                    #X4_cl = X4_cl[:,[0,1,2]]
                    #X4_cl = X4_cl[:,[0,1]]
                    X4_cl = X4_cl[:,[0,1]] # ACHTUNG!!! - contradiction?: for training X4_cl should be - rows of cluster tree , for testing - [zo, zi]

                    nc = np.max(X4_cl[:,[0,1]])+1
                    X4_cl = np.concatenate([nc*np.ones([X4_cl.shape[0],1]), X4_cl ], axis=1) # add root cluster

                    scl4 = list(set(X4_cl[:,-1])) # ACHTUNG!!! - assumption - input clusters are in the last columns
                    # tree of clusters
                    #Oi = [np.array([0,3,4]), np.array([1,3,4]), np.array([2,4])] # parents of each leaf cluster
                    nI = len(scl4)
                    Oi = [] # Oi - list: for each input clusters - indices of all parent clusters
                    for iclu in range(nI):
                        is_iclu = X4_cl[:,-1]==scl4[iclu]
                        x4_iclu = X4_cl[is_iclu,:]
                        Oi = Oi + [np.flip(x4_iclu[0,:]).astype('int')]

                    #M = {'W': W, 'Oi': Oi, 'R': R, 'RI': RI, 'Ki': Ki, 'Sig': Sig, 'Tree':Tr} #, 'n_samples': n_samples}
                    M = {'Oi':Oi}  
                    #M['Oi'] = Oi

                    #c_dim2 = len(set(X4_cl[:,-1])) # number of smallest clusters
                    #c_dim1 = X4_cl.shape[1]
                    #c_dim = X4_cl.max()+1

                    output_dim = y2train.shape[1]
                    input_dim = X2train.shape[1]
    # 400ll                

                    X4train_cl, X4test_cl = X4_cl[samples_train,:], X4_cl[samples_test,:]



    #             ########## fit model
                ##################################################################
                X4train_cl, X4test_cl = X4_cl[samples_train,:], X4_cl[samples_test,:]

                # add intercept
                do_intercept = 1
                if do_intercept==1:
                    X2train = np.concatenate([X2train, np.ones([X2train.shape[0],1])], axis=1)
                    X2test = np.concatenate([X2test, np.ones([X2test.shape[0],1])], axis=1)
                    #ng = X2train.shape[1]
                    input_dim = X2train.shape[1]

                # convert data to pytorch variables                                                                             
                # y_pred_i = np.copy(linear_reg_model.forward(x_data_test,x_dim).data.cpu().numpy())
                tr_y2train = Variable(torch.from_numpy(y2train.astype('float32')).to(cuda0))
                tr_X2train   = Variable(torch.from_numpy(X2train.astype('float32')).to(cuda0))
                tr_X2train_cl = Variable(torch.from_numpy(X4train_cl.astype('long')).to(cuda0)) 

                tr_X2test    = Variable(torch.from_numpy(X2test.astype('float32')).to(cuda0))
                tr_X2test_cl = Variable(torch.from_numpy(X4test_cl.astype('long')).to(cuda0)) 

                # convert parameters to pytorch variables
                md_par=md['parameters'].values[0]
                sig = md_par['sig'] #0.4 # 0.4 - up to median R2 = 65%, nit=15; R2=70% nit=5  for scvi_lf? #
                lyambda0 = md_par['lyambda0'] #50 #50 #0.5 -?bad?? # 50 - up to median R2 = 65%, nit=15; R2=70 nit=5 for scvi_lf? #
                ng = X2train.shape[1]
                lambdas = Variable(torch.from_numpy((lyambda0*np.ones((ng,1))).astype('float32')).to(cuda0)) #0.5*torch.tensor(np.ones(ng,1))
                sigmap  = Variable(torch.from_numpy(np.array(sig).astype('float32')).to(cuda0))
                #sigmap  = Variable(torch.from_numpy((np.ones((ng,1))*sig).astype('float32')).to(cuda0))

    # 357ll
                # set parameters for weights regression regularization
                s2=2*sigmap**2
                alpha_sigma  = md_par['alpha_sigma_factor']*sigmap # 0.01*sigmap
                beta_sigma   = md_par['beta_sigma_factor']*sigmap # 2.0*sigmap
                alpha_lambda = md_par['alpha_lambda_factor']*lyambda0*s2 #0.0000001*lyambda0*s2
                beta_lambda  = md_par['beta_lambda_factor']*1.0*lyambda0*s2 #1.0*lyambda0*s2

                nit=5         # number of iterations uMDL
                it_stop = md_par['it_stop'] #0.000001 # relative dlyambda for stop uMDL
                n_jumps = md_par['n_jumps'] #0 #10 #
                dn_jumps = md_par['dn_jumps'] #0 #2. # if dn_jumps==0 - just fit all n_o clusters, otherwise - merge sequentially , make random jumps, refit
                #par = {'nit': nit, 'relTol': it_stop} #, 'index_zi': zi, 'Ci': Ci}
                par = {'nit': nit, 'relTol': it_stop, 'alpha_sigma': alpha_sigma, 'beta_sigma': beta_sigma,
                      'alpha_lambda': alpha_lambda,  'beta_lambda': beta_lambda, 'n_jumps': n_jumps, 'dn_jumps': dn_jumps,
                       'device':cuda0, 'do_uMDL':True } # CHECK!!! - 'do_uMDL':True to do uMDL instead of elastic_net

                do_dnml = True
                if do_dnml==True:
                    #tr_y2train,tr_X2train,[Zo_jump],Lambdas, Sigmap, par
                    Out, par = fit_DNML(tr_y2train,tr_X2train,tr_X2train_cl,M,lambdas, sigmap, par)

                    nio_best = Out['nio_optimal']
                    zo_opt = Out['zo_optimal']
                    #Wp2 = Out['Wp2']
                    Wp2 = Out['Wp2']
                else:    
                    cli = tr_X2train_cl.shape[1]-1 
                    Out = fit_UMDL(tr_y2train,tr_X2train,tr_X2train_cl,lambdas, sigmap, par, cli=cli)
                    Wp2 = Out['Wp']



                # DNML Regression: test  
                #y_pred_dnmli0 = np.copy(( tr_X2train @ wp).data.cpu().numpy())   
                #y_pred_dnmli = np.copy(( tr_X2test @ wp).data.cpu().numpy()) 
                n_out = y2.shape[1] # output dimensionality
                # predictDNML( X, n_out, Wp, Sigmap=[], Wp_x=[], Sigmap_x=[], p_z=[], X_cl=None, cl_zo=0, cl_zi=-1, Loxi=None, OxI=None, par={})
                if do_dnml==True:
                    tr_X_out_train_cl = torch.clone(tr_X2train_cl) # ACHTUNG!!! - contradiction?: for training X4_cl should be - rows of cluster tree , for testing - tr_X_out_train_c : [zo,..., zi]
                    tr_X_out_train_cl[:,0] = zo_opt   # ACHTUNG!!! - tr_X2train_cl.shape[1] should be >1 (tr_X2train_cl[:,1] = zi)
                    if (dn_jumps>0): #&()
                        y_pred_dnmli0 = np.copy(predictDNML( tr_X2train , n_out,Wp2, X_cl=tr_X_out_train_cl,par=par).data.cpu().numpy()) 
                    else:
                        y_pred_dnmli0 = np.copy(predictDNML( tr_X2train , n_out,Wp2, Sigmap=Out['Sigmap'], Wp_x=Out['Wp_x'], Sigmap_x=Out['Sigmap_x'], p_z=Out['p_z'],
                                                            X_cl=tr_X_out_train_cl,par=par).data.cpu().numpy())
                else: 
                    y_pred_dnmli0 = np.copy(predictUMDL( tr_X2train , n_out,Wp2, X_cl=tr_X_out_train_cl,par=par).data.cpu().numpy())      

                tr_X_out_test_cl = torch.clone(tr_X2test_cl) 
                zi_test = tr_X2test_cl[:,-1]
                zo_opt_test = torch.zeros_like(tr_X2test_cl[:,0]) # zo_opt_test
                for i2 in range(nI):  
                  #zo_opt_test[zi_test==i2]=nio_best[i2] # best clusters zo
                  zo_opt_test[zi_test==i2]=i2  # only input clusters zo!!! should be modified to use DNML-found best output clusters (nio_best[i] for each input cluster i)
                tr_X_out_test_cl[:,0] = zo_opt_test 
                if (dn_jumps>0): #&()
                    y_pred_dnmli = np.copy(predictDNML( tr_X2test, n_out,Wp2, Sigmap=Out['Sigmap'], Wp_x=Out['Wp_x'], Sigmap_x=Out['Sigmap_x'], p_z=Out['p_z'],
                                                       X_cl=tr_X_out_train_cl, OxI = Out['OxI'],par=par).data.cpu().numpy())
                else:    
                    y_pred_dnmli = np.copy(predictDNML( tr_X2test, n_out,Wp2, X_cl=tr_X_out_train_cl, OxI = Out['OxI'],par=par).data.cpu().numpy())

                #ACHTUNG!!! - temporary use mean(y2) for nontrained input clusters 
                for iy in range(n_out): #(y2.shape[1]):   # n_out = y2.shape[1]
                    nonz = y_pred_dnmli[:,iy]!=0
                    y_pred_dnmli[nonz==False,iy] = np.mean(y2[:,iy])
                #y_pred = np.concatenate([y_pred, y_pred_i],axis=0)


                print('nio_best',nio_best)

                # correct for x-shift
                #y_pred_umdli =y_pred_umdli  + (bx_test - bx)@w2
                #y_pred_i0 =y_pred_i0  + (bx - bx)*w2

                #y_pred_dnml = y_pred_dnml + [y_pred_dnmli]
                #y_pred_dnml0 = y_pred_dnml0 + [y_pred_dnmli0]

                SSEy_dnml_tra = SSEy_dnml_tra +[1 - np.mean((y_pred_dnmli0-y2train)**2)/np.var(y2train)]
                SSEy_dnml_gen = SSEy_dnml_gen +[1 - np.mean((y_pred_dnmli-y2test)**2)/np.var(y2test)]

                y_pred_i = np.copy(y_pred_dnmli)
                y_pred_i0 = np.copy(y_pred_dnmli0)

    #             # reorder y_data!!! : x_data were reordered at 634
    #             y_pred_i[i_cl_test,:] = np.copy(y_pred_i)

    #             #y_pred_i = linear_reg_model.forward(x_data_test).data.cpu().numpy()

                y_pred = np.concatenate([y_pred, y_pred_i.reshape([-1,1])],axis=0)
                y_pred0[samples_train,:] = np.copy(y_pred_i0)



                ################################################################

                do_plot_each_cv=1
                iy=0
                if do_plot_each_cv==1:
                    f, ax =plt.subplots(figsize=(15, 6))
                    ##f, ax = plt.figure()
                    ##ax = f.add_axes()

                    #plt.title(stp_columns[iy]+", model : "+mdn+", cv : "+str(icv))
                    #plt.title(stp_columns_train[iy]+", model : "+mdn)
                    plt.title(stp_n[iy]+", model : "+mdn)

                    #ax.set_title(stp_columns[i]+' '+', subclass out of bag: '+str(i0))
                    #yy1.loc[:,[stpn_test[i],stpn_pred[i]]].plot(ax=ax)
                    plt.plot(y2[:,iy].ravel(),'b')
                    plt.plot(y_pred[:,iy].ravel(),'r')
                    plt.plot(y_pred0[:,iy].ravel(),'g')

                    # plt.plot(y2[:,iy].ravel(),'ob')
                    # plt.plot(y_pred[:,iy].ravel(),'xr')
                    # plt.plot(y_pred0[:,iy].ravel(),'xg')
                    plt.pause(0.05)


            #if do_log_y==1:  #recover stp data from logarithmic scale
            y2l = np.copy(y2)
            y_pred_il = np.copy(y_pred_i)
            y_pred_i0l = np.copy(y_pred_i0)
            y2testl = np.copy(y2test)
            y2trainl = np.copy(y2train)
            if do_normalize==1:
                #scale_y2 = np.std(y2[:,:],axis=0)
                y2l = (y2l+mean_y2)*scale_y2
                y_pred_il = (y_pred_il+mean_y2)*scale_y2
                y_pred_i0l = (y_pred_i0l+mean_y2)*scale_y2
                y2testl = (y2testl+mean_y2)*scale_y2
                y2trainl = (y2trainl+mean_y2)*scale_y2

            if do_log_y==1:  #recover stp data from logarithmic scale
                #d_log=0.3
                y2l = np.exp(y2l.astype(float))-d_log
                y2testl = np.exp(y2testl.astype(float))-d_log
                y2trainl = np.exp(y2trainl.astype(float))-d_log
                y_pred_il = np.exp(y_pred_il.astype(float))-d_log
                y_pred_i0l = np.exp(y_pred_i0l.astype(float))-d_log            

            iy=0 #iy0
            if np.var(y2[:,iy])!=0:
                nonz=[]
                nonzs = np.append(nonzs, nonz)
                #if np.sum(nonz)>0:
                # !!!!!!!!!!!!!!!!
                #!!!!  var(y2) should be with nonz !!!
                y22test = y2testl[:y_pred_i.shape[0],iy]
                r2cv[icv]=1 - np.mean((y22test[:] - y_pred_il[:,iy])**2)/np.var(y2l[:,iy])
                #r2cv[icv]=1 - np.mean((y2test[nonz,iy] - y_pred_i[nonz,iy])**2)/np.var(y2[:,iy])
                #r3cv[icv]=1 - np.median(np.abs(y2test[nonz,iy] - y_pred_i[nonz,iy])**2)/np.var(y2[:,iy])
                r3cv[icv]=1 - np.median(np.abs(y2testl[:,iy] - y_pred_il[:,iy])**2)/np.var(y2l[:,iy])
                r4cv[icv]=1 - np.mean((y2trainl[:,iy] - y_pred_i0l[:,iy])**2)/np.var(y2l[:,iy])#  np.mean((y2test[nonz,iy] - y_pred_i[nonz,iy])**2)/np.var(y2test[nonz,iy])

        Y_pred = Y_pred + [y_pred]
        Y_pred0 = Y_pred0 + [y_pred0]
        Samples_test = Samples_test +[samples_test]
        OUT = OUT + [Out]

        if do_log_y==1:  #recover stp data from logarithmic scale
            y2l = np.copy(y2)
            y_predl = np.copy(y_pred)
            y_pred0l = np.copy(y_pred0)
            if do_normalize==1:
                #scale_y2 = np.std(y2[:,:],axis=0)
                y2l = (y2l+mean_y2)*scale_y2
                y_predl = (y_predl+mean_y2)*scale_y2
                y_pred0l = (y_pred0l+mean_y2)*scale_y2

            #d_log=0.3
            y2 = np.exp(y2l.astype(float))-d_log
            #y2test = np.exp(y2test.astype(float))-d_log
            #y2train = np.exp(y2train.astype(float))-d_log
            y_pred = np.exp(y_predl.astype(float))-d_log
            y_pred0 = np.exp(y_pred0l.astype(float))-d_log


        for iy in range(y2.shape[1]):
            f, ax =plt.subplots(figsize=(16, 4))
            ##f, ax = plt.figure()
            ##ax = f.add_axes()

            #plt.title(stp_columns[iy]+", model : "+mdn+", cv : "+str(icv))
            #plt.title(stp_columns_train[iy]+", model : "+mdn)
            plt.title(stp_n[iy]+", model : "+mdn)

            #ax.set_title(stp_columns[i]+' '+', subclass out of bag: '+str(i0))
            #yy1.loc[:,[stpn_test[i],stpn_pred[i]]].plot(ax=ax)

            #plt.plot(y2[:,iy].ravel(),'ob')
            #plt.plot(y_pred[:,iy].ravel(),'xr')
            #plt.plot(y_pred0[:,iy].ravel(),'xg')

            plt.plot(y2[:,iy].ravel(),'b')
            plt.plot(y_pred[:,iy].ravel(),'r')
            plt.plot(y_pred0[:,iy].ravel(),'g')

            #plt.xticks(np.arange(len(yy1.index)), yy1.index, rotation=90)

           # f, ax =plt.subplots(figsize=(16, 4))
           # plt.plot(y_pred0[:,iy].ravel(),y2[:,iy].ravel(),'g')
           # plt.plot(y_pred[:,iy].ravel(),y2[:,iy].ravel(),'b')

            temp = Out['Temp2'][0]
            #temp = {'y':np_y, 'x':np_x, 'iszo':is_zo, 'wp':wp}
            np_y=temp['y']
            np_x=temp['x']
            f, ax =plt.subplots(figsize=(16, 4)) 
            if type(np_y) != type(list()):
                l1_ratio=0.01
                alpha = 0.5
                mode_l= sk.linear_model.ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=True,
                                                normalize=False,
                                                max_iter=1000, copy_X=True, tol=0.0001, warm_start=False,
                                                random_state=None, selection='cyclic').fit(np_x,np_y)
                #wp = np.copy(mode_l.coef_ )
                plt.plot(np_y[:].ravel(),'b')
                #plt.plot(y_pred2[:].ravel(),'r')   



        iy=0 #iy0

        R3 = []

        for iy in range(y2.shape[1]):
            r2=0
            r3=0
            r4=0
            if np.var(y2[:,iy])!=0:
                y22 = y2[:y_pred.shape[0],iy]
                r2=1 - np.mean((y22[:] - y_pred[:,iy])**2)/np.var(y22[:])
                #r2=1 - np.mean((y22[:] - np.mean(y22[:]))**2)/np.var(y22[:])
                #r2=1 - np.mean((y22[nonzs==1] - y_pred[nonzs==1,iy])**2)/np.var(y22[nonzs==1])
                #r2=1 - np.mean((y22[nonzs==1] - y_pred[nonzs==1,iy])**2)/np.var(y22[nonzs==1])
                #r3=1 - np.median(np.abs(y2[:y_pred.shape[0],iy] - y_pred[:,iy]))/np.median(np.abs(y2[:,iy] - np.median(y2[:,iy])))

                r3=1 - np.median((y22[:] - y_pred[:,iy])**2)/np.median((y22[:]-np.mean(y22[:]))**2)
                r4 = 1 - np.mean((y2[:,iy] - y_pred0[:,iy])**2)/np.var(y2[:,iy]) #np.mean((y22[nonzs==1] - y_pred[nonzs==1,iy])**2)/np.var(y22[nonzs==1])
                #r4=1 - np.var(y_pred[:,iy])/np.var(y2[:,iy])
            R3=R3 + [r3]
            print('\n\n ',stp_n[iy],'\n') 
            print(" R**2_mean = ",r2,
                  "\n R**2_median = ",r3,
                  "\n R**2_mean_nonzero = ",r4,"\n",
                  "  R**2 cv mean 2 = ", np.mean(r2cv),"\n",
                  "  R**2 cv median 2 = ", np.median(r2cv),"\n",
                  "  R**2 cv mean_median = ", np.mean(r3cv),
                  "\n  R**2 cv unnormed mean = ", np.mean(r4cv),
                  "\n  R**2 cv = ", r2cv,
                  "\n  R**2 cv unnormed = ", r4cv,
                  "\n\n")

    # 700ll    
    t2 =time.time()    
    print("Elapsed time ",t2-t1)
    print(nonzs.shape)
    print(nonzs.sum())
    print(R3)   
    
    
    return Y_pred, Y_pred0, Samples_test,  R3, OUT
        

def train_and_test_regression_models_(X2, y2, X2_cl, X2_an, H_Models, preprocessing_, ncv = 10, stp_n=[], sts=[], cla_n2=[], cuda0=None):
    
    t1 =time.time()
    
    do_log_y = preprocessing_['do_log_y']
    do_normalize = preprocessing_['do_normalize']
    if do_log_y==1:
        d_log = preprocessing_['d_log']
        
    if do_normalize==1:    
        scale_y2 = preprocessing_['scale_y2']
        mean_y2 = preprocessing_['mean_y2']
    #
    #.   TEST SIMPLEST HLM WITH ELASTIC NET LEAVES
    #    +GP 
    #
    ##### note: synapse type hierarhy in X(:,classes_columns_train) should be from H_Models!!!


#     # 200ll
#     do_model=0
#     if do_model==0:
#         H_Models = pd.DataFrame([       ['Msb1sc2',['subclass_pre_b', 'subclass_post_c']],
#                                         ],
#                                         columns = ['name','structure'])
#     elif do_model==1:
#         H_Models = pd.DataFrame([       ['BHLM',
#                                         [['class_pre', 'class_post'],
#                                          ['subclass_pre_b', 'subclass_post_c'],
#                                          ['subclass_pre', 'subclass_post']]],
#                                         ],
#                                         columns = ['name','structure'])
#     elif do_model==2:
#         H_Models = pd.DataFrame([       ['HLM',
#                                         [['class_pre', 'class_post'],
#                                          ['subclass_pre_b', 'subclass_post_c'],
#                                          ['subclass_pre', 'subclass_post']]],
#                                         ],
#                                         columns = ['name','structure']) 
#     if do_model==3:
#         H_Models = pd.DataFrame([       ['GP',['subclass_pre_b', 'subclass_post_c']],
#                                         ],
#                                         columns = ['name','structure'])   

#     #mod_index = modf.index #[3] # modf.index #[0, 3, 9, 11, 13, 15]
#     sts = pd.DataFrame(X3.iloc[0:X3.shape[0]:Dn2,:].iloc[:,0:nannot],columns = annot_columns_train)
#     sts = sts['cell_type2_pre'].map(str)+'_'+  sts['layer_pre'].map(str)+' -> '+  sts['cell_type2_post'].map(str)+'_'+sts['layer_post'].map(str)
#     #sts.to_excel('temp.xlsx')
    Y_pred = []
    Y_pred0 = []
    Samples_test = []
    for i,mdn in enumerate(H_Models.loc[:,'name']):

        # HM_name = 'Ms1c2'
        md = H_Models.loc[H_Models.loc[:,'name']==mdn, :]


        #md = modf.loc[i,:]
        #mdn = md['name']
        print(mdn)


        if type(md['structure'].values[0][0])==str:
            cli = cla_n2.index[cla_n2.isin(md['structure'].values[0])].values
            ncli = 1
        else:
            cli=[]
            cli2=md['structure'].values[0]
            ncli = len(cli2)
            for iii in range(len(cli2)):
                cli = cli + [cla_n2.index[cla_n2.isin(cli2[iii])].values]

        #mdcl2 = np.char.strip(np.array(str.split(md['classes_post'],',')))
        #mdcl1 = np.char.strip(np.array(str.split(md['classes'],','))) 

     # 300ll   
        X3_cl = np.copy(X2_cl)
        if ncli==1:
            cli5 = [cli]
        else:
            cli5 = cli

        print(set(X2_cl[:,cli5[0][0]]))
        if len(cli)>1:
            print(set(X2_cl[:,cli5[0][1]]))
        else:
            print('all')

        for iii in range(ncli): 
            cli6 = cli5[iii]
            for icl, cli2 in enumerate(cli6):
                if icl==0:
                    n2n=pd.DataFrame(list(set(X2_cl[:,cli2]))).reset_index().set_index(0)
                    X3_cl[:,cli2] = n2n.loc[X2_cl[:,cli2]].values.ravel()
                    cli20 = cli2
                else: 
                    # convert class-names to numbers for cli2
                    n2n=pd.DataFrame(list(set(X2_cl[:,cli2]))).reset_index().set_index(0)
                    X4_cl = n2n.loc[X2_cl[:,cli2]].values.ravel()

                    # combine cli2 and cli20 classes
                    X4_cl = X4_cl + n2n.shape[0]*X3_cl[:,cli20]
                    n2n = pd.DataFrame(list(set(X4_cl))).reset_index().set_index(0)
                    X3_cl[:,cli2] = n2n.loc[X4_cl].values.ravel()
                    cli20=cli2
        #### cli??        
        y_pred = np.zeros((0,y2.shape[1]))
        y_pred0 = np.zeros((y2.shape[0],y2.shape[1]))


        ncv = 10
        #ncv = 10
        n_samp_cv = np.rint(X2.shape[0]/ncv)
        samples_all = np.arange(X2.shape[0])
        r2cv = np.zeros(ncv+1)
        r3cv = np.zeros(ncv+1)
        r4cv = np.zeros(ncv+1)
        nonzs = np.array([])
        Lj=[]
        for icv in range(ncv+1): #range(ncv):  # cross-validation cycle
            if icv<ncv-1:
                samples_test = (np.arange(n_samp_cv) + icv*n_samp_cv).astype(int)
                samples_train = np.delete(np.copy(samples_all),samples_test)
            else:  
                n_samp_cv2 = int(n_samp_cv/2)
                samples_test = (np.arange(n_samp_cv2 ) + (ncv-1)*n_samp_cv+(icv-ncv+1)*n_samp_cv2  ).astype(int)
                samples_train = np.delete(np.copy(samples_all),samples_test)

            samples_test = samples_test[samples_test<X2.shape[0]]    
            X2train, y2train, X2train_cl = X2[samples_train,:], y2[samples_train,:], X3_cl[samples_train,:]
            X2test, y2test, X2test_cl = X2[samples_test,:], y2[samples_test,:], X3_cl[samples_test,:]


            if (mdn!='BHLM')&(mdn!='HLM')&(mdn!='GP'):

                #model, support = fit_classes_tree(X2train,y2train,X2train_cl,cli,model_type='HuberRegressor',
                #                                  nmin = 0,n_iter=100, alpha=1)
                #model, support = fit_classes_tree(X2train,y2train,X2train_cl,cli,model_type='ARDRegression',
                #                                  nmin = 0,n_iter=300, alpha_1=1e-06, alpha_2=1e-06,
                #                                  lambda_1=1e-06, lambda_2=1e-06)
                model, support = fit_classes_tree(X2train,y2train,X2train_cl,cli,model_type='elastic_net',
                                                   nmin = 0, alpha=0.5, l1_ratio=0.01)
                #model, support = fit_classes_tree(X2train,y2train,X2train_cl,cli,model_type='ridge',nmin = 0, alpha=1)

                # alpha_1  = 1e-6
                # alpha_2  = 1e-6
                # lambda_1 = 1e-4
                # lambda_2 = 1e-4
                # threshold_lambda=10000.0,
                # model, support = fit_classes_tree(X2train,y2train,X2train_cl,cli,model_type='ARDRegression',
                #                                    nmin = 0, alpha_1=alpha_1, alpha_2=alpha_2,
                #                                    lambda_1=lambda_1, lambda_2=lambda_2)


                y_pred_i = predict_classes_tree(model,X2test,X2test_cl,cli,nout=y2.shape[1],nmin = 0)
                for iy in range(y2.shape[1]):
                    nonz = y_pred_i[:,iy]!=0
                    y_pred_i[nonz==False,iy] = np.mean(y2[:,iy])
                y_pred = np.concatenate([y_pred, y_pred_i],axis=0)

                y_pred_i0 = predict_classes_tree(model,X2train,X2train_cl,cli,nout=y2.shape[1],nmin = 0)
                y_pred0[samples_train,:] = np.copy(y_pred_i0)   

                print('iteration ',icv)

            elif (mdn=='GP'):

                from sklearn.gaussian_process import GaussianProcessRegressor
                from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF

                kernel = 0.594**2*RBF(length_scale=0.279,length_scale_bounds=[0.1,10.0]) #DotProduct() #+ WhiteKernel()
                gpr = GaussianProcessRegressor( kernel=kernel, random_state=0).fit(X2train, y2train)
                #gpr = GaussianProcessRegressor( random_state=0).fit(X2train, y2train)
                print(gpr.score(X2train, y2train))

                y_pred_i0,_ = gpr.predict(X2train, return_std=True)
                y_pred0[samples_train,:] = np.copy(y_pred_i0)
                # sklearn.gaussian_process.GaussianProcessRegressor(kernel=None, *,
                #                                                   alpha=1e-10, optimizer='fmin_l_bfgs_b',
                #                                                   n_restarts_optimizer=0, normalize_y=False,
                #                                                   copy_X_train=True, random_state=None)
                y_pred_i,_ = gpr.predict(X2test, return_std=True)
                for iy in range(y2.shape[1]):
                    nonz = y_pred_i[:,iy]!=0
                    y_pred_i[nonz==False,iy] = np.mean(y2[:,iy])
                y_pred = np.concatenate([y_pred, y_pred_i],axis=0)
                print('iteration ',icv)    

            elif (mdn=='BHLM'):
                print('iteration ',icv)    

          # 500ll      
            if mdn=='HLM':
                if icv==0:
                    class HierarchicalLinearRegressionModel(nn.Module):
                        def __init__(self, input_dim, output_dim, c_dim, levels_dim, c_dim0, par):
                            super(HierarchicalLinearRegressionModel, self).__init__() 
                            self.out_dim = output_dim
                            self.levels_dim = levels_dim
                            self.c_dim = c_dim
                            #self.w   = 1e-7*torch.randn(c_dim,levels_dim,  input_dim, output_dim, requires_grad=True,device=cuda0)
                            #self.bw  = 1e-7*torch.randn(c_dim,levels_dim,  1,         output_dim, requires_grad=True,device=cuda0)

                            if par['regress_w']:
                                co_dim = par['c_out_dim']
                                self.c_out_dim = co_dim
                                self.c_in_out = par['c_in_out']
                                self.wo   = 1e-7*torch.randn(co_dim,1 ,  input_dim, output_dim, requires_grad=False,device=cuda0)
                                self.bwo  = 1e-7*torch.randn(co_dim,1 ,  1,   output_dim, requires_grad=False,device=cuda0)
                                self.w    = 1e-7*torch.randn(c_dim,levels_dim,  input_dim, output_dim, requires_grad=False,device=cuda0)
                                self.bw   = 1e-7*torch.randn(c_dim,levels_dim,  1,  output_dim, requires_grad=False,device=cuda0)
                                #self.wo  = nn.Parameter(self.wo)
                                #self.bwo = nn.Parameter(self.bwo)  
                            else:
                                self.w   = 1e-7*torch.randn(c_dim,levels_dim,  input_dim, output_dim, requires_grad=True,device=cuda0)
                                self.bw  = 1e-7*torch.randn(c_dim,levels_dim,  1,         output_dim, requires_grad=True,device=cuda0)
                                self.w  = nn.Parameter(self.w)
                                self.bw = nn.Parameter(self.bw) 

                            #self.u = torch.ones(c_dim0,1,1, requires_grad=True)  # c_dim0
                            self.u   = torch.ones(c_dim,levels_dim,1,1, requires_grad=True,device=cuda0) 
                            self.yc0 = torch.zeros(0,output_dim,device=cuda0)
                            #self.u20 = torch.zeros(levels_dim,1,1)


                            self.u  = nn.Parameter(self.u) 
                            #self.icl = torch.ones(c_dim,levels_dim)
                            #self.icl = [i_cl[i][0] for i in range(self.c_dim)]


                        def forward(self, x, x_dim, par): #, x_cl):
                            if par['regress_w'] and not par['is_reg_w']:
                                c_in_out = self.c_in_out
                                y=par['yo']
                                for o in range(self.c_out_dim):
                                    level_o = c_in_out[o][0]
                                    c_in_o  = c_in_out[o][1]
                                    if len(c_in_o)>1:
                                        x_c_o = [x[c] for c in c_in_o]
                                        y_c_o = [y[c] for c in c_in_o]
                                        x_c_o = torch.cat(x_c_o) # copy?!
                                        y_c_o = torch.cat(y_c_o) # copy?!
                                    else:
                                        x_c_o = x[c_in_o[0]]
                                        y_c_o = y[c_in_o[0]]



                            # Here the forward pass is simply a linear function
                            #yl = [torch.addmm(self.bw[i], x[i], self.w[i],
                            #      beta=self.u[i].item(),
                            #      alpha=self.u[i].item()) if x[i].shape[0]>0 else self.yl0 for i in range(self.c_dim)]
                            #out = torch.sum(torch.cat(yl, 0).reshape(-1, self.levels_dim, self.out_dim),1)

                            #yl = [torch.addmm(self.bw[i], x[i],
                            #            self.w[i]) if x[i].shape[0]>0 else self.yl0 for i in range(self.c_dim)]
                            #out = torch.zeros(x_dim, self.out_dim)
                            ##u2=(self.u/torch.sum(self.u,1)).reshape(-1,1) 
                            #u2 = torch.zeros(x_dim, 1)
                            #for i in range(self.c_dim):
                            #    uui=self.u[i].pow(2)
                            #    out[i_cl[i]] = yl[i]*uui
                            #    u2[i_cl[i]] = uui

                            #out=out.reshape(-1, self.levels_dim, self.out_dim) 
                            #u2=u2.reshape(-1, self.levels_dim)
                            #out=torch.sum(out,1) #/torch.sum(u2,1).reshape(-1,1)
                            #out = torch.sum(torch.cat(yl, 0).reshape(-1, self.levels_dim, self.out_dim),1)

                            #u2 = self.u/torch.sum(self.u,1).reshape(-1,1,1,1)
                            #w2 = torch.sum(u2*self.w,1)
                            #bw2 = torch.sum(u2*self.bw,1)
                            w2 = torch.sum(self.u*self.w,1)
                            bw2 = torch.sum(self.u*self.bw,1)
                            yc = [torch.matmul( x[i], w2[i])+bw2[i] \
                                  if x[i].shape[0]>0 else self.yc0 for i in range(self.c_dim)]
                            #yl = [torch.addmm(self.bw[i], x[i],
    #                                 w2[i]) if x[i].shape[0]>0 else self.yl0 for i in range(self.c_dim)]
                            return torch.cat(yc, 0) 


    #                         def __init__(self, input_dim, output_dim, c_dim, levels_dim, c_dim0):
    #                         super(HierarchicalLinearRegressionModel, self).__init__() 
    #                         self.out_dim = output_dim
    #                         self.levels_dim = levels_dim
    #                         self.c_dim=c_dim
    #                         self.w   = 1e-7*torch.randn(c_dim,levels_dim,  input_dim, output_dim, requires_grad=True)
    #                         self.bw  = 1e-7*torch.randn(c_dim,levels_dim, 1, output_dim, requires_grad=True)
    #                         #self.u = torch.ones(c_dim0,1,1, requires_grad=True)  
    #                         self.u = torch.ones(c_dim,levels_dim,1,1, requires_grad=True) 
    #                         self.yc0 = torch.zeros(0,output_dim)
    #                         self.u20 = torch.zeros(levels_dim,1,1)
    #                         self.w  = nn.Parameter(self.w)
    #                         self.bw = nn.Parameter(self.bw)
    #                         self.u = nn.Parameter(self.u) 
    #                         #self.icl = torch.ones(c_dim,levels_dim)
    #                         #self.icl = [i_cl[i][0] for i in range(self.c_dim)]


    #                     def forward(self, x, x_dim): #, x_dim): #, x_cl):
    #                         #u2 = [self.u[i].pow(2) for i in range(self.c_dim)]
    #                         #u2 = [self.u[i_cl[i]]  if i_cl[i].shape[0]>0 else self.u20 for i in range(self.c_dim)]
    #                         #yc = [((torch.matmul( x[i], self.w[i])+self.bw[i])*u2[i]).sum(0)/u2[i].sum() \
    #                         #      if x[i].shape[0]>0 else self.yc0 for i in range(self.c_dim)]
    #                         u2 = self.u/torch.sum(self.u,1)
    #                         w2 = torch.sum(u2*self.w,1)
    #                         bw2 = torch.sum(u2*self.bw,1)
    #                         yc = [torch.matmul( x[i], w2[i])+bw2[i] \
    #                               if x[i].shape[0]>0 else self.yc0 for i in range(self.c_dim)]
    #                         #yl = [torch.addmm(self.bw[i], x[i],
    # #                                 w2[i]) if x[i].shape[0]>0 else self.yl0 for i in range(self.c_dim)]
    #                         return torch.cat(yc, 0) 

                    def mse(t1, t2):
                        diff = t1 - t2
                        return torch.sum(diff * diff) / diff.numel()

                    #X4_cl_df = pd.DataFrame(X3_cl) #X3_cl_df.copy()
                    #X4_cl_df.loc[:,cli[1][1]] = X4_cl_df.loc[:,cli[1][1]]+X4_cl_df.loc[:,cli[0][1]].max()+1
                    #X4_cl_df.loc[:,cli[2][1]] = X4_cl_df.loc[:,cli[2][1]]+X4_cl_df.loc[:,cli[1][1]].max()+1
                    #X4_cl_df = X4_cl_df.loc[:,[cli[0][1],cli[1][1],cli[2][1]]]
                    #X4_cl = X4_cl_df.values

                    X4_cl_df = pd.DataFrame(X3_cl) #X3_cl_df.copy()
                    #X4_cl_df.loc[:,cli[1][1]] = X4_cl_df.loc[:,cli[1][1]]+X4_cl_df.loc[:,cli[2][1]].max()+1
                    X4_cl_df.loc[:,cli[0][1]] = X4_cl_df.loc[:,cli[0][1]]+X4_cl_df.loc[:,cli[1][1]].max()+1
                    X4_cl_df = X4_cl_df.loc[:,[cli[0][1],cli[1][1]]] #,cli[2][1]]]
                    X4_cl = X4_cl_df.values

                    #X4_cl = X4_cl # 1 synapse type model 
                    #X4_cl = X4_cl[:,[0,1,2]]
                    #X4_cl = X4_cl[:,[0,1]]
                    X4_cl = X4_cl[:,[0,1]]

                    c_dim2 = len(set(X4_cl[:,-1])) # number of smallest clusters
                    c_dim1 = X4_cl.shape[1]
                    c_dim = X4_cl.max()+1
                    output_dim = y2train.shape[1]
                    input_dim = X2train.shape[1]
    # 400ll                

                    X4train_cl, X4test_cl = X4_cl[samples_train,:], X4_cl[samples_test,:]

                    ## Regression model
                    #linear_reg_model = HierarchicalLinearRegressionModel(input_dim, output_dim, c_dim, c_dim1).cuda()

                    ## Define loss and optimize
                    #loss_fn = nn.MSELoss() #(reduction='sum')
                    ##loss_fn = mse 

                    #optim = torch.optim.Adam(linear_reg_model.parameters(), lr=0.01) #LBFGS(linear_reg_model.parameters())
                    #Adam(linear_reg_model.parameters(), lr=0.01) #, weight_decay = 1.0)
                    #optim = torch.optim.SGD(linear_reg_model.parameters(), lr = 0.0025, weight_decay = 0.1) #, weight_decay = 0.2) #Stochastic Gradient Descent
                    # optim = torch.optim.Adam(linear_reg_model.parameters(), lr = 0.005, weight_decay = 0.05) #, weight_decay = 0.2) #Stochastic Gradient Descent


                    #clusters2 = X3_cl_df.loc[:,[cli[1][1],cli[0][1]]]

                    #x_cl = X4train_cl.reshape((-1,))
                    #x_cl = Variable(torch.from_numpy(x_cl.astype('long')).to(cuda0))

                    #X4train = np.tile(X2train, (c_dim1,1))
                    #X4train=X4train.reshape((-1,1,X4train.shape[1]))
                    #x_data = Variable(torch.from_numpy(X4train.astype('float32')).to(cuda0))
                    #y_data = Variable(torch.from_numpy(y2train.astype('float32')).to(cuda0))
                    def train():
                        # initialize gradients to zero
                        optim.zero_grad()

                        # run the model forward on the data
                        #y_pred = linear_reg_model.forward(x_data, x_cl) #.squeeze(-1)
                        #y_pred, loss_reg = linear_reg_model.forward(x_data)
                        y_pred = linear_reg_model.forward(x_data, x_dim)

                        # calculate the mse loss
                        loss = loss_fn(y_pred, y_data)
    # 500ll                    
                        ## regularization loss
                        #l2_reg = torch.tensor(0., device=cuda0)
                        ##for param in linear_reg_model.parameters():
                        ##    l2_reg += torch.norm(param)
                        #u = linear_reg_model.u.reshape(-1, self.levels_dim, self.out_dim) 
                        ##out = (self.u[x_cl] * yl).reshape(-1, self.levels_dim, self.out_dim) 
                        ##out = torch.sum(out,1)
                        #nu = torch.sum(u*u,1)-1 
                        #l2_reg += lambu * (torch.sum(nu*nu)+ torch.sum(u*u*(u<0)))
                        #l2_reg += lambw * torch.norm(linear_reg_model.w)
                        #loss += loss_reg #lambw * torch.norm(linear_reg_model.w)
                        #lambw = [0.2, 0.2]
                        #l2_reg = torch.tensor(0., device=cuda0)
                        #for l in len(lambw):
                        #     l2_reg = l2_reg + (linear_reg_model.w[c[l]]).pow(2).sum()
                        #loss = loss+  l2_reg #lambw[l]*(linear_reg_model.w).pow(2).sum()


                        # backpropagate
                        loss.backward()

                        # take a gradient step
                        optim.step()

                        # for nm,p in nn.Module.named_parameters(linear_reg_model): #linear_reg_model.parameters():
                        #     #print(nm)
                        #     #print(p.size())
                        #     if nm=='u':
                        #         p.data.clamp_(0)

                        return loss

                ########## fit model

                X4train_cl, X4test_cl = X4_cl[samples_train,:], X4_cl[samples_test,:]
                #x_cl = X4train_cl.reshape((-1,))
                ##x_cl = Variable(torch.from_numpy(x_cl.astype('long')).to(cuda0))

                #X4train = np.tile(X2train, (c_dim1,1))
                ##X4train=X4train.reshape((-1,1,X4train.shape[1]))
                #X4train=X4train.reshape((-1,X4train.shape[1]))
                #x_data = [X4train[x_cl==i].astype('float32') for i in range(c_dim)]
                #x_data = [Variable(torch.from_numpy(x_data[i].astype('float32')).to(cuda0)) for i in range(c_dim)]
                #y_data = Variable(torch.from_numpy(y2train.astype('float32')).to(cuda0)) 

                x_cl   = X4train_cl
                x_dim  = X2train.shape[0]
                x_data = [X2train[x_cl[:,-1]==i] for i in range(c_dim2)]
                x_data = [Variable(torch.from_numpy(x_data[i].astype('float32')).to(cuda0)) for i in range(c_dim2)]

                # reorder y_data!!! : x_data were reordered at 547
                i_cl = np.arange(x_dim)
                i_cl = [i_cl[x_cl[:,-1]==i] for i in range(c_dim2)]
                i_cl = np.concatenate(i_cl)
                # y_data = y2train[i_cl,:]
                y_datac = np.copy(y2train)
                y_datac = [y_datac[x_cl[:,-1]==i,:] for i in range(c_dim2)]
                y_datac = np.concatenate(y_datac)
                y_data = Variable(torch.from_numpy(y_datac.astype('float32')).to(cuda0))

                # temporary variable : 
                #y_data00c = Variable(torch.from_numpy(y_datac.astype('float32')).to(cuda0)) 

                lj=[]
                print(icv)
                lambw = 0.0005 # weights regularization
                # Regression model
                #linear_reg_model = HierarchicalLinearRegressionModel(input_dim, output_dim, c_dim, c_dim1,lambw).cuda()
                linear_reg_model = HierarchicalLinearRegressionModel(input_dim, output_dim, c_dim2, c_dim1, c_dim).cuda()

                # Define loss and optimize
                loss_fn = nn.MSELoss() #(reduction='sum') #(reduction='sum')
                #loss_fn = mse
                #optim = torch.optim.Adam(linear_reg_model.parameters(), lr = 0.001, weight_decay = 0.5) #, weight_decay = 0.2) #Stochastic Gradient Descent
                #optim = torch.optim.SGD(linear_reg_model.parameters(), lr = 0.0015, weight_decay = 0.2) #, weight_decay = 0.2) #Stochastic Gradient Descent
                #optim = torch.optim.AdamW(linear_reg_model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)   

                #lambw = torch.tensor(0.2)
                #lambu = torch.tensor(1)

                lj=[]
                print(icv)
                num_iterations1 = 4000 #if not smoke_test else 2
                num_iterations2 = 4000
                num_iterations3 = 4000
                for j in range(num_iterations1+num_iterations2+num_iterations3):
                    if j==num_iterations2+num_iterations1:
                        #optim = optim3
                        optim = torch.optim.SGD(linear_reg_model.parameters(),
                                         lr=0.001, momentum=0.9, dampening=0, weight_decay=0.2, nesterov=True)
                    elif j==num_iterations1:
                        #optim = optim2
                        #optim = torch.optim.SGD(linear_reg_model.parameters(),
                        #                         lr=0.00005, momentum=0.98, dampening=0, weight_decay=0, nesterov=True)
                        optim = Adam(linear_reg_model.parameters(), lr=0.001,  weight_decay = 0.2)
                    elif j==0:
                        #optim = optim1
                        optim = Adam(linear_reg_model.parameters(), lr=0.005,  weight_decay = 0.2) # , weight_decay = 0.2)

                    loss = train()
                    lj = lj+[loss.item()]
                    if (j + 1) % 500 == 0:
                        #print("[iteration %04d] loss: %.4f" % (j + 1, loss.item()))
                        print("[iteration %04d] " % (j + 1))
                        print("loss:  ",loss.item())

                Lj = Lj + [lj]
                #### predict
                #y_pred_i0 = linear_reg_model.forward(x_data,x_cl).data.cpu().numpy()
                #y_pred_i0 = linear_reg_model.forward(x_data)[0].data.cpu().numpy()
                y_pred_i0 = np.copy(linear_reg_model.forward(x_data,x_dim).data.cpu().numpy())
                # rearrange y
                y_pred_i0[i_cl,:]=np.copy(y_pred_i0)
    # 600ll            
                #x_cl_test = Variable(torch.from_numpy(X4test_cl.astype('long')))
                #x_data_test = Variable(torch.from_numpy(X2test.astype('float32')))
                #y_data_test = Variable(torch.from_numpy(y2test.astype('float32')))
                #x_cl_test = X4test_cl.reshape((-1,))
                #x_cl_test = Variable(torch.from_numpy(x_cl_test.astype('long')).to(cuda0))
                #X4test = np.tile(X2test, (c_dim1,1))
                #X4test = X4test.reshape((-1,1,X4test.shape[1]))
                #X4test = X4test.reshape((-1,X4test.shape[1]))
                #x_data_test = Variable(torch.from_numpy(X4test.astype('float32')).to(cuda0)) 
                #x_data_test = [X4test[x_cl_test==i].astype('float32') for i in range(c_dim)]
                #x_data_test = [Variable(torch.from_numpy(x_data_test[i].astype('float32')).to(cuda0)) for i in range(c_dim)]         
                #y_pred_i = linear_reg_model.forward(x_data_test,x_cl_test).data.cpu().numpy()
                #y_pred_i = linear_reg_model.forward(x_data_test)[0].data.cpu().numpy()

                x_cl_test = X4test_cl
                ##x_dim_test=X4test.shape[0]                
                ##c_dim2 = len(set(x_cl_test[:,-1]))
                #i_cl_test = [x_cl_test[x_cl_test[:,-1]==i,:][0,:] for i in range(c_dim2)] 
                #i_cl = i_cl_test
                #i_cl = [torch.from_numpy(i_cl[i].astype('long')) for i in range(c_dim2)]

                # reorder y_data!!! : x_data were reordered at 547
                i_cl_test = np.arange(X2test.shape[0])
                i_cl_test = [i_cl_test[x_cl_test[:,-1]==i] for i in range(c_dim2)]
                i_cl_test = np.concatenate(i_cl_test)
                # y_data = y2train[i_cl,:]

                x_data_test =[X2test[x_cl_test[:,-1]==i] for i in range(c_dim2)]
                x_data_test = [Variable(torch.from_numpy(x_data_test[i].astype('float32')).to(cuda0)) for i in range(c_dim2)]
                #y_data_test = Variable(torch.from_numpy(y2test.astype('float32')))                                         

                y_pred_i = np.copy(linear_reg_model.forward(x_data_test,x_dim).data.cpu().numpy())

                # reorder y_data!!! : x_data were reordered at 634
                y_pred_i[i_cl_test,:] = np.copy(y_pred_i)

                #y_pred_i = linear_reg_model.forward(x_data_test).data.cpu().numpy()

                y_pred = np.concatenate([y_pred, y_pred_i],axis=0)
                y_pred0[samples_train,:] = np.copy(y_pred_i0)

                do_plot_each_cv=1
                iy=0
                if do_plot_each_cv==1:
                    f, ax =plt.subplots(figsize=(15, 6))
                    ##f, ax = plt.figure()
                    ##ax = f.add_axes()

                    #plt.title(stp_columns[iy]+", model : "+mdn+", cv : "+str(icv))
                    #plt.title(stp_columns_train[iy]+", model : "+mdn)
                    plt.title(stp_n[iy]+", model : "+mdn)

                    #ax.set_title(stp_columns[i]+' '+', subclass out of bag: '+str(i0))
                    #yy1.loc[:,[stpn_test[i],stpn_pred[i]]].plot(ax=ax)
                    plt.plot(y2[:,iy].ravel(),'ob')
                    plt.plot(y_pred[:,iy].ravel(),'xr')
                    plt.plot(y_pred0[:,iy].ravel(),'xg')
                    plt.pause(0.05)

            y2l = np.copy(y2)
            y_pred_il = np.copy(y_pred_i)
            y_pred_i0l = np.copy(y_pred_i0)
            y2testl = np.copy(y2test)
            y2trainl = np.copy(y2train)
            if do_log_y==1:  #recover stp data from logarithmic scale
                if do_normalize==1:
                    #scale_y2 = np.std(y2[:,:],axis=0)
                    y2l = (y2l+mean_y2)*scale_y2
                    y_pred_il = (y_pred_il+mean_y2)*scale_y2
                    y_pred_i0l = (y_pred_i0l+mean_y2)*scale_y2
                    y2testl = (y2testl+mean_y2)*scale_y2
                    y2trainl = (y2trainl+mean_y2)*scale_y2

                #d_log=0.3
                y2l = np.exp(y2l.astype(float))-d_log
                y2testl = np.exp(y2testl.astype(float))-d_log
                y2trainl = np.exp(y2trainl.astype(float))-d_log
                y_pred_il = np.exp(y_pred_il.astype(float))-d_log
                y_pred_i0l = np.exp(y_pred_i0l.astype(float))-d_log

            iy=0 #iy0
            if np.var(y2[:,iy])!=0:
                nonz=[]
                nonzs = np.append(nonzs, nonz)
                #if np.sum(nonz)>0:
                # !!!!!!!!!!!!!!!!
                #!!!!  var(y2) should be with nonz !!!
                y22testl = y2testl[:y_pred_i.shape[0],iy]
                r2cv[icv]=1 - np.mean((y22testl[:] - y_pred_il[:,iy])**2)/np.var(y2l[:,iy])
                #r2cv[icv]=1 - np.mean((y2test[nonz,iy] - y_pred_i[nonz,iy])**2)/np.var(y2[:,iy])
                #r3cv[icv]=1 - np.median(np.abs(y2test[nonz,iy] - y_pred_i[nonz,iy])**2)/np.var(y2[:,iy])
                r3cv[icv]=1 - np.median(np.abs(y2testl[:,iy] - y_pred_il[:,iy])**2)/np.var(y2l[:,iy])
                r4cv[icv]=1 - np.mean((y2trainl[:,iy] - y_pred_i0l[:,iy])**2)/np.var(y2l[:,iy])#  np.mean((y2test[nonz,iy] - y_pred_i[nonz,iy])**2)/np.var(y2test[nonz,iy])

        Y_pred = Y_pred + [y_pred]
        Y_pred0 = Y_pred0 + [y_pred0]
        Samples_test = Samples_test +[samples_test]

        if do_log_y==1:  #recover stp data from logarithmic scale
            y2l = np.copy(y2)
            y_predl = np.copy(y_pred)
            y_pred0l = np.copy(y_pred0)
            if do_normalize==1:
                #scale_y2 = np.std(y2[:,:],axis=0)
                y2l = y2l*scale_y2
                y_predl = y_predl*scale_y2
                y_pred0l = y_pred0l*scale_y2

            #d_log=0.3
            y2 = np.exp(y2l.astype(float))-d_log
            #y2test = np.exp(y2test.astype(float))-d_log
            #y2train = np.exp(y2train.astype(float))-d_log
            y_pred = np.exp(y_predl.astype(float))-d_log
            y_pred0 = np.exp(y_pred0l.astype(float))-d_log

        for iy in range(y2.shape[1]):
            f, ax =plt.subplots(figsize=(16, 6))
            ##f, ax = plt.figure()
            ##ax = f.add_axes()

            #plt.title(stp_columns[iy]+", model : "+mdn+", cv : "+str(icv))
            #plt.title(stp_columns_train[iy]+", model : "+mdn)
            plt.title(stp_n[iy]+", model : "+mdn)

            #ax.set_title(stp_columns[i]+' '+', subclass out of bag: '+str(i0))
            #yy1.loc[:,[stpn_test[i],stpn_pred[i]]].plot(ax=ax)
            plt.plot(y2[:,iy].ravel(),'ob')
            plt.plot(y_pred[:,iy].ravel(),'xr')
            plt.plot(y_pred0[:,iy].ravel(),'xg')
            #plt.xticks(np.arange(len(yy1.index)), yy1.index, rotation=90)

        iy=0 #iy0

        R3 = []

        for iy in range(y2.shape[1]):
            r2=0
            r3=0
            r4=0
            if np.var(y2[:,iy])!=0:
                y22 = y2[:y_pred.shape[0],iy]
                r2=1 - np.mean((y22[:] - y_pred[:,iy])**2)/np.var(y22[:])
                #r2=1 - np.mean((y22[:] - np.mean(y22[:]))**2)/np.var(y22[:])
                #r2=1 - np.mean((y22[nonzs==1] - y_pred[nonzs==1,iy])**2)/np.var(y22[nonzs==1])
                #r2=1 - np.mean((y22[nonzs==1] - y_pred[nonzs==1,iy])**2)/np.var(y22[nonzs==1])
                #r3=1 - np.median(np.abs(y2[:y_pred.shape[0],iy] - y_pred[:,iy]))/np.median(np.abs(y2[:,iy] - np.median(y2[:,iy])))

                r3=1 - np.median((y22[:] - y_pred[:,iy])**2)/np.median((y22[:]-np.mean(y22[:]))**2)
                r4 = 1 - np.mean((y2[:,iy] - y_pred0[:,iy])**2)/np.var(y2[:,iy]) #np.mean((y22[nonzs==1] - y_pred[nonzs==1,iy])**2)/np.var(y22[nonzs==1])
                #r4=1 - np.var(y_pred[:,iy])/np.var(y2[:,iy])
            R3=R3 + [r3]
            print('\n\n ',stp_n[iy],'\n') 
            print(" R**2_mean = ",r2,
                  "\n R**2_median = ",r3,
                  "\n R**2_mean_nonzero = ",r4,"\n",
                  "  R**2 cv mean 2 = ", np.mean(r2cv),"\n",
                  "  R**2 cv median 2 = ", np.median(r2cv),"\n",
                  "  R**2 cv mean_median = ", np.mean(r3cv),
                  "\n  R**2 cv unnormed mean = ", np.mean(r4cv),
                  "\n  R**2 cv = ", r2cv,
                  "\n  R**2 cv unnormed = ", r4cv,
                  "\n\n")

    # 700ll    
    t2 =time.time()    
    print("Elapsed time ",t2-t1)
    print(nonzs.shape)
    print(nonzs.sum())
    print(R3)
    #print(mean(R3))
    
    return Y_pred, Y_pred0, Samples_test,  R3

def train_and_test_classification_models(X2, y2, X2_cl, X2_an, H_Models, y2_reg=None, preprocessing_data=None, stp_n=[], sts=[], cuda0=None, verbosity = 1, columns={}, cv_dict={'kind':'n_fold', 'n_iterations':10 }    ):
    '''
    Train and test classification model for genes->STP
    X2        : input data, samples-x-gene expression features
    y2        : output data, samples-x-stp features
    X2_cl     : input data low level classification, samples-x-levels of classes 
    X2_an     : input data annotations, for selection of features and visualization of output
    
    y2_reg    : output data for regression phase of some classification models
    probab_y2 : ouptut data probabilities of stp classes
    
    preprocessing_data : structure described how X2 and y2 data were preprocessed
    ncv       : number of cross-validations
    stp_n     : STP features names for training
    sts       : 
    cla_n2    : names of clusters columns in X2_cl-array of gene expression clustering data 
    cuda0     : name of cuda device if gpu is used
    verbosity : if =1 - print model performance and other info 
    
    H_Models  : structure describing model type
              : columns:
              :     name       : ['HLM', 'BHLM', other: sklearn]
              :     model_kind : ['classification', other]
              : rows: md stucture with fields:
                            : regularization : cluster_structure : describes which X_cl fields to use for a particular model 
                            : parameters : 
                                         : nmin
                                         : md_function : sklearn model name
                                         : stp_parameters_to_stp_classes : function 
                                         : probab_y2
                                         : y2_syntp
                                         
    columns={'x_data':[],'y_data':[], 'clusters':[], 'annotations':[]}   
    
    '''
    stp_n= columns['y_data']
    cla_n2= columns['clusters'] # names of clusters columns in X2_cl array of data clustering information
    annot_n = columns['annotations']
    ge_n=columns['x_data']
    
    #if len(cla_n2)==0:
    #    print('add columns[clusters] to columns dictionary instead of cla_n2 argument')
    
    t1 =time.time() 
    
    do_sklearn =  [
    "Random Forest",
    "Gaussian Process",
    "Nearest Neighbors",
    "Linear SVM",
    "RBF SVM",
    "Decision Tree",
    "Neural Net",
    "AdaBoost",
    "Naive Bayes",
    "QDA",
    
    'Elastic Net',
    'ARD Regression'
    ]
    
    #do_log_y = preprocessing_['do_log_y']
    #do_normalize = preprocessing_['do_normalize']
    #if do_log_y==1:
    #    d_log = preprocessing_['d_log']
        
    #if do_normalize==1:    
    #    scale_y2 = preprocessing_['scale_y2']
    #    mean_y2 = preprocessing_['mean_y2']
    
    
    model_kind=H_Models.loc[:,'model_kind'].iloc[0] #'classification'
    
    model_kind_2='none'
    if 'model_kind_2' in H_Models.columns.tolist():
        model_kind_2=H_Models.loc[:,'model_kind_2'].iloc[0] #'classification'
    
    Y_pred_M = []
    Y_pred0_M = []
    Samples_test_M = []
    Model_performance = []
    OUT = []
    
    for i,mdn in enumerate(H_Models.loc[:,'name']):

        # HM_name = 'Ms1c2'
        md = H_Models.loc[H_Models.loc[:,'name']==mdn, :]


        #md = modf.loc[i,:]
        #mdn = md['name']
        print(mdn)


        if 'regularization' in md.keys():
            md_reg = md['regularization'].values[0]
        else:
            md_reg = {}
        
#       cluster_structure = [['class_pre', 'class_post'],
#                            ['subclass_pre_b', 'subclass_post_c'],
#                            ['subclass_pre', 'subclass_post']]
        if type(md_reg['cluster_structure'][0])==str:
            cli = cla_n2.index[cla_n2.isin(md_reg['cluster_structure'])].values
            ncli = 1 # number of levels of synaptic pairs clustering tree
            cli5 = [cli]
        else:
            cli=[]
            cli2=md_reg['cluster_structure'] # list of all levels of synaptic pair types clusters
            ncli = len(cli2) # number of levels of synaptic pairs clustering tree
            for iii in range(len(cli2)):
                cli = cli + [cla_n2.index[cla_n2.isin(cli2[iii])].values] # find all indexes of clusters in each synapse pair type sublist of cluster structure
            cli5 = cli
      
        #mdcl2 = np.char.strip(np.array(str.split(md['classes_post'],',')))
        #mdcl1 = np.char.strip(np.array(str.split(md['classes'],','))) 

     # 300ll   
        
        #if ncli==1:
        #    cli5 = [cli]
        #else:
        #    cli5 = cli

        X3_cl = np.copy(X2_cl)
        #print(set(X2_cl[:,cli5[0][0]]))
        #if len(cli)>1:
        #    print(set(X2_cl[:,cli5[0][1]]))
        #else:
        #    print('all')

        # combine all clusters in sublists - for example pre and postsynaptic clusters  - assign unique number to each individual combined cluster    
        # combine all clusters in sublists of structure list - for example pre and postsynaptic clusters  - assign unique number to each individual combined cluster  
        X3_cl = np.copy(X2_cl)
        X5_cl = np.zeros([X3_cl.shape[0],ncli]).astype(int)
        for iii in range(ncli): 
            cli6 = cli5[iii]

            # combine all clusters in the iii-th columns list
            for icl, cli2 in enumerate(cli6):
                n2n=pd.DataFrame(list(set(X2_cl[:,cli2]))).reset_index().set_index(0) # frame with indexes = cluster names from X2_cl column and values = unique number for  each cluster name
                if icl==0:
                    #print('n2n',n2n)
                    X3_cl[:,cli2] = n2n.loc[X2_cl[:,cli2]].values.ravel() # assign indeces of clusters instead of their names to the first columns of final tree structure 
                    #print('X3_cl_cli2',X3_cl[:,cli2])
                    cli20 = cli2
                else: 
                    # convert class-names to numbers for cli2
                    X4_cl = n2n.loc[X2_cl[:,cli2]].values.ravel()  # assign indeces of clusters instead of their names to a column of final tree structure 

                    # combine cli2 and cli20 classes
                    X4_cl = X4_cl + n2n.shape[0]*X3_cl[:,cli20]   # modify clusters indexes to make them unique for all combinations of clusters from 0 to icl column of cli6 list
                    n2n = pd.DataFrame(list(set(X4_cl))).reset_index().set_index(0) # make a dictionary of all unique combining clusters indexes and their unique numbers from 0 to icl
                    X3_cl[:,cli2] = n2n.loc[X4_cl].values.ravel() # translate clusters indexes to obtain unique and dense indexing of combined clusters
                    cli20=cli2
            X5_cl[:,iii]=X3_cl[:,cli2]       


        # make the column with input clusters the last 
        if 'input_clusters_column' in md_reg.keys():
            input_clusters_column = md_reg['input_clusters_column']
        else:
            input_clusters_column = ncli-1 # columns with input clusters is last by default
        idc = np.arange(ncli)
        idc = np.delete(idc,input_clusters_column)
        idc = np.insert(idc,idc.size,input_clusters_column)
        X3_cl = X5_cl[:,idc]


        # make cluster indexes unique between columns, lowest clusters indexes corresponds to inputs
        X4_cl = X3_cl.copy()
        for iii in range(1,ncli):
            iii2 = ncli-iii-1
            X4_cl[:,iii2]=X3_cl[:,iii2] + X4_cl[:,iii2+1].max()+1
        X3_cl =  X4_cl  
                
        #### cli??        
        #y_pred = np.zeros((y2.shape[0],y2.shape[1]))
        #y_pred0 = np.zeros((y2.shape[0],y2.shape[1]))
                
        Y_pred = []
        Y_pred0 = []
        Samples_test = []
        #Model_performance = []
        
        # cv_dict - dictionary with cross validation arangement information : 
        #           - times and size of cv sampling datasets, 
        #           - random or sequential? 
        #           - level of spt hierarhy to sample?
        # cv_dict = {'kind':{'n_fold', 'n_random_spt_types', 'n_sequential_spt_types'}, 'n_iterations':10,
        #            'part_cv_test_sample':0.1, 'cv_levels':['classes_spt', 'subclasses_spt', 'markered_clusters_spt'] }        
        #cv_dict = parameters['cross validation parameters']
        if cv_dict['kind']=='n_fold':
            ncv = cv_dict['n_iterations']
            n_samp_cv = np.rint(X2.shape[0]/ncv)
        else: # cv_dict['kind']=='n_random_spt':  
            ncv = cv_dict['n_iterations']
            if 'part_cv_test_sample' in cv_dict.keys():
                n_samp_test_cv = np.rint(X2.shape[0]*cv_dict['part_cv_test_sample'])
            else:    
                n_samp_test_cv = np.rint(X2.shape[0]/ncv)
            cv_levels = cv_dict['cv_levels'] #!!!!!!!!!!!!!!!
            #cli = cla_n2.index[cla_n2.isin(md_reg['cluster_structure'])].values
            cv_levels2 = cla_n2.index[cla_n2.isin(cv_levels)].values # indexes of cv_levels columns in X2_cl
            X2_cl_cv = X2_cl[:,cv_levels2[0]].astype(str) # all clusters in cv_levels columns of X2_cl combined
            for jl in range(1,cv_levels2.size):
                X2_cl_cv = np.char.add(X2_cl_cv, np.repeat(' -> ',X2_cl_cv.size))
                X2_cl_cv = np.char.add(X2_cl_cv, X2_cl[:,cv_levels2[jl]].astype(str))
            i_l=-1
            all_sptl=list(set(X2_cl_cv)) # all cluster types in all combined cv_levels columns of X2_cl
            n_sptl=len(all_sptl)    
            print('cross validation : all spt on cv_level : ',all_sptl)
            print('cross validation : n spt on cv_level : ',n_sptl)

        samples_all = np.arange(X2.shape[0])
        nonzs = np.array([])
        Lj=[]

        # cross-validation cycle
        for icv in range(ncv+1): #range(ncv):
            print('cross validation : ',icv)

            # make cross validational training and testing samples
            if cv_dict['kind']=='n_fold': # case n_fold : test on sequentially selected 1/n part of data
                if icv<ncv-1:
                    samples_test = (np.arange(n_samp_cv) + icv*n_samp_cv).astype(int)
                    samples_train = np.delete(np.copy(samples_all),samples_test)
                else:  
                    n_samp_cv2 = int(n_samp_cv/2) # ACHTUNG!!!! ITS NOT TRUE K-FOLD CV
                    samples_test = (np.arange(n_samp_cv2 ) + (ncv-1)*n_samp_cv+(icv-ncv+1)*n_samp_cv2  ).astype(int)
                    samples_train = np.delete(np.copy(samples_all),samples_test) 
            else: # case n_spt_levels random or sequentially : test on sequentially or randomly selected 1/n part of data from the same synaptic pair class, subclass or markered cluster

                is_samples_test = np.zeros([X2.shape[0],1])==1
                while is_samples_test.sum() < n_samp_test_cv:
                    if cv_dict['kind']=='n_random_spt_types':
                        i_l = np.random.randint(0, high=n_sptl-1)
                    else: # cv_kind=='n_sequential_spt_types'
                        i_l = i_l+1
                        if i_l>=n_sptl:
                            i_l= np.random.randint(0, high=n_sptl-1)# restart from random spt type

                    add_spt = X2_cl_cv==all_sptl[i_l]
                    #print('i_l - ', i_l,' , n_spt_i_l - ',add_spt.sum())
                    #if add_spt.sum()+is_samples_test.sum() > n_samp_test_cv:
                    #    n_add = n_samp_test_cv - is_samples_test.sum() 
                    #    add_spt[samples_all[add_spt][int(n_add):]] = False
                    is_samples_test[add_spt] = True


                samples_test =  samples_all[is_samples_test.ravel()].astype(int) 
                samples_train =  samples_all[is_samples_test.ravel()==False].astype(int) 


                print('cross validation : samples_test.size : ',samples_test.size)
                
                if 'Dn2' in preprocessing_data.keys():
                    Dn2_ = preprocessing_data['Dn2']
                    print('cross validation : samples_test : ',samples_test[0::Dn2_])
                    print('cross validation : samples_test : ',samples_test[0::Dn2_]/Dn2_)

                #samples_all_test = np.arange(samples_test.shape[0])
                #samples_2_test = np.random.permutation(samples_all_test )[0:n_samp_test_cv].sort()
                #samples_test = samples_test[samples_2_test]

                print('cross validation : i_l last : ',i_l)
                       
                        
 
                
            X2train, y2train, X2train_cl = X2[samples_train,:], y2[samples_train,:], X3_cl[samples_train,:]
            X2test, y2test, X2test_cl = X2[samples_test,:], y2[samples_test,:], X3_cl[samples_test,:]


            if (mdn!='BHLM')&(mdn!='HLM')&(mdn not in do_sklearn):
                
                #print()

                #alpha_1  = 1e-6
                #alpha_2  = 1e-6
                #lambda_1 = 1e-6
                #lambda_2 = 1e-6
                #threshold_lambda=10000.0,
                
                if (model_kind=='classification'):
                    y2train_reg = y2_reg[samples_train,:]
                    y2test_reg = y2_reg[samples_test,:]
                else:
                    y2train_reg = y2train
                    y2test_reg = y2test
                    
                
                parameters = md['parameters'].values[0] 


                #model, support = fit_classes_tree(X2train,y2train,X2train_cl,cli,model_type='HuberRegressor',
                #                                  nmin = 0,n_iter=100, alpha=1)
                #model, support = fit_classes_tree(X2train,y2train,X2train_cl,cli,model_type='ARDRegression',
                #                                  nmin = 0,n_iter=300, alpha_1=1e-06, alpha_2=1e-06,
                #                                  lambda_1=1e-06, lambda_2=1e-06)
                #model, support = fit_classes_tree(X2train,y2train,X2train_cl,cli,model_type='elastic_net',
                #                                   nmin = 0, alpha=0.5, l1_ratio=0.01)
                
                model, support = fit_classes_tree(X2train,y2train_reg,X2train_cl,cli,**parameters)                                   
                                                  
                #model, support = fit_classes_tree(X2train,y2train,X2train_cl,cli,model_type='ridge',nmin = 0, alpha=1)
                # model, support = fit_classes_tree(X2train,y2train,X2train_cl,cli,model_type='ARDRegression',
                #                                    nmin = 0, alpha_1=alpha_1, alpha_2=alpha_2,
                #                                    lambda_1=lambda_1, lambda_2=lambda_2)

                nmin = parameters['nmin']
                y_pred_i = predict_classes_tree(model,X2test,X2test_cl,cli,nout=y2train_reg.shape[1],nmin = nmin)
                # for iy in range(y2.shape[1]):   # n_out = y2.shape[1]
                #     nonz = y_pred_i[:,iy]!=0
                #     y_pred_i[nonz==False,iy] = np.mean(y2[:,iy])
                # y_pred = np.concatenate([y_pred, y_pred_i],axis=0)
                y_pred_i0 = predict_classes_tree(model,X2train,X2train_cl,cli,nout=y2train_reg.shape[1],nmin = nmin)
                
                

            #elif (mdn=='BHLM'):
           
            if mdn in do_sklearn:
                    #nrf=4
                    #Dn3_train = int(X_train.shape[0]/Dn2)
                    #Dn5 = Dn3_train*nrf
                    #regr = RandomForestClassifier(random_state=2026,max_depth=5,min_samples_leaf=5,
                    #               n_estimators=200, oob_score=True, n_jobs=-1, max_samples=None)

                    #regr = sklearn.svm.LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=1.0, multi_class='ovr',
                    #                            fit_intercept=True, intercept_scaling=1,
                    #                            class_weight=None, verbose=0, random_state=None, max_iter=1000)
                    #regr = sk.neighbors.KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', 
                    #                leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=-1)

                    parameters = md['parameters'].values[0]
                    md_parameters = parameters['par_md_function']
                    md_obj  = parameters["md_function"]
                    #md_obj = md_function(**md_parameters) 

                    if (model_kind=='classification')&(model_kind_2=='classify_from_regression'):
                        y2train_reg = y2_reg[samples_train,:]
                        y2test_reg = y2_reg[samples_test,:]
                    else: # model_kind=='regression'
                        y2train_reg = y2train
                        y2test_reg = y2test                
                      
                    
                    classify_from_regression = False
                    if model_kind_2 == 'classify_from_regression':
                        classify_from_regression = True
                                    
                    #regr.fit(X2train, y2train)
                    getattr(md_obj, "fit")(X2train, y2train_reg)


                    ##y_pred_i = regr.predict(X2test).reshape((-1,1))
                    if classify_from_regression:
                        y_pred_i = getattr(md_obj, "predict")(X2test)
                        y_pred_i0 =  getattr(md_obj, "predict")(X2train)
                    else:    
                        y_pred_i = getattr(md_obj, "predict")(X2test).reshape((-1,1))
                        y_pred_i0 =  getattr(md_obj, "predict")(X2train).reshape((-1,1))   

                  
                    # STP classes from STP amplitudes
                    if (model_kind=='classification'):
                        

                        
                        
                        if icv==0:
                            y2_reg_out = inverse_processed_data(y2_reg, preprocessing_data)
                            y2_out = parameters['y2_syntp_sub'] #np.copy(y2)
                            y_pred = y2_out #np.zeros((y2.shape[0],y2.shape[1]))
                            y_pred0_ = y2_out


                            
                        # convert regression data to original form (rescale and denormilize)
                        if classify_from_regression:
                            y_pred_dnmli_inv = inverse_processed_data(y_pred_i, preprocessing_data)
                            y_pred_dnmli0_inv = inverse_processed_data(y_pred_i0, preprocessing_data)
                        else:    
                            y_pred_dnmli_inv = y_pred_i.reshape([-1,1])
                            y_pred_dnmli0_inv = y_pred_i0.reshape([-1,1])

                        print('y_pred_dnmli_inv',y_pred_dnmli_inv.shape)
                        #print('y_pred_dnmli0_inv',y_pred_dnmli0_inv.shape)

                        # classify regression data
                        if classify_from_regression:
                            parameters['classify_from_regression'] = classify_from_regression
                            parameters['inds'] = samples_test
                            y_pred_i_out, probab_y_pred_i_out, samples_probab_pred_i = parameters['stp_parameters_to_stp_classes'](y_pred_dnmli_inv,**parameters)
                            parameters['inds'] = samples_train
                            y_pred_i0_out, probab_y_pred_i0_out, samples_probab_pred_i0 = parameters['stp_parameters_to_stp_classes'](y_pred_dnmli0_inv,**parameters) 
                            
                            # compute model performance measures
                            probab = {'y2':parameters['probab_y2'], 'y2test':probab_y_pred_i_out,'y2train':probab_y_pred_i0_out}
                            probab2={'probab_y2':parameters['probab_y2'], 'probab_y2test':probab_y_pred_i_out,'probab_y2train':probab_y_pred_i0_out}
                            samples_probab =[parameters['samples_probab_y2'], samples_probab_pred_i, samples_probab_pred_i0]
                            samples_probab2={'samples_y2':parameters['samples_probab_y2'],
                                             'samples_y2test':samples_probab_pred_i,
                                             'samples_y2train':samples_probab_pred_i0}
                            
                            
                                                    #model_perf  = {}
                            Dn2=parameters['Dn2']
                            model_perf  = model_performance(y2_out, y_pred_i_out, y_pred_i0_out, 
                                                       samples_test=samples_test, samples_train=samples_train, samples_probab=samples_probab,
                                                       Dn2=Dn2, model_kind=model_kind, probab=probab, 
                                                       y2_reg_out=y2_reg_out, y_pred_reg_out=y_pred_dnmli_inv, y_pred0_reg_out=y_pred_dnmli0_inv,
                                                       classify_from_regression=classify_from_regression)
                        else:
                            y_pred_i_out = y_pred_i
                            y_pred_i0_out = y_pred_i0
                            
                            probab = {}
                            samples_probab =[]

                            #model_perf  = {}
                            Dn2=parameters['Dn2']
                            model_perf  = model_performance(y2_out, y_pred_i_out, y_pred_i0_out, 
                                                           samples_test=samples_test, samples_train=samples_train, samples_probab=samples_probab,
                                                           Dn2=Dn2, model_kind=model_kind, probab=probab, 
                                                           classify_from_regression=classify_from_regression)

                        
                        #print('y_pred_i_out',y_pred_i_out.shape)
                        #print('y_pred_i0_out',y_pred_i0_out.shape)



            if mdn=='HLM':
                if icv==0:
                    #X4_cl_df = pd.DataFrame(X3_cl) #X3_cl_df.copy()
                    ##X4_cl_df.loc[:,cli[1][1]] = X4_cl_df.loc[:,cli[1][1]]+X4_cl_df.loc[:,cli[2][1]].max()+1
                    #X4_cl_df.loc[:,cli[0][1]] = X4_cl_df.loc[:,cli[0][1]]+X4_cl_df.loc[:,cli[1][1]].max()+1     # THIS MUST BE MODIFIED!!!!
                    #X4_cl_df = X4_cl_df.loc[:,[cli[0][1],cli[1][1]]] #,cli[2][1]]]
                    #X4_cl = X4_cl_df.values
                    X4_cl = X3_cl

                    #X4_cl = X4_cl # 1 synapse type model 
                    #X4_cl = X4_cl[:,[0,1,2]]
                    #X4_cl = X4_cl[:,[0,1]]
                    #X4_cl = X4_cl[:,[0,1]] # ACHTUNG!!! - contradiction?: for training X4_cl should be - rows of cluster tree , for testing - [zo, zi]

                    #nc = np.max(X4_cl[:,[0,1]])+1
                    nc = X4_cl.max()+1 # whole number of clusters in tree[:,:]
                    print('number_of_clusters ',nc)
                    X4_cl = np.concatenate([nc*np.ones([X4_cl.shape[0],1]), X4_cl ], axis=1) # add root cluster

                    
                    scl4 = list(set(X4_cl[:,-1])) 
                    # tree of clusters
                    #Oi = [np.array([0,3,4]), np.array([1,3,4]), np.array([2,4])] # parents of each leaf cluster
                    nI = len(scl4)
                    Oi = [] # Oi - list: for each input clusters - indices of all parent clusters
                    for iclu in range(nI):
                        is_iclu = X4_cl[:,-1]==scl4[iclu]
                        x4_iclu = X4_cl[is_iclu,:]
                        Oi = Oi + [np.flip(x4_iclu[0,:]).astype('int')]

                    #M = {'W': W, 'Oi': Oi, 'R': R, 'RI': RI, 'Ki': Ki, 'Sig': Sig, 'Tree':Tr} #, 'n_samples': n_samples}
                    M = {'Oi':Oi}  
                    #M['Oi'] = Oi

                    #c_dim2 = len(set(X4_cl[:,-1])) # number of smallest clusters
                    #c_dim1 = X4_cl.shape[1]
                    #c_dim = X4_cl.max()+1

                    output_dim = y2train.shape[1]
                    input_dim = X2train.shape[1]
    # 400ll                

                    X4train_cl, X4test_cl = X4_cl[samples_train,:], X4_cl[samples_test,:]



    #             ########## fit model
                ##################################################################
                X4train_cl, X4test_cl = X4_cl[samples_train,:], X4_cl[samples_test,:]
                
                if (model_kind=='classification'):
                    y2train_reg = y2_reg[samples_train,:]
                    y2test_reg = y2_reg[samples_test,:]
                else:
                    y2train_reg = y2train
                    y2test_reg = y2test

                # add intercept
                do_intercept = 1
                if do_intercept==1:
                    X2train = np.concatenate([X2train, np.ones([X2train.shape[0],1])], axis=1)
                    X2test = np.concatenate([X2test, np.ones([X2test.shape[0],1])], axis=1)
                    #ng = X2train.shape[1]
                    input_dim = X2train.shape[1]

                # convert data to pytorch variables                                                                             
                # y_pred_i = np.copy(linear_reg_model.forward(x_data_test,x_dim).data.cpu().numpy())
                tr_y2train = Variable(torch.from_numpy(y2train_reg.astype('float32')).to(cuda0))
                tr_X2train   = Variable(torch.from_numpy(X2train.astype('float32')).to(cuda0))
                tr_X2train_cl = Variable(torch.from_numpy(X4train_cl.astype('long')).to(cuda0)) 

                tr_X2test    = Variable(torch.from_numpy(X2test.astype('float32')).to(cuda0))
                tr_X2test_cl = Variable(torch.from_numpy(X4test_cl.astype('long')).to(cuda0)) 

                # convert parameters to pytorch variables
                parameters=md['parameters'].values[0]
                sig = parameters['sig'] #0.4 # 0.4 - up to median R2 = 65%, nit=15; R2=70% nit=5  for scvi_lf? #
                ng = X2train.shape[1]
                sigmap  = Variable(torch.from_numpy(np.array(sig).astype('float32')).to(cuda0))
                #sigmap  = Variable(torch.from_numpy((np.ones((ng,1))*sig).astype('float32')).to(cuda0))

    # 357ll
                # set parameters for weights regression regularization
                s2=2*sigmap**2
                alpha_sigma  = parameters['alpha_sigma_factor']*sigmap # 0.01*sigmap
                beta_sigma   = parameters['beta_sigma_factor']*sigmap # 2.0*sigmap
                
                lyambda0 = parameters['lyambda0'] #50 #50 #0.5 -?bad?? # 50 - up to median R2 = 65%, nit=15; R2=70 nit=5 for scvi_lf? #
                alpha_lambda = parameters['alpha_lambda_factor']*lyambda0*s2 #0.0000001*lyambda0*s2
                beta_lambda  = parameters['beta_lambda_factor']*1.0*lyambda0*s2 #1.0*lyambda0*s2
                lambdas = Variable(torch.from_numpy((lyambda0*2*sig**2*np.ones((ng,1))).astype('float32')).to(cuda0)) #0.5*torch.tensor(np.ones(ng,1))

                nit=5         # number of iterations uMDL
                it_stop = parameters['it_stop'] #0.000001 # relative dlyambda for stop uMDL
                n_jumps = parameters['n_jumps'] #0 #10 #
                dn_jumps = parameters['dn_jumps'] #0 #2. # if dn_jumps==0 - just fit all n_o clusters, otherwise - merge sequentially , make random jumps, refit
                do_simplified_tree = parameters['do_simplified_tree'] # use only input leaf nodes for output clusters
                
                #par = {'nit': nit, 'relTol': it_stop} #, 'index_zi': zi, 'Ci': Ci}
                par = {'nit': nit, 'relTol': it_stop, 'alpha_sigma': alpha_sigma, 'beta_sigma': beta_sigma,
                      'alpha_lambda': alpha_lambda,  'beta_lambda': beta_lambda, 'n_jumps': n_jumps, 'dn_jumps': dn_jumps,
                       'device':cuda0, 'do_uMDL':True, 'do_simplified_tree':do_simplified_tree } # CHECK!!! - 'do_uMDL':True to do uMDL instead of elastic_net
                
                par = dict(par, **parameters)
                
                do_dnml = True
                if do_dnml==True:
                    #tr_y2train,tr_X2train,[Zo_jump],Lambdas, Sigmap, par
                    Out,par = fit_DNML(tr_y2train,tr_X2train,tr_X2train_cl,M,lambdas, sigmap, par)

                    nio_best = Out['nio_optimal']
                    zo_opt = Out['zo_optimal']
                    #Wp2 = Out['Wp2']
                    Wp2 = Out['Wp2']
                else:    
                    cli = tr_X2train_cl.shape[1]-1 # ACHTUNG!!! - assumed input clusters in last columns
                    Out = fit_UMDL(tr_y2train,tr_X2train,tr_X2train_cl,lambdas, sigmap, par, cli=cli)
                    Wp2 = Out['Wp']                   
                    
                    


                # DNML Regression: test  
                #y_pred_dnmli0 = np.copy(( tr_X2train @ wp).data.cpu().numpy())   
                #y_pred_dnmli = np.copy(( tr_X2test @ wp).data.cpu().numpy()) 
                n_out = y2train_reg.shape[1] # output dimensionality
                if do_dnml==True:
                    tr_X_out_train_cl = torch.clone(tr_X2train_cl) # ACHTUNG!!! - contradiction?: for training X4_cl should be - rows of cluster tree , for testing - tr_X_out_train_c : [zo,..., zi]
                    tr_X_out_train_cl[:,0] = zo_opt   # ACHTUNG!!! - tr_X2train_cl.shape[1] should be >1 (tr_X2train_cl[:,1] = zi)
                    #y_pred_dnmli0 = np.copy(predictDNML( tr_X2train, n_out,Wp2, X_cl=tr_X_out_train_cl,par=par).data.cpu().numpy()) 
                    if (dn_jumps>0)&(do_simplified_tree==False):
                        y_pred_dnmli0 = np.copy(predictDNML( tr_X2train , n_out,Wp2, Sigmap=Out['Sigmap'], Wp_x=Out['Wp_x'], Sigmap_x=Out['Sigmap_x'], p_z=Out['p_z'],
                                                            X_cl=tr_X_out_train_cl,par=par).data.cpu().numpy())
                    else:
                        y_pred_dnmli0 = np.copy(predictDNML( tr_X2train , n_out,Wp2, X_cl=tr_X_out_train_cl,par=par).data.cpu().numpy()) 
                else: 
                    y_pred_dnmli0 = np.copy(predictUMDL( tr_X2train, n_out,Wp2, X_cl=tr_X_out_train_cl,par=par).data.cpu().numpy())      

                tr_X_out_test_cl = torch.clone(tr_X2test_cl) 
                zi_test = tr_X2test_cl[:,-1]
                zo_opt_test = torch.zeros_like(tr_X2test_cl[:,0]) # zo_opt_test
                for i2 in range(nI):  
                    #zo_opt_test[zi_test==i2]=nio_best[i2] # best clusters zo
                    zo_opt_test[zi_test==i2]=i2  # only input clusters zo!!! should be modified to use DNML-found best output clusters (nio_best[i] for each input cluster i)
                tr_X_out_test_cl[:,0] = zo_opt_test 
                #print('tr_X2test',tr_X2test.shape)
                #print('tr_X2test_cl',tr_X_out_train_cl.shape)
                #y_pred_dnmli = np.copy(predictDNML( tr_X2test,  n_out, Wp2, OxI = Out['OxI'],X_cl=tr_X_out_test_cl,par=par).data.cpu().numpy())
                if (dn_jumps>0)&(do_simplified_tree==False):
                    y_pred_dnmli = np.copy(predictDNML( tr_X2test, n_out,Wp2, Sigmap=Out['Sigmap'], Wp_x=Out['Wp_x'], Sigmap_x=Out['Sigmap_x'], p_z=Out['p_z'],
                                                       X_cl=tr_X_out_test_cl, OxI = Out['OxI'],par=par).data.cpu().numpy())
                else:    
                    y_pred_dnmli = np.copy(predictDNML( tr_X2test, n_out,Wp2, X_cl=tr_X_out_test_cl, OxI = Out['OxI'],par=par).data.cpu().numpy())
    
          

                #ACHTUNG!!! - temporary use mean(y2) for nontrained input clusters 
                do_this_7 = False
                if do_this_7:
                    for iy in range(n_out): #(y2.shape[1]):   # n_out = y2.shape[1]
                        nonz = y_pred_dnmli[:,iy]!=0
                        y_pred_dnmli[nonz==False,iy] = np.mean(y2_reg[:,iy])
                #y_pred = np.concatenate([y_pred, y_pred_i],axis=0)


                #print('nio_best',nio_best)

                # correct for x-shift
                #y_pred_umdli =y_pred_umdli  + (bx_test - bx)@w2
                #y_pred_i0 =y_pred_i0  + (bx - bx)*w2

                #y_pred_dnml = y_pred_dnml + [y_pred_dnmli]
                #y_pred_dnml0 = y_pred_dnml0 + [y_pred_dnmli0]

                #SSEy_dnml_tra = SSEy_dnml_tra +[1 - np.mean((y_pred_dnmli0-y2train)**2)/np.var(y2train)]
                #SSEy_dnml_gen = SSEy_dnml_gen +[1 - np.mean((y_pred_dnmli-y2test)**2)/np.var(y2test)]
                
                # STP classes from STP amplitudes
                if (model_kind=='classification'):
                    if icv==0:
                        y2_reg_out = inverse_processed_data(y2_reg, preprocessing_data)
                        y2_out = parameters['y2_syntp_sub'] #np.copy(y2)
                        y_pred = y2_out #np.zeros((y2.shape[0],y2.shape[1]))
                        y_pred0_ = y2_out
                    
                    # convert regression data to original form (rescale and denormilize)
                    y_pred_dnmli_inv = inverse_processed_data(y_pred_dnmli, preprocessing_data)
                    y_pred_dnmli0_inv = inverse_processed_data(y_pred_dnmli0, preprocessing_data)
                    
                    #print('y_pred_dnmli_inv',y_pred_dnmli_inv.shape)
                    #print('y_pred_dnmli0_inv',y_pred_dnmli0_inv.shape)
                    
                    # classify regression data
                    parameters['inds'] = samples_test
                    #print('parameters',parameters)
                    y_pred_i_out, probab_y_pred_i_out, samples_probab_pred_i = parameters['stp_parameters_to_stp_classes'](y_pred_dnmli_inv,**parameters)
                    
                    #print('probab_y_pred_i_out',probab_y_pred_i_out.shape)
                    
                    parameters['inds'] = samples_train
                    y_pred_i0_out, probab_y_pred_i0_out, samples_probab_pred_i0 = parameters['stp_parameters_to_stp_classes'](y_pred_dnmli0_inv,**parameters) 
                    
                    #print('y_pred_i_out',y_pred_i_out.shape)
                    #print('y_pred_i0_out',y_pred_i0_out.shape)
                    
                    # compute model performance measures
                    probab = {'y2':parameters['probab_y2'], 'y2test':probab_y_pred_i_out,'y2train':probab_y_pred_i0_out}
                    probab2={'probab_y2':parameters['probab_y2'], 'probab_y2test':probab_y_pred_i_out,'probab_y2train':probab_y_pred_i0_out}
                    samples_probab =[parameters['samples_probab_y2'], samples_probab_pred_i, samples_probab_pred_i0]
                    samples_probab2={'samples_y2':parameters['samples_probab_y2'], 'samples_y2test':samples_probab_pred_i,'samples_y2train':samples_probab_pred_i0}
                    classify_from_regression = True
                    
                    Dn2=parameters['Dn2']
                    model_perf  = model_performance(y2_out, y_pred_i_out, y_pred_i0_out, 
                                                   samples_test=samples_test, samples_train=samples_train, samples_probab=samples_probab,
                                                   Dn2=Dn2, model_kind=model_kind, probab=probab, 
                                                   y2_reg_out=y2_reg_out, y_pred_reg_out=y_pred_dnmli_inv, y_pred0_reg_out=y_pred_dnmli0_inv)
                    


    #             # reorder y_data!!! : x_data were reordered at 634
    #             y_pred_i[i_cl_test,:] = np.copy(y_pred_i)
    #             #y_pred_i = linear_reg_model.forward(x_data_test).data.cpu().numpy()
    
            # model train and test performance
            if (model_kind=='regression'):
                #y2_out = inverse_processed_data(np.copy(y2), preprocessing_data)
                y_pred_i_out = inverse_processed_data(np.copy(y_pred_i), preprocessing_data)
                y_pred_i0_out = inverse_processed_data(np.copy(y_pred_i0), preprocessing_data)
                
                if icv==0:
                    y2_out = inverse_processed_data(np.copy(y2), preprocessing_data)
                    y_pred = np.tile(np.mean(y2_out,axis=0),(y2.shape[0],1)) #np.zeros((y2.shape[0],y2.shape[1]))
                    y_pred0_ = np.copy(y_pred)
                    
                probab2={}
                samples_probab2={}
                Dn2=parameters['Dn2']        
                model_perf  = model_performance(y2_out, y_pred_i_out, y_pred_i0_out, 
                                                   samples_test=samples_test, samples_train=samples_train, samples_probab=samples_probab,
                                                   Dn2=Dn2, model_kind=model_kind)
                
                

                    
            #y_pred = np.concatenate([y_pred, y_pred_i.reshape([-1,1])],axis=0)
            #y_pred0[samples_train,:] = np.copy(y_pred_i0)
            y_pred[samples_test,:] = y_pred_i_out[:,0:y_pred.shape[1]]
            
            y_pred0 = np.copy(y_pred0_)
            y_pred0[samples_train,:]=y_pred_i0_out[:,0:y_pred0.shape[1]]
            Y_pred0 = Y_pred0 + [y_pred0]
            Samples_test = Samples_test +[samples_test]
            
            if icv==0:
                model_perf_cv=[]
            if (model_kind=='regression'): 
                model_perf_cv = model_perf_cv + [{**model_perf}]    
            if (model_kind=='classification'):    
                model_perf_cv = model_perf_cv + [{**model_perf, **probab2, **samples_probab2}]    

            do_plot_each_cv=True
            if do_plot_each_cv==True:
                if (model_kind=='regression'):
                    plot_each_cv_results(y2_out, y_pred, y_pred0_, model_kind=model_kind, **kwarg)
                if (model_kind=='classification'): 
                    kwarg = {'samples_test':samples_test, 'samples_train':samples_train,
                             'y2_reg':y2_reg_out, 'y_pred_reg':y_pred_dnmli_inv, 'y_pred0_reg':y_pred_dnmli0_inv, 
                             'model_name': mdn, 'stp_names':stp_n, 'data_annotation':X2_an, 'classify_from_regression': classify_from_regression}
                    plot_each_cv_results(y2_out, y_pred_i_out, y_pred_i0_out, model_kind=model_kind, **kwarg)

                ################################################################


        Y_pred_M = Y_pred_M + [y_pred]
        Y_pred0_M = Y_pred0_M + [Y_pred0]
        Samples_test_M = Samples_test_M +[Samples_test]
        Model_performance = Model_performance +[model_perf_cv]
        OUT = OUT + [Out]
        
        if verbosity==1:
            kwarg = {'stp_names':stp_n}
            print_model_performance(model_perf_cv, model_kind=model_kind, **kwarg)
            

    t2 =time.time()    
    display(Markdown('Elapsed time '+str(t2-t1)+'s'))
    
    
    return Y_pred_M, Y_pred0_M, Samples_test_M,  Model_performance, OUT
    

# def stp_parameters_to_stp_classes(y,**parameters):
    
#     y_labels = y
    
#     return y_labels
    
    
def inverse_processed_data(y2, processing_):
    
        if 'do_log_y' in processing_:
            do_log_y = processing_['do_log_y']
            if do_log_y==1:
                d_log = processing_['d_log']
        else:
            do_log_y = 0
            
        if 'do_normalize' in processing_:
            do_normalize = processing_['do_normalize']
            if do_normalize==1:    
                scale_y2 = processing_['scale_y2']
                mean_y2  = processing_['mean_y2']
        else:
            do_normalize  = 0              
    
        y2l = np.copy(y2)
        if do_normalize==1:
                #scale_y2 = np.std(y2[:,:],axis=0)
                y2l = (y2l+mean_y2)*scale_y2
        
        if do_log_y==1:  #recover stp data from logarithmic scale
            #y2l[y2l<=0] = 0.01 # ACHTUNG!!! UNPRECISE CUT OFF OF NEGATIVE VALUES
            y2l = np.exp(y2l.astype(float))-d_log
            
        return y2l
        
        
def model_performance(y2, y2test, y2train, samples_test=[], samples_train=[],samples_probab=[], Dn2=1, model_kind='', probab=None,
                      y2_reg_out=[], y_pred_reg_out=[], y_pred0_reg_out=[], classify_from_regression=True):
        # model_performance(y2_out, y_pred_i_out, y_pred_i0_out, samples_test=samples_test, samples_train=samples_train, model_kind=model_kind, perf=model_perf, probab=probab)
#         if perf is None:
#                 z = []
#                 perf = {'regression':{'mean_R2_test':z, 'mean_R2_train':z,  'median_R2_test':z},
#                         'classification':{'confusion_test':z, 'confusion_train':z,  'crossentropy_test':z, 'crossentropy_train':z, 'entropy_data':z }}
                
        if (model_kind=='regression'):
            vout=(y2.shape[1],1)
            r2cv=np.zeros(vout)  
            r3cv=np.zeros(vout)
            r4cv=np.zeros(vout)
            for  iy in range(y2.shape[1]): #iy0
                if (np.var(y2[:,iy])!=0):
                    r2cv[iy]=1 - np.mean((y2test[:,iy] - y2[samples_test,iy])**2)/np.var(y2[:,iy])
                    r3cv[iy]=1 - np.median(np.abs(y2test[:,iy] - y2[samples_test,iy])**2)/np.var(y2[:,iy])
                    r4cv[iy]=1 - np.mean((y2train[:,iy] - y2[samples_train,iy])**2)/np.var(y2[:,iy])
                    
            #perf['regression']['mean_R2_test'].append([r2cv])
            #perf['regression']['median_R2_test'].append([r3cv])
            #perf['regression']['mean_R2_train'].append([r4cv])
            perf = { 'mean_R2_test':r2cv,'median_R2_test':r3cv,'mean_R2_train':r4cv}
            
        if (model_kind=='classification'):
            # regression part
            if type(y2_reg_out)==list:
                vout=(1,1)
            else:    
                vout=(y2_reg_out.shape[1],1)
            r2cv=np.zeros(vout)  
            r3cv=np.zeros(vout)
            r4cv=np.zeros(vout)
            if classify_from_regression:
                for  iy in range(y2_reg_out.shape[1]): #iy0
                    if (np.var(y2_reg_out[:,iy])!=0):
                        r2cv[iy]=1 - np.mean((y_pred_reg_out[:,iy] - y2_reg_out[samples_test,iy])**2)/np.var(y2_reg_out[:,iy])
                        r3cv[iy]=1 - np.median(np.abs(y_pred_reg_out[:,iy] - y2_reg_out[samples_test,iy])**2)/np.var(y2_reg_out[:,iy])
                        #r3cv[iy]=1 - np.median((y_pred_reg_out[:,iy] - y2_reg_out[samples_test,iy])**2)/np.median((y2_reg_out[:,iy]-np.mean(y2_reg_out[:,iy]))**2)
                        
                        r4cv[iy]=1 - np.mean((y_pred0_reg_out[:,iy] - y2_reg_out[samples_train,iy])**2)/np.var(y2_reg_out[:,iy])

                #perf['regression']['mean_R2_test'].append([r2cv])
                #perf['regression']['median_R2_test'].append([r3cv])
                #perf['regression']['mean_R2_train'].append([r4cv])
            
            
            # classification part
            ii=[samples_test, samples_train]
            
            #list_val=list(set(y2))
            #conf_test = np.zeros((4,0))
            #conf_train = np.zeros((4,0))
            #conf=[]
#             for v in list_val:
#                 for i in ii:
#                     tp_v=np.sum((y2test[:]==v)&(v==y2[i]))
#                     fp_v=np.sum((y2test[:]==v)&(v!=y2[i]))
#                     fn_v=np.sum((y2test[:]!=v)&(v==y2[i]))
#                     tn_v=np.sum((y2test[:]!=v)&(v!=y2[i]))
#                     conf = conf + [np.append(conf_test, np.array([tp_v,fp_v,fn_v,tn_v]),axis=1)]
            conf=[[],[]]
                    
            ce=[]
            eps=1e-10
            # probab = { 'y2':parameters['probab_y2'], 'y2test':probab_y_pred_i_out,  'y2train':probab_y_pred_i0_out }
            list_p=list(probab.values())
            #print(probab)
            #print(list_p)
            
            for ind, i in enumerate(ii):
                #print(ind)
                #print(list_p[ind+1])
                #i2 = i.reshape([Dn])
                #print(samples_probab)
                i2 = samples_probab[ind+1] #list(set((i/Dn2).astype(int)))
                #print(len(i2))
                cei = -np.sum(probab['y2'][i2]*np.log(np.maximum(list_p[ind+1], eps)))
                #if ind==0:
                #    cei = -np.sum(probab['y2'][i,:]*np.log(np.max(probab['y2test'][i,:], eps)))
                #else:
                #    cei = -np.sum(probab['y2'][i,:]*np.log(np.max(list_p[ind+1], eps)))
                    
                ce = ce + [cei]

                
            #perf['classification']['confusion_test'].append([conf[0]])
            #perf['classification']['confusion_train'].append([conf[1]])
            #perf['classification']['crossentropy_test'].append([ce[0]])
            #perf['classification']['crossentropy_train'].append([ce[1]])
            ##perf['classification']['entropy_data'].append([ent_d])
            
            perf = {'confusion_test':conf[0], 'confusion_train':conf[1], 'crossentropy_test':ce[0], 'crossentropy_train':ce[1], 
                    'mean_R2_test':r2cv,'median_R2_test':r3cv,'mean_R2_train':r4cv,
                    'y_reg':y2_reg_out, 'y_pred_reg':y_pred_reg_out, 'y_pred0_reg':y_pred0_reg_out }
        
        #print(perf)
        return perf
            
            
def plot_each_cv_results(y2, y_pred, y_pred0, model_kind='', **kwarg):
        # kwarg = {'samples_test':samples_test, 'samples_train':samples_train, 'model_name': mdn, 'stp_names':stp_n, 'data_annotation':X2_an}
        mdn = kwarg['model_name']  
        stp_n = kwarg['stp_names']  
        X2_an = kwarg['data_annotation'] 
        if (model_kind=='classification'): 
            if 'classify_from_regression' in kwarg.keys():
                classify_from_regression = kwarg['classify_from_regression'] 
            else:
                classify_from_regression =  True
            
            samples_test = kwarg['samples_test'] 
            samples_train = kwarg['samples_train'] 
            if classify_from_regression:
                y2_reg = kwarg['y2_reg']
                y_pred_reg =kwarg['y_pred_reg']
                y_pred0_reg =kwarg['y_pred0_reg']
             
                
            if classify_from_regression==False:
                # draw classification results
                #for iy in range(y2_reg.shape[1]):
                f, ax =plt.subplots(figsize=(15, 6))
                #plt.title(stp_n[iy]+", model : "+mdn)
                stp_n_str=', '.join([istr for istr in stp_n])
                plt.title(stp_n_str+", model : "+mdn)
                if classify_from_regression:
                    for iy in range(y2_reg.shape[1]):
                        plt.plot(y2_reg[:,iy].ravel()) #,'ob')

                plt.plot(y2[:].ravel(),'ob')
                plt.plot(samples_test,y_pred[:].ravel(),'xr')
                plt.plot(samples_train,y_pred0[:].ravel(),'xg')
            
            if classify_from_regression:
                # draw regression results
                f, ax =plt.subplots(figsize=(15, 6))
                for iy in range(y2_reg.shape[1]):
                    plt.plot(y2_reg[:,iy].ravel(),'.b')
                    plt.plot(samples_test, y_pred_reg[:,iy].ravel(),'.r')
                    plt.plot(samples_train,y_pred0_reg[:,iy].ravel(),'.g')
            
            plt.pause(.000001)
            #plt.draw()
        
        if (model_kind=='regression'):    
            for iy in range(y2.shape[1]):
                X2_an2 = pd.DataFrame(X2_an)
                X2_an2 = X2_an2[0].map(str)+'_'+X2_an2[2].map(str)+'->'+X2_an2[1].map(str)+'_'+X2_an2[3].map(str)
                
                f, ax =plt.subplots(figsize=(16, 4))
                ##f, ax = plt.figure()
                ##ax = f.add_axes()
    
                #plt.title(stp_columns[iy]+", model : "+mdn+", cv : "+str(icv))
                #plt.title(stp_columns_train[iy]+", model : "+mdn)
                plt.title(stp_n[iy]+", model : "+mdn)
    
                #ax.set_title(stp_columns[i]+' '+', subclass out of bag: '+str(i0))
                #yy1.loc[:,[stpn_test[i],stpn_pred[i]]].plot(ax=ax)
    
                #plt.plot(y2[:,iy].ravel(),'ob')
                #plt.plot(y_pred[:,iy].ravel(),'xr')
                #plt.plot(y_pred0[:,iy].ravel(),'xg')
    
                plt.plot(y2[:,iy].ravel(),'b')
                plt.plot(y_pred[:,iy].ravel(),'r')
                plt.plot(y_pred0[:,iy].ravel(),'g')
    
                #plt.xticks(np.arange(len(yy1.index)), yy1.index, rotation=90)
    
               # f, ax =plt.subplots(figsize=(16, 4))
               # plt.plot(y_pred0[:,iy].ravel(),y2[:,iy].ravel(),'g')
               # plt.plot(y_pred[:,iy].ravel(),y2[:,iy].ravel(),'b')
    
                temp = Out['Temp2'][0]
                #temp = {'y':np_y, 'x':np_x, 'iszo':is_zo, 'wp':wp}
                np_y=temp['y']
                np_x=temp['x']
                l1_ratio=0.01
                alpha = 0.5
                mode_l= sk.linear_model.ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=True,
                                                normalize=False,
                                                max_iter=1000, copy_X=True, tol=0.0001, warm_start=False,
                                                random_state=None, selection='cyclic').fit(np_x,np_y)
                #wp = np.copy(mode_l.coef_ )
                f, ax =plt.subplots(figsize=(16, 4))   
                plt.plot(np_y[:].ravel(),'b')
                #plt.plot(y_pred2[:].ravel(),'r') 
                
                #i_part  = Dn2*52
                #plt.xticks(np.arange(0,X2_an2.shape[0]-i_part,Dn2), X2_an2.iloc[i_part:-1:Dn2], rotation=90)

def print_model_performance(model_performance,model_kind='', **kwarg):
        R3 = []
        stp_n = kwarg['stp_names']     
        if (model_kind=='regression'): 

            for iy in range(y2.shape[1]):
                r2=0
                r3=0
                r4=0
                if np.var(y2[:,iy])!=0:
                    y22 = y2[:y_pred.shape[0],iy]
                    r2=1 - np.mean((y22[:] - y_pred[:,iy])**2)/np.var(y22[:])
    
                    r3=1 - np.median((y22[:] - y_pred[:,iy])**2)/np.median((y22[:]-np.mean(y22[:]))**2)
                    r4 = 1 - np.mean((y2[:,iy] - y_pred0[:,iy])**2)/np.var(y2[:,iy]) #np.mean((y22[nonzs==1] - y_pred[nonzs==1,iy])**2)/np.var(y22[nonzs==1])
                R3=R3 + [r3]
                print('\n\n ',stp_n[iy],'\n') 
                print(" R**2_mean = ",r2,
                      "\n R**2_median = ",r3,
                      "\n R**2_mean_nonzero = ",r4,"\n",
                      "  R**2 cv mean 2 = ", np.mean(r2cv),"\n",
                      "  R**2 cv median 2 = ", np.median(r2cv),"\n",
                      "  R**2 cv mean_median = ", np.mean(r3cv),
                      "\n  R**2 cv unnormed mean = ", np.mean(r4cv),
                      "\n  R**2 cv = ", r2cv,
                      "\n  R**2 cv unnormed = ", r4cv,
                      "\n\n")
                      
            #print(nonzs.shape)
            #print(nonzs.sum())
            #print(R3)
            
        if (model_kind=='classification'):
            #print(model_performance)
            r2cv = []
            r3cv = []
            r4cv = []
            ncv = len(model_performance)
            
            for mi in model_performance:
                r2cv = r2cv+[mi['mean_R2_test']]
                r3cv = r3cv+[mi['median_R2_test']]
                r4cv = r4cv+[mi['mean_R2_train']]
            r2cv = np.squeeze(np.array(r2cv)).T
            r3cv = np.squeeze(np.array(r3cv)).T
            r4cv = np.squeeze(np.array(r4cv)).T
            
            #print('r2cv ',r2cv)
            
#             print(    "  R**2 cv mean 2 = ", np.mean(r2cv),"\n",
#                       "  R**2 cv median 2 = ", np.median(r2cv),"\n",
#                       "  R**2 cv mean_median = ", np.mean(r3cv),
#                       "\n  R**2 cv unnormed mean = ", np.mean(r4cv),
#                       "\n  R**2 cv = ", r2cv,
#                       "\n  R**2 cv unnormed = ", r4cv,
#                       "\n\n")
            
#             r2cv[iy]=1 - np.mean((y_pred_reg_out[:,iy] - y2_reg_out[samples_test,iy])**2)/np.var(y2_reg_out[:,iy])
#             ##r3cv[iy]=1 - np.median(np.abs(y_pred_reg_out[:,iy] - y2_reg_out[samples_test,iy])**2)/np.var(y2_reg_out[:,iy])
#             r3cv[iy]=1-np.median((y_pred_reg_out[:,iy]-y2_reg_out[samples_test,iy])**2)/np.median((y2_reg_out[:,iy]-np.mean(y2_reg_out[:,iy]))**2)
#             r4cv[iy]=1 - np.mean((y_pred0_reg_out[:,iy] - y2_reg_out[samples_train,iy])**2)/np.var(y2_reg_out[:,iy])
#             #r'$\alpha > \beta$'
#             plt.ylabel(r'$w^{2}$,%',fontsize=16)
            
            #from IPython.display import display, Latex
            #for i in range(3):
            #display(Latex(f'$x_{i}$'))
            #display(Markdown('*some markdown* $\phi$'))
            # If you particularly want to display maths, this is more direct:
            #display(Latex('\phi'))
            
            display(Markdown('**RESULTS of REGRESSION CROSS-VALIDATION: Part of STP parameters ($y_{data}$) VARIABILITY EXPLAINED by the model:**'))
            display(Markdown('TESTING:  Median of $R^{2} = 1 - \\frac{<(y_{predicted} - y_{data})^{2}>}{var(y_{data})}$ by all cross-validation iterations : '+str(np.median(r2cv))))
            display(Markdown('TESTING:  Mean of $R^{2} = 1 - \\frac{median((y_{predicted} - y_{data})^{2})}{median((y_{data}-<y_{data}>)^{2})}$ by all cross-validation iterations :  '+str(np.mean(r3cv))))
            display(Markdown('TESTING:  Mean of $R^{2} = 1 - \\frac{<(y_{predicted} - y_{data})^{2}>}{var(y_{data})}$ by all cross-validation iterations, : '+str(np.mean(r2cv))))
            display(Markdown('TRAINING: Mean of $R^{2} = 1 - \\frac{<(y_{trained} - y_{data})^{2}>}{var(y_{data})}$ by all cross-validation iterations : '+str(np.mean(r4cv))))
            display(Markdown('TESTING:  $R^{2} = 1 - \\frac{<(y_{predicted} - y_{data})^{2}>}{var(y_{data})}$ for all cross-validation iterations , % :'))
            df_r2cv = pd.DataFrame(100*r2cv, index=stp_n)
            display(df_r2cv)
            display(Markdown('TRAINING: $R^{2} = 1 - \\frac{<(y_{trained} - y_{data})^{2}>}{var(y_{data})}$ for all cross-validation iterations, % :'))
            df_r4cv = pd.DataFrame(100*r4cv, index=stp_n)
            display(df_r4cv)
            
            
def predict_STP_model(y2,par={}): #STP_model_predictor_type='precise'):
    if par['STP_model_predictor_type']=='precise':
    
        model_type = 'tm4' #'tm4x2'
        model_type2 = '' 
        nstim = 5
        DT1 = 50
        DT2 = 20
        #T_A_ = (0+np.arange(nstim)*DT1).tolist()+(0+DT1*(nstim-1)+np.array([100,350,1350])).tolist()
        #T_A_ = (T_A_[-1]+0+np.arange(nstim)*DT2).tolist()+(T_A_[-1]+0+DT2*(nstim-1)+np.array([500,2500])).tolist()
        T_A_ = (0+np.arange(nstim)*DT1).tolist()+[np.inf]
        T_A_ = (0+np.arange(nstim)*DT2).tolist()                  
        T_A_ = np.array(T_A_)
        #T_A_ =         [0, 50, 100, 150, 200, inf, 0, 20, 40, 60, 80]  
        vv    = np.array([  1,            4,           7,          10])
        sig=1
        F_D =np.zeros([y2.shape[0],2])
        print('stp bbp type estimation progress : ')
        for ii in range(y2.shape[0]):
            amps = y2[ii,:]
            x,Q2,amps2 = stpdb.fit_STP_model(amps,sig,T_A_,par,indexes_amps=vv,model_type = model_type,model_type2 = model_type2)

            #x = [tF, p0,tD,  dp]
            F_D[ii,0] = x[0]
            F_D[ii,1] = x[2]
            
            
        FbyD = F_D[:,0]/F_D[:,1]
        is_tp10 = np.concatenate([(FbyD<0.4), (FbyD>0.4)&(FbyD<2), (FbyD>2) ], axis=1)
        y_syntp = np.argmax(is_tp10, axis=1).reshape([-1,1])   
    if par['STP_model_predictor_type']=='sklearn':
        #par={'STP_classes_type':'bbp', 'STP_model_predictor_type':'precise','STP_type_predictor':m0,'bbp_columns':bbp_n}
        m=par['STP_type_predictor'][0]
        y_syntp = m.predict(y2)
        all_y=list(set(y_syntp.tolist()))
        #is_tp10 = np.zeros([y_syntp.size,len(all_y)])==1
        is_tp10 = np.zeros([y_syntp.size,3])==1
        for o in range(len(all_y)):
            is_tp10[y_syntp==all_y[o], all_y[o]]=True
            
    return is_tp10, y_syntp.reshape([-1,1])      
            
                
            
def do_probab_y(X_train0, N_bootstraps=1, Dn=1, Dn2=1, Dn3=1, stp_n=[], 
                classes_columns_train=[],synptp_boundaries=[], classify_from_regression=True, par={'STP_classes_type':'objective', 'STP_model_predictor_type':'precise'}):
    
    ####
    ####   ESTIMATE stp class from y2 predictors
    ####
    if classify_from_regression==True:
        # stp_n = ['A2_20Hz','A3_20Hz','A4_20Hz','A5_20Hz','A2_50Hz','A3_50Hz','A4_50Hz','A5_50Hz']
        if True: #stp_n==['A2_20Hz','A5_20Hz']:

            if str(type(X_train0)) == "<class 'numpy.ndarray'>" :
                y2 =  X_train0
                # use arguments Dn2, Dn3
            else:    
                y2 =  X_train0.loc[:,stp_n].iloc[0:X_train0.shape[0]:Dn,:].values
                # redefine Dn2, Dn3
                Dn2 = int(N_bootstraps/Dn) # 100 - should be a number of bootstraps per synapse type !
                Dn3 = int(X_train0.shape[0]/N_bootstraps) # number of synapse_types


            # is_synapse_stp_class_1? estimator for each bootstraped sample
            if (stp_n==['A2_20Hz','A5_20Hz'])&(par['STP_classes_type']=='objective'):
                is_tp10 =( 1.25 - 1.25/1.75*y2[:,0] -y2[:,1]< 0  ).reshape([-1,1])
                y_syntp = np.concatenate([is_tp10,is_tp10==False],axis=1)
                y_syntp = np.argmax(y_syntp, axis=1).reshape([-1,1])
                
            elif par['STP_classes_type']=='bbp':
                stp_n_a= np.array(stp_n)
                if True: #(stp_n==['A2_20Hz','A5_20Hz','A2_50Hz','A5_50Hz']):
                    #print('par2',par)
                    is_tp10, y_syntp = predict_STP_model(y2,par=par) #STP_model_predictor_type=par['STP_model_predictor_type']) #STP_model_predictor_type={ 'precise', 'nn'}
                    
                    #FbyD = F_D[ii,0]/F_D[ii,1]
                    #is_tp10 = np.concatenate([(FbyD<0.4), (FbyD>0.4)&(FbyD<2), (FbyD>2) ], axis=1)
                    #y_syntp = np.argmax(is_tp10, axis=1).reshape([-1,1])


            # estimate probability of each synapse stp class in each synaptic pair type
            if (stp_n==['A2_20Hz','A5_20Hz'])&(par['STP_classes_type']=='objective'):
                if (type(synptp_boundaries)==list)&(len(synptp_boundaries)==0):
                        Dn4 = is_tp10.shape[0]
                        Dn5 = int(Dn4/Dn2)
                        if Dn5*Dn2!=Dn4:
                            #warn('2962 genes2stp : number of samples must be equal to Dn2*Dn3')
                            #print('2962 genes2stp : number of samples must be equal to Dn2*Dn3')     
                            dl =  Dn2 + Dn5*Dn2 - Dn4
                            is_tp1 = np.concatenate([is_tp10,np.ones([dl,1])==1])
                            is_tp1 = is_tp1.reshape([-1,Dn2]).transpose()
                            #print(is_tp1.shape)
                        else: 
                            is_tp1 = is_tp10.reshape([Dn3,Dn2]).transpose()

                        #is_tp1 = is_tp10.reshape([Dn3,Dn2]).transpose()
                        probab_y = np.concatenate([np.sum(is_tp1, axis=0).reshape([-1,1]),
                                           np.sum(is_tp1==False, axis=0).reshape([-1,1]) ], axis=1)/Dn2
                        samples_probab = np.arange(Dn3)
                else:   
                        # this should be rewrited : use synptp_boundaries to define sets of samples for each syn pair type
                        probab_y = [] #np.array([]) 
                        samples_probab = list(set(synptp_boundaries))
                        for i in samples_probab:
                            isi = synptp_boundaries==i
                            #if np.sum(isi):
                            n_tp1 = np.sum(is_tp10[isi])
                            n_tpall=np.sum(isi)
                            tp_i = np.array([n_tp1, n_tpall-n_tp1]/n_tpall).reshape([1,-1]) #np.array([n_tp1, n_tpall-n_tp1]/n_tpall).reshape([1,-1])
                            probab_y = probab_y + [tp_i]

                        probab_y = np.concatenate(probab_y, axis=0)
                        #probab_y = np.concatenate([np.sum(is_tp1, axis=0).reshape([-1,1]),
                        #                   np.sum(is_tp1==False, axis=0).reshape([-1,1]) ], axis=1)/Dn2
            else:   
                if (type(synptp_boundaries)==list)&(len(synptp_boundaries)==0):
                    synptp_boundaries=(np.arange(is_tp10.shape[0])/Dn2).astype(int)
                    samples_probab = np.arange(Dn3)
                    
                # this should be rewrited : use synptp_boundaries to define sets of samples for each syn pair type
                probab_y = [] #np.array([]) 
                samples_probab = list(set(synptp_boundaries))
                for i in samples_probab:
                    isi = synptp_boundaries==i
                    n_tpall=np.sum(isi)
                    tp_i = []
                    for cl in range(is_tp10.shape[1]):
                        tp_i = tp_i + [is_tp10[isi,cl].sum()]
                    
                    tp_i = np.array(tp_i/n_tpall).reshape([1,-1])
                    probab_y = probab_y + [tp_i]

                probab_y = np.concatenate(probab_y, axis=0)    


        else:
            print('stp_n should be one of: [A2_20Hz,A5_20Hz], [A2_20Hz,A5_20Hz,A2_50Hz,A5_50Hz], etc.')
            #stp_n = ['A2_20Hz','A3_20Hz','A4_20Hz','A5_20Hz','A2_50Hz','A3_50Hz','A4_50Hz','A5_50Hz']
            
            
    else: 
        y2 =  X_train0.reshape([-1,1])
        
        
        # is_synapse_stp_class_1? estimator for each bootstraped sample
        is_tp10 =( y2[:,0] == 0  ).reshape([-1,1])
        y_syntp = np.concatenate([is_tp10,is_tp10==False],axis=1)
        y_syntp = np.argmax(y_syntp, axis=1).reshape([-1,1])


        # estimate probability of each synapse stp class in each synaptic pair type
        if (type(synptp_boundaries)==list)&(len(synptp_boundaries)==0):
            
                    Dn4 = is_tp10.shape[0]
                    Dn5 = int(Dn4/Dn2)
                    if Dn5*Dn2!=Dn4:
                            #warn('2962 genes2stp : number of samples must be equal to Dn2*Dn3')
                            #print('2962 genes2stp : number of samples must be equal to Dn2*Dn3')     
                            dl =  Dn2 + Dn5*Dn2 - Dn4
                            is_tp1 = np.concatenate([is_tp10,np.ones([dl,1])==1])
                            is_tp1 = is_tp1.reshape([-1,Dn2]).transpose()
                            #print(is_tp1.shape)
                    else: 
                            is_tp1 = is_tp10.reshape([Dn3,Dn2]).transpose()
            
            
                    #is_tp1 = is_tp10.reshape([Dn3,Dn2]).transpose()
                    probab_y = np.concatenate([np.sum(is_tp1, axis=0).reshape([-1,1]),
                                       np.sum(is_tp1==False, axis=0).reshape([-1,1]) ], axis=1)/Dn2
                    samples_probab = np.arange(Dn3)
        else:   
                    # this should be rewrited : use synptp_boundaries to define sets of samples for each syn pair type
                    probab_y = [] #np.array([]) 
                    samples_probab = list(set(synptp_boundaries))
                    for i in samples_probab:
                        isi = synptp_boundaries==i
                        #if np.sum(isi):
                        n_tp1 = np.sum(is_tp10[isi])
                        n_tpall=np.sum(isi)
                        tp_i = np.array([n_tp1, n_tpall-n_tp1]/n_tpall).reshape([1,-1]) #np.array([n_tp1, n_tpall-n_tp1]/n_tpall).reshape([1,-1])
                        probab_y = probab_y + [tp_i]

                    probab_y = np.concatenate(probab_y, axis=0)
                    #probab_y = np.concatenate([np.sum(is_tp1, axis=0).reshape([-1,1]),
                    #                   np.sum(is_tp1==False, axis=0).reshape([-1,1]) ], axis=1)/Dn2


    return probab_y, y_syntp, samples_probab

def stp_parameters_to_stp_classes(y_inv,**parameters):
    
    N_bootstraps=parameters['N_bootstraps'] 
    Dn=parameters['Dn']
    Dn2=parameters['Dn2']
    Dn3=parameters['Dn3']
    stp_n = parameters['stp_n']
    
    inds = parameters['inds']
    synptp_boundaries = (inds/Dn2).astype(int)
    
    #print('parameters', parameters)
    if 'classify_from_regression' in parameters.keys():
        classify_from_regression = parameters['classify_from_regression'] 
    else:
        classify_from_regression = True
     
    #print('parameters', parameters)
    if  'STP_classes_type' in parameters.keys():
        par={'STP_classes_type':parameters['STP_classes_type'], 'STP_model_predictor_type':parameters['STP_model_predictor_type']}  
        #print('parameters', parameters)
        if 'STP_type_predictor' in parameters.keys():
            par['STP_type_predictor'] = parameters['STP_type_predictor']
    else:
        par={'STP_classes_type':'objective', 'STP_model_predictor_type':'precise'}    
    
    #print('par',par)
    probab_y_i_out, y_i_out, samples_probab = do_probab_y(y_inv, N_bootstraps=N_bootstraps, Dn=Dn, Dn2=Dn2, Dn3=Dn3, stp_n=stp_n, 
                classes_columns_train=[], synptp_boundaries=synptp_boundaries, classify_from_regression=classify_from_regression, par=par)
    
    
    return y_i_out, probab_y_i_out, samples_probab