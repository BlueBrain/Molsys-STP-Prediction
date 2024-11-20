import pandas as pd
import numpy as np
#import scipy as sci
import scipy.optimize
import scipy.optimize as opt
import time

'''
Scripts for work with the Short Term Plasticity Database (STPDB)
'''

def Q_TM_aba_synphys(x, amps,sig,Ts,DT0,model_type='tm5'):
    '''
    '''
    #Q, A, A3 =  QA_TM_aba_synphys(x, amps,sig,Ts)
    Q, A =  QA_TM_aba_synphys(x, amps,sig,Ts,DT0,model_type)
    return Q  

def QA_TM_aba_synphys(x, amps,sig,Ts,DT0,model_type='tm5'):
    #amps=args[0] 
    #sig=args[1] 
    #Ts=args[2]
    
    
    fl=0
    if fl==0:
        ams = x[4:7]
        #breakpoint()
        if len(x)>7:
            if model_type=='tm5_fdr2':
                x2=np.copy(np.delete(x,[5,6]))
            if model_type=='tm5_smr':
                x2=np.copy(np.delete(x,[5,6]))    
            
            #x2[5:] = x[7:]
        else:
            x2=np.copy(x[0:5])
    
    n  = [0, 32, 44]
    t2 = [6, 1,  1]
    #T2  =[125, 250, 500, 1000, 2000, 4000]
    eps = np.finfo(float).eps
    z4 = np.arange(4)
    z8 = np.arange(8)
    A = np.zeros((amps.shape[0],))
    #A3 = np.ones((amps.shape[0],))
    #S3 = np.zeros((amps.shape[0],))
    Q=0
    for ii in range(3):
        vv = z8 + n[ii]
        A1e = amps[vv]  
        S1e = sig[vv]
        if fl==0:
            x2[4] = ams[ii] # independent amplitudes for each stimulation frequency - to account for rundown etc.
        
        #vv_nonz = np.nonzero(np.abs(A1e[:])>eps*1e3)[0]
        vv_z = np.abs(A1e[:])<eps*1e3
        #print(vv_nonz)
        #if len(vv_nonz)>0:
        if sum(vv_z)!=len(A1e):
            # preconditioning:
            A1, states = STP_sim(x2, Ts[vv],model_type=model_type ) 
            np0 = states[-1]
            
            # first 8 stimuli responces:
            #A1, states = STP_sim(x2, Ts[vv]+DT0, init_state=np0, model_type=model_type ) 
            #breakpoint()
            A1[vv_z] = 0
            A[vv] = A1
            
            #Q = Q + np.sum((A1e[vv_nonz]-A1[vv_nonz])**2/(S1e[vv_nonz]**2 + eps))
            Q = Q + np.sum((A1e-A1)**2/(S1e**2 + eps))
            # recovery responces
            #np0 = [n12[-1], p12[-1], d12[-1]]
            np0 = states[-1]
            for i3 in range(t2[ii]):
                vv2 = vv[-1]+1+z4+4*i3
                A2e = amps[vv2] 
                S2e = sig[vv2] 
                ##A3[vv2] = np.median(A1e)
                #vv_nonz2 = np.nonzero(np.abs(A2e[:])>eps*1e3)[0]
                #print(vv_nonz2)
                vv_z2 = np.abs(A2e[:])<eps*1e3
                #if len(vv_nonz2)>0:
                if sum(vv_z2)!=len(A2e):
                    A2, states2 = STP_sim(x2, Ts[vv2], init_state=np0, model_type=model_type )
                    
                    A2[vv_z2] = 0
                    
                    A[vv2] = A2
                    
                    #breakpoint()
                    #A2 = A2/np.median(A1)
                    #A[vv2] = A2
                    #breakpoint()
                    
                    #Q = Q + np.sum((A2e[vv_nonz2]-A2[vv_nonz2])**2/(S2e[vv_nonz2]**2 + eps))
                    Q = Q + np.sum((A2e-A2)**2/(S2e**2 + eps))
    
    return Q, A #, A3
# stpdb.fit_STP_model(amps,sig,T_A_,par,indexes_amps=vv,model_type = model_type,model_type2 = model_type2)
def Q_TM_simple(x, amps,sig,Ts,indexes_amps=None,model_type='tm4',model_type2 = ''):
    #Q, A, A3 =  QA_TM_aba_synphys(x, amps,sig,Ts)
    Q, _ =  QA_TM_simple(x, amps,sig,Ts,indexes_amps=indexes_amps,model_type=model_type,model_type2 = '')
    return Q  

def QA_TM_simple(x, amps,sig,T_A,indexes_amps=None,model_type='tm4',model_type2 = ''):
    eps = np.finfo(float).eps
    #                    #  0:tF  1:p0  2:tD   3:dp    4:A      5:tF2    6:p02   7:tD2   dd  t_FDR  t_SMR  dp0 
    #par['x0']  = np.array([100,  0.1,  100,   0.1,    0.2,     200,     0.2,    200,  0.05, 100,   100,   0.02])    
    if model_type=='tm4':
        #x = [tF, p0,tD,  dp]
        x[3] = x[1] # dp = p0
        amps2, sts2 = STP_sim2(x,T_A,model_type='tm5')
    else: 
        if model_type=='tm4x2':
            #x = [tF, p0,tD,  dp, A2, tF2, p02, tD2, dp2]
            #x[3] = x[1] # dp = p0
            #x[8] = x[6] # dp2 = p02
#             x[4] = np.abs(x[4])
#             x[9] = np.abs(x[9])
#             if x[4]+x[9]>1:
#                 cc=x[4] +x[9]
#                 x[4], x[9] = x[4]/cc, x[9]/cc
            
            #model_types_2      = ['',  'same_tF',  'same_tD',  'same_tD_tF',  'same_p0',  'same_p0_tD',  'same_p0_tF',  'same_p0_tD_tF']
            if model_type2=='same_tF':
                x[5] = x[0]
            elif model_type2=='same_tD':
                x[7] = x[2]    
            elif model_type2=='same_tD_tF':
                x[7] = x[2]             
            elif model_type2=='same_p0':
                x[6] = x[1]             
            elif model_type2=='same_p0_tD':
                x[7] = x[2] 
                x[6] = x[1] 
            elif model_type2=='same_p0_tF':
                x[5] = x[0] 
                x[6] = x[1]
            elif model_type2=='same_p0_tD_tF':
                x[7] = x[2]  # tD2=tD
                x[5] = x[0]  # tF2=tF
                x[6] = x[1]  # p02=p0
                
        amps2, sts2 = STP_sim2(x,T_A,model_type=model_type)
    
    amps2 = amps2/amps2[0]
    if indexes_amps is not None:
        amps2 = amps2[indexes_amps]
    #print('x ',x)
    #print('amps2 ',amps2)
    Q = np.sum((amps2-amps)**2/(2*sig**2 + 10*eps))
    return Q, amps2    


def fit_STP_model(amps,sig,Ts,par,indexes_amps=None,model_type = 'tm4',model_type2 = ''):

    npar = amps.shape[1]


         #  tF   p0     tD  dp         A            A1           A2      tDmin  dd   t_FDR t_SMR dp0 
    x0 = [ 100,  0.1,  100, 0.1, amps[0]/0.1, amps[0]/0.1,  amps[0]/0.1, 10,   0.05, 100,  100,  0.02 ]
    x0 = np.array(x0)
    
    #dxx = np.array([1e3, 20, 1e3, 1e2,  20, 20, 20,  1e1, 20, 1e2 ])
    dxx = np.array([1e3,  10, 1e3, 10,  20, 20, 20,  1e3, 20, 1e3, 1e3, 50 ])
    
    #breakpoint()
    if model_type=='tm5':
        x0 = x0[0:4]
        dxx = dxx[0:4]

        
    if model_type=='tm4':
        x0 = x0[0:4]
        dxx = dxx[0:4]
        
    if model_type=='tm5_smr':  
        x0 = par['x0'][0:10]
        dxx = par['dxx'][0:10]
        
        
    if model_type=='tm4x2':
        do_TT3=False
#         x0 = par['x0'][0:9]
#         dxx = par['dxx'][0:9] 
        x0 = par['x0'][0:10]
        dxx = par['dxx'][0:10]
        if do_TT3:
            x0 = par['x0'][0:14]
            dxx = par['dxx'][0:14]    
        

    #print(x0)
    Q0, A0 = QA_TM_simple(x0, amps, sig, Ts,indexes_amps=indexes_amps,model_type=model_type, model_type2=model_type2)
    #print(x0)
    print('initial Q: ',str(Q0))

    #           tF      p0    tD        dp           A                A1                  A2              tDmin   dd   tD2
    #xlower = [  0.1,   0.01,   0.1,     0.0001,     amps[0]/1*1e-1,   amps[0]/1*1e-1,     amps[0]/1*1e-1,  1,     1e-3, 1]
    xlower = x0/dxx
    xlower = np.array(xlower)
    #xupper = [  1e5,   1,      1e5,     1,          amps[0]/0.01*1e1, amps[0]/0.01*1e1,   amps[0]/0.01*1e1, 100,  1,    1e4]
    #xupper = np.array(xupper)
    xupper = x0*dxx
    bounds = [] #np.zeros(len(xlower))
    for ibo in range(len(xlower)):
        bounds = bounds + [(xlower[ibo],xupper[ibo])]
    tt=[]
    #import time
    #import scipy.optimize as opt
    t1 = time.time()

    do_this=1
    if do_this==1:
        #res4 = opt.basinhopping(Q_TM_aba_synphys, x0 , niter=100, T=1.0, stepsize=1.0,
        #                        minimizer_kwargs={'method':"L-BFGS-B", 'args':(amps, sig, Ts,DT0)}, 
        #                        take_step=None, accept_test=None, callback=None, interval=50, disp=False,
        #                        niter_success=None, seed=None)
        #nst=50 # good
        nst = 50 #100 #150 # better fit
        Qs =  np.zeros(nst)
        As =  np.zeros((nst,npar))
        #vx = np.arange(5)  # fit 1 amplitude
        vx = np.arange(len(x0)) # fit separate amplitude for each frequency
        nx = len(x0)
        xs =  np.zeros((nst,len(x0[vx]) ))
        for ist in range(nst):

            #x0i = np.exp(np.log(xlower) + np.random.rand(len(xlower))*(np.log(xupper)-np.log(xlower)))
            x0i = x0*np.exp( 2*(np.random.rand(len(xlower))-0.5)*np.log(dxx)  )
            x0i = x0i[vx]
            #x0i[4:7] = x0i[4]
            fl=1
            #breakpoint()
            #print(x0i)
            resi=opt.minimize(Q_TM_simple, x0i, args=((amps, sig, Ts, indexes_amps, model_type, model_type2)), method=None, jac=None,
                      hess=None, hessp=None, bounds=bounds[0:nx],
                      constraints=(), tol=None, callback=None, 
                      options=None)
            x=resi.x
            if (model_type=='tm4'):
                #x = [tF, p0,tD,  dp]
                x[3] = x[1] # dp = p0
            if (model_type=='tm4x2'):
                #x = [tF, p0,tD,  dp, A2, tF2, p02, tD2, dp2, A3, tF3, p03, tD3, dp3]
                #x = [tF, p0,tD,  dp, A2, tF2, p02, tD2, dp2]
                 
#                 x[4] = np.abs(x[4])
#                 x[9] = np.abs(x[9])
#                 if x[4]+x[9]>1:
#                     cc=x[4] +x[9]
#                     x[4], x[9] = x[4]/cc, x[9]/cc
                x[3] = x[3] 
#                 x[3] = x[1] # dp = p0
#                 x[8] = x[6] # dp2 = p02
            Q,A = QA_TM_simple(x, amps, sig, Ts,indexes_amps=indexes_amps,model_type=model_type, model_type2=model_type2)
            As[ist,:] = A
            xs[ist,:] = x
            Qs[ist] = Q

            
            Q_rs = np.min(Qs)
            ia = np.nonzero(Q_rs==Qs)[0]
            x_rs=xs[ia,:].ravel()
            A_rs = As[ia,:].ravel()
    else: 
        x_rs=x0
        Q_rs = 1e5
        A_rs = np.zeros((npar,))
            
    t2=time.time()
    tt = tt + [t2-t1]
    print('random_start: elapsed time '+str(t2-t1)+'s ')
    
    return x_rs, Q_rs, A_rs

# Amplitudes from STP model
def  STP_sim(ge_data, T, init_state=None ):
    # 
    
    # transform labels from TM to An:A1
    #f = 20 # Hz
    #N = 3
    #T = np.arange(N)*1000/f

    N    = len(T)
    nc   = len(ge_data.index)

    x_lower = np.array([1,       0.01,    1,       0.1,     1])
    x_upper = np.array([10000,   1,      10000,    10,      1])
    stp_ns  =          ['tF',   'p0',    'tD',    'dp/p0', 'A']
    for jj in range(len(stp_ns)):
        nsj = stp_ns[jj]
        ge_data.loc[:,nsj] = np.maximum(ge_data.loc[:,nsj].values,x_lower[jj])
        ge_data.loc[:,nsj] = np.minimum(ge_data.loc[:,nsj].values,x_upper[jj])
        if nsj=='dp/p0':
            p0   = ge_data.loc[:,'p0'].values
            dpp0 = ge_data.loc[:,'dp/p0'].values
            dp = p0*dpp0
            dp = np.minimum(dp,1)
            ge_data.loc[:,nsj] = dp/(p0 +np.finfo(float).eps)

    dpp0 = ge_data.loc[:,'dp/p0'].values
    p0   = ge_data.loc[:,'p0'].values
    tF   = ge_data.loc[:,'tF'].values
    tD   = ge_data.loc[:,'tD'].values
    A    = 1 + 0*ge_data.loc[:,'A'].values # simplify A

    As = np.zeros((nc,N))
    n = np.zeros((nc,))
    p = np.zeros((nc,))
    ns2 = np.zeros((nc,N))
    ps2 = np.zeros((nc,N))

    i=0
    
    if init_state is None :
        n[:] = 1
        p[:] = p0
    else:
        n = init_state[0]
        p = init_state[1]


    As[:,i] = A*n*p
    
    n = n*(1-p)
    p = p + dp*(1-p)
    
    ns2[:,i]=n
    ps2[:,i]=p

    for i in range(1,N):
        Dt=T[i]-T[i-1]
        #n = 1 - (1 - (n -p*n))*np.exp((-Dt/tD).astype(float))
        #p=p0 -(p0-(p + dpp0*p0*(1-p)))*np.exp((-Dt/tF).astype(float))
        #As[:,i]=A*n*p
        #ns2[:,i]=n
        #ps2[:,i]=p

        
        n = 1 - (1 - n )*np.exp((-Dt/tD ).astype(float))
        p=p0 +(p -p0)*np.exp((-Dt/tF).astype(float))
            

        As[:,i]=A*n*p
       
        n = n*(1-p)
        p = p + dp*(1-p)
        
        ns2[:,i]=n
        ps2[:,i]=p

    
    #aa = [As, ns2, ps2]
    
    return As, ns2, ps2, dpp0, p0, tF, tD, A


def  STP_sim2(x, T, init_state=None, model_type='tm5' ):
    do_TT3=False
    N    = len(T)
    tF      = x[0] #.astype(float)
    p00     = x[1]
    tD      = x[2] #.astype(float)
    dp      = x[3]
    A       = 1 #x[4] # simplify A
    
    #breakpoint()
    mod_fdr2=False
    if model_type=='tm5_fdr2':  # should be :check freq. dependent recovery
        tDmin     = x[5]
        dd        = x[6]
        t_FDR     = x[7]
        mod_fdr2=True
        tDmax  = tD
        itDmin = 1/tDmin
        itDmax = 1/tDmax
        #breakpoint()
        
    mod_smr=False
    if model_type=='tm5_smr':  # should be :check freq. dependent recovery
        t_SMR   = x[8] #x[5] #x[8]
        dp0     = x[9] #x[8] #x[9]
        mod_smr=True
        #p00  = p00
        
    mod_tmx2=False
    p002 = 0
    p003 = 0
    if model_type=='tm4x2':  # should be :check freq. dependent recovery
        tF2      = x[5] #.astype(float)
        p002     = x[1] #x[6]
        tD2      = x[7] #.astype(float)
        A2       = x[4]
        dp2      = x[8] #p002
        mod_tmx2=True
        #p00  = p00 
        
        ## TTx3
        A3       = x[9] #ACHTUNG!!!
        if do_TT3:
            tF3      = x[10] #.astype(float)
            p003     = x[11]
            tD3      = x[12] #.astype(float)
            A3       = x[9]
            dp3      = x[13] #p002

    As = np.zeros((N,))
    dim=4
    if mod_tmx2:
        dim=6
        ## TTx3
        if do_TT3:
            dim=8
    state = np.zeros((N*2,dim))

   
    if init_state is None :
        n = 1
        p0=p00
        p = p0
        d = 0
        
        n2 = 1
        p02=p002
        p2 = p02
        d2 = 0
        
        n3 = 1
        p03=p003
        p3 = p03
        
    else:
        n = init_state[0]
        p = init_state[1]
        d = init_state[2]
        p0= init_state[3]
        if mod_tmx2:
            n2 = init_state[4]
            p2 = init_state[5]
            p02 = p002
            
            ## TTx3
            if do_TT3:
                n3 = init_state[6]
                p3 = init_state[7]
                p03 = p003
            

    
    for i in range(0,N):
        if i==0:
            Dt = T[i]     
        else:
            if (T[i-1]==np.inf):
                Dt = T[i]
            else:
                Dt = T[i]-T[i-1]       
        if (T[i]==np.inf):
                Dt = 0
                n = 1
                p0=p00
                p = p0
                d = 0

                n2 = 1
                p02=p002
                p2 = p02
                  
                if do_TT3:
                    n3=1
                    p03=p003
                    p3=p03
                
                        
        if mod_fdr2:
            d0=d
            d = d*np.exp(-Dt/t_FDR) 
            n = 1 - (1 - n )*np.exp(-Dt*itDmax -(itDmin -itDmax)*t_FDR*(d0-d))
        else:
            #print(x,tD)
            n = 1 - (1 - n )*np.exp(-Dt/tD )
            
        if mod_smr:
            p01=p0
            p0=p00 + (p0 -p00)*np.exp(-Dt/t_SMR)
            p=p0 +(p -p01)*np.exp(-Dt/tF)
        elif mod_tmx2:
            n2 = 1 - (1 - n2 )*np.exp(-Dt/tD2 )
            
            p21 = p2
            p2=p02 +(p2 -p02)*np.exp(-Dt/tF2)
            
            p=p2 +(p -p21)*np.exp(-Dt/tF)
            #p=p0 +(p -p0)*np.exp(-Dt/tF)
            
            ## TTx3
            if do_TT3:
                n3 = 1 - (1 - n3 )*np.exp(-Dt/tD3 )
                p3=p03 +(p3 -p03)*np.exp(-Dt/tF3)            
        else:
            p=p0 +(p -p0)*np.exp(-Dt/tF)


        if mod_tmx2:
            state[2*i] = [n,p,d,p0, n2,p2]
            ## TTx3
            if do_TT3:
                state[2*i] = [n,p,d,p0, n2,p2,n3,p3]
        else:
            state[2*i] = [n,p,d,p0]    
        
       
        #n = n*(1-p)
        #p = p + dp*(1-p)
        if mod_tmx2: 
            if do_TT3:
                As[i] = (1-A2)*As[i] + A2*A*n2*p2  +A3*A*n3*p3 - A3*As[i]
            else:
                #As[i] = (1-A2)*A*n*p + A2*A*n2*p2 
                As[i] = A*p*((1-A2)*n + A2*n2)
#                 As[i] = A*((1-A3)*p + A3*p2)*((1-A2)*n + A2*n2)
            
#             n = n*(1-p)
#             n2 = n2*(1-p)
#             p = p + dp*(1-p)
#             p2 = p2 + dp2*(1-p2)
            
            n = n*(1-p)
            p = p + dp*(1-p)
            n2 = n2*(1-p)
            p2 = p2 + dp2*(-p2)
            
            ## TTx3
            if do_TT3:
                n3 = n3*(1-p3)
                p3 = p3 + dp3*(1-p3)

        else: 
            As[i]=A*n*p
            n = n*(1-p)
            p = p + dp*(1-p)                
        
        
        if mod_fdr2:
            d  = d + dd*(1-d) 
        if mod_smr:
            p0  = p0 - dp0*p0    
 
        if mod_tmx2:
            state[2*i+1] = [n,p,d,p0, n2,p2]
            ## TTx3
            if do_TT3:
                state[2*i+1] = [n,p,d,p0, n2,p2,n3,p3]
        else:
            state[2*i+1] = [n,p,d,p0]

    #return As, ns2, ps2, dpp0, p0, tF, tD, A
    return As, state


def STP_sim_complex(ge_data,l_pre_post2,stp_aba_names):
    # transform labels from TM to An:A1
    #fs = [20, 50, 10] # Hz
    fs = [20, 50, 10]
    N = 5

    #Trec = [250, 500, 1000]
    Trec =[250, 1000]
    DT0 = 25000
    if l_pre_post2<ge_data.shape[0]:
        xs  =ge_data.iloc[l_pre_post2:,:].loc[:,stp_aba_names].values
        xs  =np.delete(xs, [5,6],axis=1)
        As2=np.zeros((xs.shape[0],0))
    
    
    As=np.zeros((l_pre_post2,0))
    if 1:
        raz=0
        
        for f in fs:
            raz=raz+1
            T = np.arange(N)*1000/f

            if l_pre_post2>0:
                if raz==1:
                    print('model type = 4 parameters TM')
                Asf, ns, ps, dpp0, p0, tF, tD, A = STP_sim(ge_data.iloc[0:l_pre_post2,:],T)
                As = np.concatenate([As,Asf],axis=1)

                for ri in range(len(Trec)):
                    Asr, nsr, psr, dpp0, p0, tF, tD, A = STP_sim(ge_data.iloc[0:l_pre_post2,:],[Trec[ri]],init_state=[ns[:,-1],ps[:,-1]])
                    As = np.concatenate([As,Asr],axis=1)

            if l_pre_post2<ge_data.shape[0]:
                if raz==1:
                    print('model type =  TM 5 parameters , smr')
                    
                As2f = np.zeros((xs.shape[0],N+len(Trec)))
                for i2 in range(xs.shape[0]):
                    #as2, sts2 = STP_sim2(xs[i2,:],np.arange(8)*50,model_type = 'tm5') # preconditioning series???
                    #as2, sts2 = STP_sim2(xs[i2,:], T+DT0, init_state=sts2[-1], model_type='tm5' ) 

                    as2, sts2 = STP_sim2(xs[i2,:],T,model_type = 'tm5_smr')

                    As2f[i2,0:N] = as2
                    for ri in range(len(Trec)):
                        as2r, sts2r = STP_sim2(xs[i2,:], [Trec[ri]], init_state=sts2[-1], model_type='tm5_smr' ) 
                        As2f[i2,N+ri] = as2r

                As2 = np.concatenate([As2,As2f],axis=1)

        if l_pre_post2<ge_data.shape[0]:
            As2 = As2/As2[:,0].reshape((-1,1))
        else:
            As2  = np.zeros((0,As.shape[1]))
        
        
        if l_pre_post2>0:
            As = As/As[:,0].reshape((-1,1))
            As = np.concatenate([As,As2],axis=0)
        else:
            As =As2
    else:
        As, ns, ps, dpp0, p0, tF, tD, A = STP_sim(ge_data,T)


    #import matplotlib.pyplot as plt    
    #plt.plot(As2[400:410,:].transpose(),'o-')
    
    return As




# EPSP from STP amplitudes
def epsp_sim(A, T_A,dt,tplus,tminus,Tmax):
    
    l, v = scipy.linalg.eig(np.array([[-1/tplus,0],[1/tplus,-1/tminus]])) 
    l = np.real(l).reshape([-1,1])
    vm = scipy.linalg.inv(v)
    #x = v*np.exp(l*t)*np.array([1,0]).reshape([-1,1])                  
    X = np.array([0,0]).reshape([-1,1])

    t=np.array([0])
    for i in np.arange(1,T_A.size):
        if T_A[i]-T_A[i-1]>Tmax+10*dt:
            t2  =  np.arange(int(Tmax/dt))[np.newaxis]*dt  
            t3  =  - 10*dt + np.arange(10)[np.newaxis]*dt   
            
            t5 = np.append(t2,t3+Tmax+10*dt ,axis=1)
            t2 = np.append(t2,t3+T_A[i] - T_A[i-1],axis=1)

        else: 
            t2  =  np.arange(int((T_A[i]-T_A[i-1])/dt))[np.newaxis]*dt 
            t5 = t2

        #print(i)    
        x0 = X[:,[-1]]   
        x0[0,:] = x0[0,:]+A[i-1] 
        vvmx=(v@np.diag((vm@x0).ravel()))
        x = vvmx@np.exp(l@t2)
        X = np.append(X,x,axis=1)
        #t = np.append(t,T_A[i-1] +t5)   
        t = np.append(t,t[-1] +t5)   
    
    return X, t

def plot_fig9b(i0, ge_data, tplus=5, tminus=50, dt=0.1, Tmax=1000, Tend=400, T_A=np.array([0,100]), Nbootstraps=200, Dn_ams=1, Dn_epsp=5, model_type='tm',only_amlitudes=True):
    """
      i0 : index of synaptic pair type in ge_data
      ge_data : dataframe with a set of bootstraped stp parameters for each synaptic pair type
      Nbootstraps : number of bootstraps 
      Dn : step of indexes - stp data from i0*Nbootstraps to (i0+1)*Nbootstraps lines with step Dn will be taken from ge_data 
      
    """
    #dt=0.1
    #tplus=5
    #tminus=50
    #Tmax = 1000
    #DT = 200/6

    #i0=23 #42 #[18,23,42]
    #ii = i0*Nbootstraps + np.arange(0,Nbootstraps,Dn)
    ii = i0*Nbootstraps + np.arange(0,Nbootstraps,Dn_ams)
    ##T_A = np.array((100+np.arange(8)*200/6).tolist()+[100+200/6*8+500]) #  from BBP cell paper figure
    #T_A = np.array((100+np.arange(9)*200/6).tolist()+[100+200/6*8+400]) # for classification task
    
    
    
    if model_type=='tm':
        #stp_aba_names = ['tF', 'p0','tD','dp','A','A1','A2'] # tm5
        stp_aba_names = ['tF', 'p0', 'tD', 'dp/p0', 'A'] #, 'A1', 'A2','tDmin','dd','t_FDR','t_SMR','dp0']
        #stp_aba_names = ['tF', 'p0','tD','dp','A','A1','A2','tDmin','dd','t_FDR','t_SMR','dp0'] # tm5+smr
    if model_type=='tm_smr':
        stp_aba_names = ['tF', 'p0','tD','dp','A','A1','A2','tDmin','dd','t_FDR','t_SMR','dp0'] # tm5+smr
    
    
    #print(model_type)
    if model_type=='tm':
#         T_A0 = T_A-T_A[0]
#         As, ns, ps, dpp0, p0, tF, tD, A = STP_sim(ge_data.iloc[ii,:],T_A0)
#         stp_aba_names = ['tF', 'p0','tD','dp','A','A1','A2']
#         xs  =ge_data.iloc[ii,:].loc[:,stp_aba_names].values
#         STS = [ns,ps]
        
        xs  =ge_data.iloc[ii,:].loc[:,stp_aba_names].values
        #print(ii[0]/200)
        #print('xs0 _ ',xs[0,:])
        
        #xs  =np.delete(xs, [5,6],axis=1)
        xs  = np.concatenate([xs, np.ones([xs.shape[0],5])],axis=1)
        
        stp_aba_names = ['tF', 'p0','tD','dp','A','tDmin','dd','t_FDR','t_SMR','dp0']
        
        #if xs[0,3]==np.nan:
        xs[:,3] = xs[:,1]*xs[:,3] # dp = p0 * dp/p0
        xs[xs[:,3]>1,3]=1 #dp = np.minimum(dp,1)
        xs[:,8] = 100
        xs[:,9] = 0
        
        #print('xs _ ',xs[0,:])
        
        T_A0 = T_A-T_A[0]
        As = np.zeros((xs.shape[0],T_A.shape[0]))
        STS = []
        for i2 in range(xs.shape[0]):
            #print(T_A0)
            as2, sts2 = STP_sim2(xs[i2,:],T_A0,model_type='tm5_smr') 
            As[i2,0:as2.shape[0]] = as2
            STS = STS +[sts2]        
        
        
        
    if model_type=='tm_smr':
        xs  =ge_data.iloc[ii,:].loc[:,stp_aba_names].values
        xs  =np.delete(xs, [5,6],axis=1)
        stp_aba_names = ['tF', 'p0','tD','dp','A','tDmin','dd','t_FDR','t_SMR','dp0']
        
        #print(xs)
        
        T_A0 = T_A-T_A[0]
        As = np.zeros((xs.shape[0],T_A.shape[0]))
        STS = []
        for i2 in range(xs.shape[0]):
            #print(T_A0)
            as2, sts2 = STP_sim2(xs[i2,:],T_A0,model_type='tm5_smr') 
            As[i2,0:as2.shape[0]] = as2
            STS = STS +[sts2]

    
    
    #A = np.mean(As,axis=0) #As[0,:Na1]#np.mean(As[:100,:Na1],axis=0)
    #A = np.append([0],A)
    ##T_A = Tall_A
    
    T_A =np.append([0],T_A)
    T_A = np.append(T_A,[T_A[-1]+Tend])
    
    
    Epsp =[]
    mEpsp = np.array([])
    t = np.array([])
    if only_amlitudes==False:
        for iii in range(0,As.shape[0],Dn_epsp):
            #A = As[iii,:]
            A = As[iii,:]/As[iii,0]
            A = np.append([0],A)
            A = np.append(A,[0])

            #print(T_A, A)
            epsp, t = epsp_sim(A, T_A,dt,tplus,tminus,Tmax)
            Epsp = Epsp + [epsp[1,:].reshape([-1,1])]


        Epsp = np.concatenate(Epsp, axis=1)
        mEpsp = np.mean(Epsp,axis=1)
    
    
    
    #plt.plot(t, epsp[1,:].T)
    
    return mEpsp, t, As, T_A0, Epsp, (T_A/dt).astype(int), pd.DataFrame(xs,columns=stp_aba_names), STS