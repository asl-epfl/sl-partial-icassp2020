import numpy as np
from decimal import *
import matplotlib.pyplot as plt

def gaussian(x, m, var):
    '''
    Computes the Gaussian pdf value at x.
    x: value at which the pdf is computed (Decimal type)
    m: mean (Decimal type)
    var: variance
    '''
    p = np.exp(-(x-m)**Decimal(2)/(Decimal(2)*Decimal(var)))/(np.sqrt(Decimal(2)*Decimal(np.pi)*Decimal(var)))
    return p

def bayesian_update(L, mu):
    '''
    Computes the Bayesian update.
    L: likelihoods matrix
    mu: beliefs matrix
    '''
    aux = L*mu
    bu = aux/aux.sum(axis = 1)[:, None]
    return bu

def DKL(m,n,dx):
    '''
    Computes the KL divergence between m and n.
    m: true distribution in vector form
    n: second distribution in vector form
    dx : sample size
    '''
    mn=m/n
    mnlog= np.log(mn)
    return np.sum(m*dx*mnlog)

def decimal_array(arr):
    '''
    Converts an array to an array of Decimal objects.
    arr: array to be converted
    '''
    if len(arr.shape)==1:
        return np.array([Decimal(y) for y in arr])
    else:
        return np.array([[Decimal(x) for x in y] for y in arr])

def partial_info(mu_0, csi, A_dec, N_ITER, theta, var, M, N, tx = 0, self_aware = False, partial=True):
    '''
    Executes the social learning algorithm.
    mu_0: initial beliefs
    csi: observations
    A_dec: Combination matrix (Decimal type)
    N_ITER: number of iterations
    theta: vector of means for the Gaussian likelihoods
    var: variance of Gaussian likelihoods
    M: number of hypothese
    N: number of agents
    tx: transmitted hypothesis (can be a numerical value or 'max')
    self_aware: self-awareness flag
    partial: partial information flag
    '''
    mu = mu_0.copy()
    MU = [mu]
    for i in range(N_ITER):
        L_i = np.array([gaussian(csi[:,i], t, var) for t in theta]).T

        psi = bayesian_update(L_i, mu)
        decpsi = np.array([[x.ln() for x in y] for y in psi])
        if partial:
            if tx=='max': # share theta max
                aux = np.array([[Decimal(1) for x in y] for y in mu])*(1-np.max(psi,axis=1))[:,None]/(M-1)
                aux[np.arange(N), np.argmax(psi,axis=1)]=np.max(psi,axis=1)
            else: # share tx
                aux = np.array([[Decimal(1) for x in y] for y in mu])*(1-psi[:,tx])[:, None]/(M-1)
                aux[np.arange(N), np.ones(N, dtype=int)*tx]=psi[:,tx]
            decaux = np.array([[x.ln() for x in y] for y in aux])

        if partial:
            if not self_aware: # Without non-cooperative term:
                mu = np.exp((A_dec.T).dot(decaux))/np.sum(np.exp((A_dec.T).dot(decaux)),axis =1)[:,None]
            else: # With non-cooperative term:
                mu = np.exp(np.diag(np.diag(A_dec)).dot(decpsi)+(A_dec.T-np.diag(np.diag(A_dec))).dot(decaux))/np.sum(np.exp(np.diag(np.diag(A_dec)).dot(decpsi)+(A_dec.T-np.diag(np.diag(A_dec))).dot(decaux)),axis =1)[:,None]
        else:
            mu = np.exp((A_dec.T).dot(decpsi))/np.sum(np.exp((A_dec.T).dot(decpsi)),axis =1)[:,None]

        MU.append(mu)
    return MU

def mc_partial_info(N_MC, A_dec, N_ITER, theta, var, M, N, tx=0, self_aware = False, partial=True):
    '''
    Executes N_MC Monte Carlo runs of the social learning algorithm.
    N_MC: number of Monte Carlo runs
    A_dec: combination matrix (Decimal type)
    N_ITER: number of iterations
    theta: vector of means for the Gaussian likelihoods
    var: variance of Gaussian likelihoods
    M: number of hypotheses
    N: number of agents
    tx: transmitted hypothesis (can be a numerical value or 'max')
    self_aware: self-awareness flag
    partial: partial information flag
    '''
    ALL_MU = [
    for i in range(N_MC):
        mu_0 = np.random.rand(N,M)
        mu_0 = mu_0/np.sum(mu_0, axis = 1)[:, None]
        mu_0 = decimal_array(mu_0)

        csi=[]
        for l in range(N):
            csi.append(theta[0]+np.sqrt(var)*decimal_array(np.random.randn(N_ITER)))
        csi=np.array(csi)

        MU = partial_info(mu_0, csi, A_dec, N_ITER, theta, var, M, N, tx, self_aware = False, partial=partial)

        ALL_MU.append(MU)
    return ALL_MU

def partial_info_d(mu_0, csi, A_dec, L, N_ITER, M, N, tx=0, self_aware = False):
    '''
    Executes the social learning algorithm for a discrete family of likelihoods.
    mu_0: initial beliefs
    csi: observations
    A_dec: combination matrix (Decimal type)
    L: likelihoods matrix
    N_ITER: number of iterations
    M: number of hypotheses
    N: number of agents
    tx: transmitted hypothesis (can be a numerical value or 'max')
    self_aware: self-awareness flag
    '''
    mu = mu_0.copy()
    MU = [mu]
    for i in range(N_ITER):
        L_i = L[:,csi[:,i]].T
        psi = bayesian_update(L_i, mu)

        if tx=='max': # share theta max
            aux = np.array([[Decimal(1) for x in y] for y in mu])*(1-np.max(psi,axis=1))[:,None]/(M-1)
            aux[np.arange(N), np.argmax(psi,axis=1)]=np.max(psi,axis=1)
        else: # share tx
            aux = np.array([[Decimal(1) for x in y] for y in mu])*(1-psi[:,tx])[:, None]/(M-1)
            aux[np.arange(N), np.ones(N, dtype=int)*tx]=psi[:,tx]

        decpsi = np.array([[x.ln() for x in y] for y in psi])
        decaux = np.array([[x.ln() for x in y] for y in aux])

        if not self_aware: # Without non-cooperative term:
            mu = np.exp((A_dec.T).dot(decaux))/np.sum(np.exp((A_dec.T).dot(decaux)),axis =1)[:,None]
        else: # With non-cooperative term:
            mu = np.exp(np.diag(np.diag(A_dec)).dot(decpsi)+(A_dec.T-np.diag(np.diag(A_dec))).dot(decaux))/np.sum(np.exp(np.diag(np.diag(A_dec)).dot(decpsi)+(A_dec.T-np.diag(np.diag(A_dec))).dot(decaux)),axis =1)[:,None]

        MU.append(mu)
    return MU

def add_subplot_axes(ax, rect, axisbg='w'):
    '''
    Adds an axis within a graph.
    '''
    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position  = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]  # <= Typo was here
    subax = fig.add_axes([x,y,width,height])
    return subax
