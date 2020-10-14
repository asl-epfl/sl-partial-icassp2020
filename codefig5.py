"""
This code can be used to generate simulations similar to Fig. 5 in the following paper:
Virginia Bordignon, Vincenzo Matta, and Ali H. Sayed, ``Social learning with partial information sharing,''  Proc. IEEE ICASSP, Barcelona, Spain, May 2020.

Please note that the code is not generally perfected for performance, but is rather meant to illustrate certain results from the paper. The code is provided as-is without guarantees.

July 2020 (Author: Virginia Bordignon)
"""
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import random
import os
import networkx as nx
from functions import *
#%%
mpl.style.use('seaborn-deep')
plt.rcParams.update({'text.usetex': True})
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Computer Modern Roman'
getcontext().prec = 200
#%%
FIG_PATH = 'figs/'
if not os.path.isdir(FIG_PATH):
    os.makedirs(FIG_PATH)
#%%
N=10
M=3
N_ITER = 200
N_MC = 1
np.random.seed(2)
#%%
################################ Build Network Topology ################################
G = np.random.choice([0.0,1.0],size=(N,N), p=[0.5,0.5])
G = G+G.T+np.eye(N)
G=(G>0)*1.0
#%%
lamb = .5
A = np.zeros((N,N))
for i in range(N):
    A[G[i]>0, i]=(1-lamb)/(np.sum(G[i])-1)
    A[i,i]=lamb
A_dec = np.array([[Decimal(x) for x in y] for y in A])
#%%
################################ Run Social Learning ################################
theta = np.array([Decimal(0), Decimal(0.2), Decimal(1)])
var = Decimal(1)
x = np.linspace(-4, 6, 1000)
x = decimal_array(x)
dt = (max(x)-min(x))/len(x)
#%%
np.random.seed(0)
mu_0 = np.ones((N,M))
mu_0 = mu_0/np.sum(mu_0, axis = 1)[:, None]
mu_0 = decimal_array(mu_0)
#%%
csi=[]
for l in range(N):
    csi.append(theta[0]+np.sqrt(var)*decimal_array(np.random.randn(N_ITER)))
csi=np.array(csi)
#%%
MU_max_uni= partial_info(mu_0, csi, A_dec, N_ITER, theta, var, M, N, tx='max', self_aware = False)
mu_uni=[MU_max_uni[k][0] for k in range(len(MU_max_uni))]
#%%
N_MC=100
#%%
getcontext().prec = 20
ALL_MU = mc_partial_info(N_MC, A_dec, N_ITER, theta, var, M, N, tx='max', self_aware = False)
#%%
plt.figure(figsize=(5,2.2))
gs = gridspec.GridSpec(1, 2, width_ratios=[ 1, 1])
plt.subplot(gs[:,0])
plt.plot(np.array(mu_uni)[:,0], linewidth='1.5', alpha=1,color = 'C0',label=r'$\theta=1$')
plt.plot(np.array(mu_uni)[:,1], linewidth='1.5', alpha=1,color = 'C1',label=r'$\theta=2$')
plt.plot(np.array(mu_uni)[:,2], linewidth='1.5', alpha=1,color = 'C2',label=r'$\theta=3$')
plt.annotate('%0.2f' % np.array(mu_uni)[:,0][-1], (1.02, np.array(mu_uni)[:,0][-1]),xycoords=('axes fraction', 'data'), color = 'C0')
plt.annotate('%0.2f' % np.array(mu_uni)[:,1][-1], (1.02, np.array(mu_uni)[:,1][-1]),xycoords=('axes fraction', 'data'), color = 'C1')
plt.annotate('%0.2f' % np.array(mu_uni)[:,2][-1], (1.02, np.array(mu_uni)[:,2][-1]-Decimal(.12)),xycoords=('axes fraction', 'data'), color = 'C2')
plt.xlim([0,N_ITER])
plt.ylim([-0.1,1.1])
plt.xlabel(r'$i$', fontsize = 15)
plt.ylabel(r'$\mu_{1,i}(\theta)$',fontsize=16)
plt.figlegend(ncol=3,fontsize=12, handlelength=1, bbox_to_anchor=[.32,-.31,.5,.5])
plt.tight_layout()
plt.title('Uniform initialization', fontsize=13)

plt.subplot(gs[:,1])
for j in range(len(ALL_MU)):
    h0=plt.plot([ALL_MU[j][k][0,0] for k in range(len(ALL_MU[j]))], linewidth='.2', alpha=.7,color = 'C0')
plt.xlim([0,N_ITER])
plt.ylim([-0.1,1.1])
plt.ylabel(r'$\mu_{1,i}(\theta)$',fontsize=16)
plt.xlabel(r'$i$', fontsize = 15)
plt.title('Random initialization', fontsize=13)
plt.tight_layout()
plt.subplots_adjust(bottom=.35, wspace=.55)

plt.savefig(FIG_PATH+'fig5.pdf', bbox_inches='tight')
