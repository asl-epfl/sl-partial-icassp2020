
"""
This code can be used to generate simulations similar to Fig. 3 in the following paper:
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
np.random.seed(0)
getcontext().prec = 100
#%%
L = np.array([np.array([Decimal(.1), Decimal(.7), Decimal(.2)]), np.array([Decimal(.15), Decimal(.65), Decimal(.2)]), np.array([Decimal(.4), Decimal(.2), Decimal(.4)])])
Lf = np.array([[float(x) for x in y] for y in L])
#%%
################################ Run Social Learning ################################
mu_0 = np.random.rand(N,M)
mu_0 = mu_0/np.sum(mu_0, axis = 1)[:, None]
mu_0 = np.array([[Decimal(x) for x in y] for y in mu_0])
#%%
np.random.seed(0)
N_ITER=500
#%%
csi = []
for l in range(N):
    csi.append(np.random.choice([0,1,2],size=N_ITER, p=Lf[0]))
csi=np.array(csi)
#%%
lamb = .9
A = np.zeros((N,N))
for i in range(N):
    A[G[i]>0, i]=(1-lamb)/(np.sum(G[i])-1)
    A[i,i]=lamb
A_dec = np.array([[Decimal(x) for x in y] for y in A])
#%%
MU_sa_29 = partial_info_d(mu_0, csi, A_dec, L, N_ITER, M, N, tx=2, self_aware = True)
vec_2sa_9=np.array([MU_sa_29[k][0] for k in range(len(MU_sa_29))])
#%%
lamb = .99
A = np.zeros((N,N))
for i in range(N):
    A[G[i]>0, i]=(1-lamb)/(np.sum(G[i])-1)
    A[i,i]=lamb
A_dec = np.array([[Decimal(x) for x in y] for y in A])
#%%
MU_sa_299 = partial_info_d(mu_0, csi, A_dec, L, N_ITER, M, N, tx=2, self_aware = True)
vec_2sa_99=np.array([MU_sa_299[k][0] for k in range(len(MU_sa_299))])
#%%
plt.figure(figsize=(5,2.2))
gs = gridspec.GridSpec(1, 2, width_ratios=[1,1])

plt.subplot(gs[0])
plt.plot(vec_2sa_9[:,0], linewidth='1.5', color = 'C0', label=r'$\theta=1$')
plt.plot(vec_2sa_9[:,1], linewidth='1.5', color = 'C1', label = r'$\theta=2$')
plt.plot(vec_2sa_9[:,2], linewidth='1.5', color = 'C2', label = r'$\theta=3$')
plt.annotate('%0.2f' % vec_2sa_9[-1,0], (1.01, vec_2sa_9[-1,0]),xycoords=('axes fraction', 'data'), color = 'C0')
plt.annotate('%0.2f' % vec_2sa_9[-1,1], (1.01, vec_2sa_9[-1,1]+Decimal(.07)),xycoords=('axes fraction', 'data'), color = 'C1')
plt.annotate('%0.2f' % vec_2sa_9[-1,2], (1.01, vec_2sa_9[-1,2]),xycoords=('axes fraction', 'data'), color = 'C2')
plt.xlim([0,200])
plt.ylim([-0.1,1.1])
plt.title(r'$\lambda = 0.9$', fontsize = 13)
plt.xlabel(r'$i$', fontsize = 15)
plt.ylabel(r'$\mu_{{{l},i}}(\theta)$'.format(l=1), fontsize = 16)
plt.figlegend(ncol=3,fontsize=12, handlelength=1, bbox_to_anchor=[.32,-.31,.5,.5])
plt.tight_layout()

plt.subplot(gs[1])
plt.plot(vec_2sa_99[:,0], linewidth='1.5', color = 'C0', label=r'$\theta=1$')
plt.plot(vec_2sa_99[:,1], linewidth='1.5', color = 'C1', label = r'$\theta=2$')
plt.plot(vec_2sa_99[:,2], linewidth='1.5', color = 'C2', label = r'$\theta=3$')
plt.annotate('%0.2f' % vec_2sa_99[-1,0], (1.01, vec_2sa_99[-1,0]),xycoords=('axes fraction', 'data'), color = 'C0')
plt.annotate('%0.2f' % vec_2sa_99[-1,1], (1.01, vec_2sa_99[-1,1]+Decimal(.07)),xycoords=('axes fraction', 'data'), color = 'C1')
plt.annotate('%0.2f' % vec_2sa_99[-1,2], (1.01, vec_2sa_99[-1,2]),xycoords=('axes fraction', 'data'), color = 'C2')
plt.xlim([0,200])
plt.ylim([-0.1,1.1])
plt.title(r'$\lambda = 0.99$', fontsize = 13)
plt.xlabel(r'$i$', fontsize = 15)
plt.ylabel(r'$\mu_{{{l},i}}(\theta)$'.format(l=1), fontsize = 16)
plt.tight_layout()
plt.subplots_adjust(bottom=.35, wspace=.55)
plt.savefig(FIG_PATH+'fig4.pdf', bbox_inches='tight')
