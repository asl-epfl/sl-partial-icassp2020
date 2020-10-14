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
lamb = .03
A = np.zeros((N,N))
for i in range(N):
    A[G[i]>0, i]=(1-lamb)/(np.sum(G[i])-1)
    A[i,i]=lamb
A_dec = np.array([[Decimal(x) for x in y] for y in A])
#%%
np.random.seed(0)
getcontext().prec = 100
#%%
L = np.array([np.array([Decimal(.1), Decimal(.7), Decimal(.2)]), np.array([Decimal(.15), Decimal(.65), Decimal(.2)]), np.array([Decimal(.4), Decimal(.2), Decimal(.4)])])
Lf = np.array([[float(x) for x in y] for y in L])
#%%
mu_0 = np.random.rand(N,M)
mu_0 = mu_0/np.sum(mu_0, axis = 1)[:, None]
mu_0 = np.array([[Decimal(x) for x in y] for y in mu_0])
#%%
csi = []
for l in range(N):
    csi.append(np.random.choice([0,1,2],size=N_ITER, p=Lf[0]))
csi=np.array(csi)
#%%
MU_sa_0 = partial_info_d(mu_0, csi, A_dec, L, N_ITER, M, N, tx=0, self_aware = True)
vec_0sa= np.array([MU_sa_0[k][0] for k in range(len(MU_sa_0))])
MU_sa_1 = partial_info_d(mu_0, csi, A_dec, L, N_ITER, M, N, tx=1, self_aware = True)
vec_1sa= np.array([MU_sa_1[k][0] for k in range(len(MU_sa_1))])
MU_sa_2 = partial_info_d(mu_0, csi, A_dec, L, N_ITER, M, N, tx=2, self_aware = True)
vec_2sa= np.array([MU_sa_2[k][0] for k in range(len(MU_sa_2))])
#%%
import matplotlib.gridspec as gridspec
plt.figure(figsize=(12,2.5))
gs = gridspec.GridSpec(1, 4, width_ratios=[1.5,1,1,1])

plt.subplot(gs[0])
plt.grid(True, axis='y')
h1=plt.bar(np.arange(3)-.1,L[0], width=.1)
h2=plt.bar(np.arange(3),L[1], width=.1)
h3=plt.bar(np.arange(3)+.1,L[2], width=.1)
plt.ylabel(r'$L(\xi|\theta)$', fontsize = 15)
plt.xlabel(r'$\xi$', fontsize = 15)
plt.tight_layout()
plt.title('Discrete Likelihoods', fontsize=15)
plt.xticks(np.arange(3))

plt.subplot(gs[1])
plt.plot(vec_0sa[:,0], linewidth='1.5', color = 'C0')
plt.plot(vec_0sa[:,1], linewidth='1.5', color = 'C1')
plt.plot(vec_0sa[:,2], linewidth='1.5', color = 'C2')
plt.annotate('%0.2f' % vec_0sa[-1,0], (1.01, vec_0sa[-1,0]),xycoords=('axes fraction', 'data'), color = 'C0')
plt.annotate('%0.2f' % vec_0sa[-1,1], (1.01, vec_0sa[-1,1]),xycoords=('axes fraction', 'data'), color = 'C1')
plt.annotate('%0.2f' % vec_0sa[-1,2], (1.01, vec_0sa[-1,2]-Decimal(.1)),xycoords=('axes fraction', 'data'), color = 'C2')
plt.xlim([0,200])
plt.ylim([-0.1,1.1])
plt.title(r'$\theta_{\sf TX}=1$', fontsize = 15)
plt.xlabel(r'$i$', fontsize = 15)
plt.ylabel(r'$\mu_{{{l},i}}(\theta)$'.format(l=1), fontsize = 15)
plt.tight_layout()

plt.subplot(gs[2])
plt.plot(vec_1sa[:,0], linewidth='1.5', color = 'C0')
plt.plot(vec_1sa[:,1], linewidth='1.5', color = 'C1')
plt.plot(vec_1sa[:,2], linewidth='1.5', color = 'C2')
plt.annotate('%0.2f' % vec_1sa[-1,0], (1.01, vec_1sa[-1,0]),xycoords=('axes fraction', 'data'), color = 'C0')
plt.annotate('%0.2f' % vec_1sa[-1,1], (1.01, vec_1sa[-1,1]),xycoords=('axes fraction', 'data'), color = 'C1')
plt.annotate('%0.2f' % vec_1sa[-1,2], (1.01, vec_1sa[-1,2]-Decimal(.1)),xycoords=('axes fraction', 'data'), color = 'C2')
plt.xlim([0,200])
plt.ylim([-0.1,1.1])
plt.title(r'$\theta_{\sf TX}=2$', fontsize = 15)
plt.xlabel(r'$i$', fontsize = 15)
plt.ylabel(r'$\mu_{{{l},i}}(\theta)$'.format(l=1), fontsize = 15)
plt.tight_layout()

a=plt.subplot(gs[3])
a.plot(vec_2sa[:,0], linewidth='1.5', color = 'C0', label=r'$\theta=1$')
a.plot(vec_2sa[:,1], linewidth='1.5', color = 'C1', label = r'$\theta=2$')
a.plot(vec_2sa[:,2], linewidth='1.5', color = 'C2', label = r'$\theta=3$')
a.annotate('%0.2f' % vec_2sa[-1,0], (1.005, vec_2sa[-1,0]),xycoords=('axes fraction', 'data'), color = 'C0')
a.annotate('%0.2f' % vec_2sa[-1,1], (1.005, vec_2sa[-1,1]-Decimal(.1)),xycoords=('axes fraction', 'data'), color = 'C1')
a.annotate('%0.2f' % vec_2sa[-1,2], (1.005, vec_2sa[-1,2]),xycoords=('axes fraction', 'data'), color = 'C2')
a.set_xlim([0,100])
a.set_ylim([-0.1,1.1])
rect = [0.5,0.7,0.3,0.25]
ax1 = add_subplot_axes(a,rect)
ax1.plot(vec_2sa[:,0], linewidth='1.5', color = 'C0')
ax1.plot(vec_2sa[:,1], linewidth='1.5', color = 'C1')
ax1.set_ylim([0.499, 0.501])
ax1.set_xlim([80, 90])
ax1.set_xticklabels('')
ax1.set_yticks([0.499, 0.501])
ax1.set_yticklabels([0.499, 0.501])
a.set_title(r'$\theta_{\sf TX}=3$', fontsize = 15)
a.set_xlabel(r'$i$', fontsize = 15)
a.set_ylabel(r'$\mu_{{{l},i}}(\theta)$'.format(l=1), fontsize = 15)
a.add_line(mpl.lines.Line2D([80,55],[.5, 0.7], color='black', linestyle='dashed' ,linewidth=.8))
a.add_line(mpl.lines.Line2D([90,85],[.5, 0.7], color='black', linestyle='dashed',linewidth=.8))

plt.tight_layout()
plt.subplots_adjust(bottom=0.32)
plt.figlegend(ncol=M,fontsize = 14, bbox_to_anchor=(0.13,-0.31, 0.5, 0.5), handlelength=1)
plt.savefig(FIG_PATH + 'fig3.pdf', bbox_inches='tight')
