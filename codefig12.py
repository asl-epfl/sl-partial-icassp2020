"""
This code can be used to generate simulations similar to Figs. 1 and 2 in the following paper:
Virginia Bordignon, Vincenzo Matta, and Ali H. Sayed, ``Social learning with partial information sharing,''  Proc. IEEE ICASSP, Barcelona, Spain, May 2020.

Please note that the code is not generally perfected for performance, but is rather meant to illustrate certain results from the paper. The code is provided as-is without guarantees.

July 2020 (Author: Virginia Bordignon)
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
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
Gr= nx.from_numpy_array(A)
pos = nx.spring_layout(Gr)
#%%
f,ax=plt.subplots(1,1, figsize=(4,2))
plt.axis('off')
plt.xlim([-1.2,1.2])
plt.ylim([-1.15,1.1])
nx.draw_networkx_nodes(Gr, pos=pos, node_color= 'C4',nodelist=[0],node_size=700, edgecolors='k', linewidths=.5)
nx.draw_networkx_nodes(Gr, pos=pos, node_color= 'C2',nodelist=range(1,N),node_size=700, edgecolors='k', linewidths=.5)
nx.draw_networkx_labels(Gr,pos,{i: i+1 for i in range(N)},font_size=16, font_color='black', alpha = 1)
nx.draw_networkx_edges(Gr, pos = pos, node_size=500, alpha=1, arrowsize=6, width=1);
plt.tight_layout()
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.savefig(FIG_PATH + 'fig1.pdf', format='pdf', bbox_inches='tight', pad_inches=0)
#%%
################################ Run Social Learning ################################
theta = np.array([Decimal(0), Decimal(0.2), Decimal(1)])
var = Decimal(1)
x = np.linspace(-4, 6, 1000)
x = decimal_array(x)
dt = (max(x)-min(x))/len(x)
#%%
mu_0 = np.random.rand(N,M)
mu_0 = mu_0/np.sum(mu_0, axis = 1)[:, None]
mu_0 = decimal_array(mu_0)
#%%
csi=[]
for l in range(N):
    csi.append(theta[0]+np.sqrt(var)*decimal_array(np.random.randn(N_ITER)))
csi=np.array(csi)
#%%
MU_ob_0 = partial_info(mu_0, csi, A_dec, N_ITER, theta, var, M, N)
vec_0 = np.array([MU_ob_0[k][0] for k in range(len(MU_ob_0))])
MU_ob_1 = partial_info(mu_0, csi, A_dec, N_ITER, theta, var, M, N, tx=1)
vec_1 = np.array([MU_ob_1[k][0] for k in range(len(MU_ob_1))])
MU_ob_2 = partial_info(mu_0, csi, A_dec, N_ITER, theta, var, M, N, tx=2)
vec_2 = np.array([MU_ob_2[k][0] for k in range(len(MU_ob_2))])
#%%
plt.figure(figsize=(12,2.5))
gs = gridspec.GridSpec(1, 4, width_ratios=[1.5,1,1,1])
plt.subplot(gs[0])

L0 = gaussian(x, theta[0], var,)
L1 = gaussian(x, theta[1], var)
L2 = gaussian(x, theta[2], var)

plt.plot(x, L0)
plt.plot(x, L1)
plt.plot(x, L2)
plt.ylabel(r'$L(\xi|\theta)$', fontsize = 15)
plt.xlabel(r'$\xi$', fontsize = 15)
plt.tight_layout()
plt.xlim(-4,6)
plt.title('Gaussian Likelihoods', fontsize=15)
plt.tight_layout()

plt.subplot(gs[1])
plt.plot(vec_0[:,0], linewidth='1.5', color = 'C0')
plt.plot(vec_0[:,1], linewidth='1.5', color = 'C1')
plt.plot(vec_0[:,2], linewidth='1.5', color = 'C2')
plt.annotate('%0.2f' % vec_0[-1,0], (1.01, vec_0[-1,0]),xycoords=('axes fraction', 'data'), color = 'C0')
plt.annotate('%0.2f' % vec_0[-1,1], (1.01, vec_0[-1,1]),xycoords=('axes fraction', 'data'), color = 'C1')
plt.annotate('%0.2f' % vec_0[-1,2], (1.01, vec_0[-1,2]-Decimal(.1)),xycoords=('axes fraction', 'data'), color = 'C2')
plt.xlim([0,200])
plt.ylim([-0.1,1.1])
plt.title(r'$\theta_{\sf TX}=1$', fontsize = 15)
plt.xlabel(r'$i$', fontsize = 15)
plt.ylabel(r'$\mu_{{{l},i}}(\theta)$'.format(l=1), fontsize = 15)
plt.tight_layout()

plt.subplot(gs[2])
plt.plot(vec_1[:,0], linewidth='1.5', color = 'C0')
plt.plot(vec_1[:,1], linewidth='1.5', color = 'C1')
plt.plot(vec_1[:,2], linewidth='1.5', color = 'C2')
plt.annotate('%0.2f' % vec_1[-1,0], (1.01, vec_1[-1,0]),xycoords=('axes fraction', 'data'), color = 'C0')
plt.annotate('%0.2f' % vec_1[-1,1], (1.01, vec_1[-1,1]),xycoords=('axes fraction', 'data'), color = 'C1')
plt.annotate('%0.2f' % vec_1[-1,2], (1.01, vec_1[-1,2]-Decimal(.1)),xycoords=('axes fraction', 'data'), color = 'C2')
plt.xlim([0,200])
plt.ylim([-0.1,1.1])
plt.title(r'$\theta_{\sf TX}=2$', fontsize = 15)
plt.xlabel(r'$i$', fontsize = 15)
plt.ylabel(r'$\mu_{{{l},i}}(\theta)$'.format(l=1), fontsize = 15)
plt.tight_layout()

plt.subplot(gs[3])
plt.plot(vec_2[:,0], linewidth='1.5', color = 'C0', label=r'$\theta=1$')
plt.plot(vec_2[:,1], linewidth='1.5', color = 'C1', label = r'$\theta=2$')
plt.plot(vec_2[:,2], linewidth='1.5', color = 'C2', label = r'$\theta=3$')
plt.annotate('%0.2f' % vec_2[-1,0], (1.01, vec_2[-1,0]),xycoords=('axes fraction', 'data'), color = 'C0')
plt.annotate('%0.2f' % vec_2[-1,1], (1.01, vec_2[-1,1]-Decimal(.1)),xycoords=('axes fraction', 'data'), color = 'C1')
plt.annotate('%0.2f' % vec_2[-1,2], (1.01, vec_2[-1,2]),xycoords=('axes fraction', 'data'), color = 'C2')
plt.xlim([0,200])
plt.ylim([-0.1,1.1])
plt.title(r'$\theta_{\sf TX}=3$', fontsize = 15)
plt.xlabel(r'$i$', fontsize = 15)
plt.ylabel(r'$\mu_{{{l},i}}(\theta)$'.format(l=1), fontsize = 15)
plt.tight_layout()
plt.subplots_adjust(bottom=0.32)
plt.figlegend(ncol=M,fontsize = 14, bbox_to_anchor=(0.13,-0.31, 0.5, 0.5), handlelength=1)
plt.savefig(FIG_PATH + 'fig2.pdf', bbox_inches='tight')
