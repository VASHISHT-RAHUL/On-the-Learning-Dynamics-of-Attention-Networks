#!/usr/bin/env python
# coding: utf-8

# In[56]:


# import wandb 
# wandb.login()

get_ipython().run_line_magic('reset', '-f')
import sys



sys.path.append('/Users/rahulvashisht/Downloads')

from models import *
from train import *
from generate_data import *

import numpy as np
import pandas as pd
from tqdm import tqdm as tqdm

from matplotlib import pyplot as plt
import matplotlib


import torch
import torchvision

import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset,DataLoader


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# # solving differential equation numerically
# 
# ## Soft Attention
# $$\dfrac{\partial \mu(t)}{\partial t} = \dfrac{\alpha(t)}{\exp(\alpha(t)\mu(t))+C-1}$$
# 
# $$\dfrac{\partial \nu(t)}{\partial t} = \dfrac{\alpha(t)(1-\alpha(t))(C-1)\mu(t)}{C\exp(\alpha(t)\mu(t))+C-1}$$
# 
# ## Hard Attention
# 
# $$\dfrac{\partial \mu(t)}{\partial t} =\dfrac{\alpha(t)}{\exp(mu(t))+C-1} $$
# $$ \dfrac{\partial \nu(t)}{\partial t}= \frac{1}{C}\alpha(t)(1-\alpha(t)) \log[C\beta(t)]t $$
# 
# ## Marginal Loss based Hard Attention
# 
# $$\dfrac{\partial \mu(t)}{\partial t} = \dfrac{\alpha(t)(\beta(t))^2}{Z(t)\exp(\mu(t))} $$
# 
# $$ \dfrac{\partial \nu(t)}{\partial t} = \dfrac{\alpha(t)}{C}\Bigg[ \dfrac{\beta(t)}{Z(t)}-1\Bigg]$$
# 
# 
# where, $$Z(t) = \alpha(t) \beta(t) + \dfrac{1-\alpha(t)}{C}$$

# In[172]:


m = 100 #number of patches
c = 1000 # number of classes
n = 40000000  # 100000 for m 20 c 20 for higher values of m and c needs to run for a large number of steps

def calculate_(mu,nu,method="soft"):
    alpha = np.exp(nu)/(np.exp(nu)+m-1)
    beta = np.exp(mu) / (np.exp(mu)+c-1)
    #print(alpha)
    if method == "soft":
        a = alpha/(np.exp(alpha*mu)+c-1)
        mu_hat = mu*alpha
        b = (((1)/c)*((m-1)*(c-1)*mu_hat))/( (np.exp(nu)+m-1) *(np.exp(mu_hat)+c-1) ) 
        #b = ((alpha)*(1-alpha)*c-1*mu)/(np.exp(alpha*mu)+c-1)
    elif method == "hard":
        a = (alpha/ (np.exp(mu) +c-1))
        temp1 = alpha*((alpha*np.log(beta)) + (np.log(1/c)*(1-alpha)) ) 
        temp2 = np.log(beta)*alpha
        b = (1/c)*(temp2-temp1)
    elif method == "marginal":
        z_t = (alpha *beta ) + ( (1-alpha)/c ) 
        a = (beta*alpha )/ ((np.exp(mu)+c-1)*z_t)
        b = (alpha/c)*( (beta/z_t) - 1)
    mu = mu + a*1e-2
    nu = nu + b*1e-2
    return mu,nu


# In[174]:


mu_sa_list =  []
nu_sa_list = []
mu_ha_list = []
nu_ha_list = []
mu_ml_list = []
nu_ml_list = []


mu_sa = 0
nu_sa = 0
mu_ha = 0
nu_ha = 0
mu_ml = 0
nu_ml = 0


for i in tqdm(range(n)):
    mu_sa,nu_sa = calculate_(mu_sa,nu_sa,"soft")
    #print(nu)
    mu_sa_list.append(mu_sa)
    nu_sa_list.append(nu_sa)
    
    mu_ha, nu_ha = calculate_(mu_ha,nu_ha,"hard")
    mu_ha_list.append(mu_ha)
    nu_ha_list.append(nu_ha)   
    
    mu_ml, nu_ml = calculate_(mu_ml,nu_ml,"marginal")
    mu_ml_list.append(mu_ml)
    nu_ml_list.append(nu_ml)   


# In[175]:


plt.plot(mu_sa_list[0:n:100],label="analytic")
#plt.plot(mu1_sa[:,0],label="training")
plt.legend()


# In[176]:


plt.plot(nu_sa_list[0:n:100],label="analytic")
# plt.plot(nu_sa[:,0],label ="training")
plt.legend()


# In[177]:


# plt.plot(np.ediff1d(nu_sa_list[0::100]))


# In[178]:


# plt.plot(np.ediff1d(mu_sa_list[0:50000000:100]))


# In[179]:


plt.plot(mu_ha_list[0:n:100],label="analytic")


# In[180]:


plt.plot(nu_ha_list[0:n:100],label="analytic")


# In[181]:


# mu_ml_list =  []
# nu_ml_list = []
# mu = 0
# nu = 0
# for i in tqdm(range(50000000)):
#   mu,nu = calculate_(mu,nu,"marginal")
#   #print(nu)
#   mu_ml_list.append(mu)
#   nu_ml_list.append(nu)


# In[182]:


# plt.plot(mu_ml_list[0:50000000:100],label="analytic")


# In[183]:


# plt.plot(nu_ml_list[0:50000000:100],label="analytic")


# In[184]:


plt.figure(figsize=(8,6))
plt.plot(mu_sa_list[0:n:100],label="Soft Attention",linewidth=3)
plt.plot(mu_ha_list[0:n:100],label="Hard Attention",linewidth=3)
plt.plot(mu_ml_list[0:n:100],label="LVML",linewidth=3)
plt.xticks([0,100000,200000,300000,400000], 
           ["0", "100K", "200K", "300K", "400K"],weight="bold",fontsize=15)
plt.yticks([0,3,6,9,12],weight="bold",fontsize=15)
# plt.xticks([0,200,400,600,800,1000],weight="bold",fontsize=15)
# plt.yticks([0,2,4,6],weight="bold",fontsize=15)
plt.xlabel("time",weight="bold",fontsize=15)
plt.ylabel(r"$\bf \mu(t)$",weight="bold",fontsize=15)
plt.legend(bbox_to_anchor=[1,0.25],prop={'size':13})
plt.savefig("mu_t.pdf",bbox_inches="tight")


# In[185]:


plt.figure(figsize=(8,6))
ax = plt.plot(nu_sa_list[0:n:100],label="Soft Attention",linewidth=3)
plt.plot(nu_ha_list[0:n:100],label="Hard Attention",linewidth=3)
plt.plot(nu_ml_list[0:n:100],label="LVML",linewidth=3)


plt.xticks([0,100000,200000,300000,400000], 
           ["0", "100K", "200K", "300K", "400K"],weight="bold",fontsize=15)
plt.yticks([0,3,6,9,12],weight="bold",fontsize=15)
# plt.xticks([0,200,400,600,800,1000],weight="bold",fontsize=15)
# plt.yticks([0,2,4,6],weight="bold",fontsize=15)
plt.xlabel("time",weight="bold",fontsize=15)
plt.ylabel(r"$\bf \nu(t)$",weight="bold",fontsize=15)
#plt.legend(bbox_to_anchor=[1,0.3],prop={'size':18})
plt.savefig("nu_t.pdf",bbox_inches="tight")


# In[ ]:





# In[ ]:




