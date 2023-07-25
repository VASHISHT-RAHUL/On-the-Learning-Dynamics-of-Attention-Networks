#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys

sys.path.append('/home/malt/Documents/Notebooks')


# In[2]:


from models import *
from train import *
from generate_data import *




import numpy as np
import pandas as pd
from tqdm import tqdm as tqdm
from matplotlib import pyplot as plt
#%matplotlib inline




import torch
import torchvision
from torch.utils.data import Dataset,DataLoader

import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import torchvision.transforms as transforms



torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# # Generate Dataset

# In[3]:
#----------------------------------------------------------------------------------------------

# Change here for different settings such as number of classes, number of patches etc
n_classes = 20  # no of classes
data_size = 2000  # base data size
patch_dim = n_classes+1 

batch = 256  # batch size for mosaic data training
tr = 5000    # training data size

size = 5000 # mosaic data size

m  = 20 # number of patches
# ---------------------------------------------------------------------------------


y = []
for i in range(patch_dim):
    if i ==0:
        y.append(np.zeros(data_size,dtype=np.compat.long))
    else:
        y.append(np.ones(data_size,dtype=np.compat.long)*int(i))
y = np.hstack(y)


idx = []
for i in range(patch_dim):
    idx.append(y==i)

x = np.zeros((y.shape[0],patch_dim),dtype=np.float32)
print("creating base data")
for i in tqdm(range(patch_dim)):
    if i!=n_classes:
        mu = np.eye(1,patch_dim,i)[0,:]
        covariance = np.diag(mu*1e-10)   # variance for foreground data 1e-10
        x[idx[i]] = np.random.multivariate_normal(mean=mu,
                                                 cov=covariance,size=sum(idx[i])).astype(dtype=np.float32)
    else:
        mu = np.zeros(patch_dim)
        covariance = np.diag(np.concatenate([np.zeros(patch_dim-1),
                                             np.ones(1)*1e-2],axis=0)) # variance for background data 1e-2
        x[idx[i]] = np.random.multivariate_normal(mean=mu,
                                                  cov = covariance,size=sum(idx[i])).astype(dtype=np.float32)
print("base data created ")
    



mosaic_data = []
mosaic_label = []
fore_idx = []
for j in tqdm(range(size)):
    np.random.seed(j)
    fg_class = np.random.randint(0,n_classes)
    fg_idx = 0 #np.random.randint(0,m)
    a = []
    for i in range(m):
        if i == fg_idx:
            b = np.random.choice(np.where(idx[fg_class]==True)[0],size=1)
            a.append(x[b])
        else:
            bg_class = np.random.randint(n_classes,n_classes+1)
            b = np.random.choice(np.where(idx[bg_class]==True)[0],size=1)
            a.append(x[b])
    a = np.concatenate(a,axis=0)
    mosaic_data.append(np.reshape(a,(m,patch_dim)))
    mosaic_label.append(fg_class)
    fore_idx.append(fg_idx)
print("Mosaic Data Created")


# In[4]:


class MosaicDataset(Dataset):
  """MosaicDataset dataset."""

  def __init__(self, mosaic_list_of_images, mosaic_label, fore_idx):
    """
      Args:
        csv_file (string): Path to the csv file with annotations.
        root_dir (string): Directory with all the images.
        transform (callable, optional): Optional transform to be applied
            on a sample.
    """
    self.mosaic = mosaic_list_of_images
    self.label = mosaic_label
    self.fore_idx = fore_idx

  def __len__(self):
    return len(self.label)

  def __getitem__(self, idx):
    return self.mosaic[idx] , self.label[idx], self.fore_idx[idx]


# In[5]:



msd = MosaicDataset(mosaic_data[0:tr], mosaic_label[0:tr] , fore_idx[0:tr])
train_loader = DataLoader( msd,batch_size= batch ,shuffle=False)

batch = 256
msd1 = MosaicDataset(mosaic_data[tr:], mosaic_label[tr:] , fore_idx[tr:])
test_loader = DataLoader( msd1,batch_size= batch ,shuffle=False)


# In[6]:


n_epochs = 100
learning_rate = 0.05#,0.1,0.05,0.02,0.01,0.005,0.002,0.001,0.0005]


alphas = [0.2,0.4,0.6,0.8,1]

loss_dict = {}
loss_diff_dict = {}
for run_no in range(len(alphas)):

    torch.manual_seed(12)
    classification = Classification_linear(patch_dim,n_classes)
    classification = classification.to(device)


    lr = learning_rate
    

    #Criterion = nn.CrossEntropyLoss(reduction="none") #nn.BCELoss(reduction="none")
    optimizer_classification = optim.SGD(classification.parameters(), lr=lr,momentum=0.9)
    loss_list = []
    loss_alpha_list = []
    
    if alphas[run_no] ==1:
        alpha_ = alphas[run_no]
    else:
        alpha_ = alphas[run_no] + 0.01
    
    epoch_loss = []
    epoch_loss_alpha_ = []
    classification_epoch_loss = []
    for j,data in enumerate(train_loader):
        images,labels,foreground_index = data
        images = images.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            loss = marginal_loss_sin(classification,images,labels,alphas[run_no])
            loss_alpha_ = marginal_loss_sin(classification,images,labels,alpha_)
        epoch_loss.append(loss.item())
        epoch_loss_alpha_.append(loss_alpha_.item())
    print("*"*60)
    print("Epoch: " + str(0)+", Loss: "+str(np.mean(epoch_loss)))
    loss_list.append(np.mean(epoch_loss))
    loss_alpha_list.append(np.mean(epoch_loss_alpha_))

    for i in range(n_epochs):
        epoch_loss = []
        epoch_loss_alpha_ = []
        classification_epoch_loss = []
        for j,data in enumerate(train_loader):
            images,labels,foreground_index = data
            images = images.to(device)
            labels = labels.to(device)
        
            optimizer_classification.zero_grad()
            
            loss = marginal_loss_sin(classification,images,labels,alphas[run_no])
            loss.backward()
            optimizer_classification.step()
            
            with torch.no_grad():
                loss = marginal_loss_sin(classification,images,labels,alphas[run_no])
                loss_alpha_ = marginal_loss_sin(classification,images,labels,alpha_)
                epoch_loss.append(loss.item())
                epoch_loss_alpha_.append(loss_alpha_.item())
        print("*"*60)
        print("Epoch: " + str(i+1)+", Loss: "+str(np.mean(epoch_loss)))

        loss_list.append(np.mean(epoch_loss))
        loss_alpha_list.append(np.mean(epoch_loss_alpha_))
    loss_dict[str(alphas[run_no])] = np.array(loss_list)
    loss_diff_dict[str(alphas[run_no])] = np.array(loss_list) - np.array(loss_alpha_list) 
    


# In[10]:


k = 0 

color_list = ["#e41a1c", "#377eb8","#4daf4a","#984ea3","#ff7f00"]
plt.figure(figsize=(8,6))
for alpha_no in alphas:
    plt.plot(loss_dict[str(alpha_no)][:100],label = r"$\bf alpha=$"+str(alpha_no),c=color_list[k],linewidth=3.0)
    k = k+1

plt.xlabel("Epochs",weight="bold",fontsize=15)
plt.ylabel(r"$\bf L^{\mathrm{\bf LV}}$($ \alpha$)",weight="bold",fontsize=15)
plt.xticks([0,20,40,60,80,100],weight="bold",fontsize=15)
plt.yticks([3,2,1,0],weight="bold",fontsize=15)
plt.legend(bbox_to_anchor=[1., 1],prop={'size': 14})
plt.savefig("LVML_m_"+str(m)+"_c_"+str(n_classes)+".pdf",bbox_inches='tight')
plt.show()


# In[11]:


k = 0 
color_list = ["#e41a1c", "#377eb8","#4daf4a","#984ea3","#ff7f00"]
plt.figure(figsize=(8,6))
for alpha_no in alphas:
    plt.plot(loss_diff_dict[str(alpha_no)][:100],label = r"$\bf alpha=$"+str(alpha_no),c=color_list[k],linewidth=3.0)
    k = k+1

plt.xlabel("Epochs",weight="bold",fontsize=15)
plt.ylabel(r"$\bf L^{\mathrm{\bf LV}}$($ \alpha$)",weight="bold",fontsize=15)
plt.xticks([0,20,40,60,80,100],weight="bold",fontsize=15)
plt.yticks([0.03,0.02,0.01,0],weight="bold",fontsize=15)
plt.legend(bbox_to_anchor=[1., 1],prop={'size': 14})
plt.savefig("LVML_m_"+str(m)+"_c_"+str(n_classes)+".pdf",bbox_inches='tight')
plt.show()


# In[ ]:




