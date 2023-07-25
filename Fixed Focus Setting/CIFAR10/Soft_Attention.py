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

#---------------------------------------------------------------------------------------------------------
# Change here for different settings such as number of classes, number of patches etc


batch = 256  # batch size for mosaic data training

m  = 5 # number of patches

desired_num = 20000  # total number of points training +test
tr = 10000    # number of training points

nos_epochs = 100

learning_rates = 0.004#,0.1,0.05,0.02,0.01,0.005,0.002,0.001,0.0005]
#----------------------------------------------------------------------------------------------

fg1, fg2, fg3 = 0,1,2
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)


testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=10, shuffle=False)
testloader = torch.utils.data.DataLoader(testset, batch_size=10, shuffle=False)


classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

foreground_classes = {'plane', 'car', 'bird'}

background_classes = {'cat', 'deer', 'dog', 'frog', 'horse','ship', 'truck'}

# print(type(foreground_classes))

dataiter = iter(trainloader)
background_data=[]
background_label=[]
foreground_data=[]
foreground_label=[]
batch_size=10

for i in tqdm(range(5000)):   #5000*batch_size = 50000 data points
    images, labels = dataiter.next()
    for j in range(batch_size):
        if(classes[labels[j]] in background_classes):
            img = images[j].tolist()
            background_data.append(img)
            background_label.append(labels[j])
        else:
            img = images[j].tolist()
            foreground_data.append(img)
            foreground_label.append(labels[j])
            
foreground_data = torch.tensor(foreground_data)
foreground_label = torch.tensor(foreground_label)
background_data = torch.tensor(background_data)
background_label = torch.tensor(background_label)
print("Foreground Background Data created")



mosaic_data =[]      # list of mosaic images, each mosaic image is saved as list of 9 images
fore_idx =[]                   # list of indexes at which foreground image is present in a mosaic image i.e from 0 to 9               
mosaic_label=[]                # label of mosaic image = foreground class present in that mosaic
list_set_labels = [] 
for i in tqdm(range(desired_num)):
    set_idx = set()
    np.random.seed(i)
    bg_idx = np.random.randint(0,35000,m-1)
    set_idx = set(background_label[bg_idx].tolist())
    fg_idx = np.random.randint(0,15000)
    set_idx.add(foreground_label[fg_idx].item())
    fg = 0 #np.random.randint(0,m)
    fore_idx.append(fg)
    image_list,label = create_mosaic_img(foreground_data,background_data,foreground_label,bg_idx,fg_idx,fg,m)
    mosaic_data.append(image_list)
    mosaic_label.append(label)
    list_set_labels.append(set_idx)
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


# In[7]:




alphas = [0.2,0.4,0.6,0.8,1]


n_seeds = [12]
loss_dict = {}
loss_diff_dict = {}

for seeds in n_seeds: 
    for run_no in range(len(alphas)):
        
        
        torch.manual_seed(seeds)
        classification = Classification_cnn(3)
        classification = classification.to(device)


        lr = learning_rates

        Criterion = nn.CrossEntropyLoss() #nn.BCELoss(reduction="none")
        classification_optimizer = optim.SGD(classification.parameters(),lr=lr,momentum=0.9)

        loss_list = []
        loss_alpha_list = []
        
        if alphas[run_no] == 1:
            alpha_ = alphas[run_no]
        else:
            alpha_ = alphas[run_no] + 0.01
        
        epoch_loss = [] 
        epoch_loss_alpha_ = []
        for i,data in enumerate(train_loader):
            with torch.no_grad():
                images,labels,fore_idx = data
                batch = images.size(0)
                patches = images.size(1)
                images,labels = images.to(device),labels.to(device)
                    
                alpha_weights = torch.ones((batch,patches,3,32,32),device=device)*((1-alphas[run_no])/(patches-1))
                alpha_weights[:,0] = torch.ones((batch,3,32,32),device=device)*alphas[run_no]

                    
                images_1 = torch.mul(alpha_weights,images)

                outputs = classification(torch.sum(images_1,dim=1))
                    
                loss = Criterion(outputs,labels)
                    
                alpha_weights_ = torch.ones((batch,patches,3,32,32),device=device)*((1-alpha_)/(patches-1))
                alpha_weights_[:,0] = torch.ones((batch,3,32,32),device=device)*alpha_
                    
                images_ = torch.mul(alpha_weights_,images)
                
    
                outputs_ = classification(torch.sum(images_,dim=1))                              
                                              
                loss_alpha_ = Criterion(outputs_,labels,)

            epoch_loss.append(loss.item())
            epoch_loss_alpha_.append(loss_alpha_.item())
        print('[%d] loss: %.3f' %(0,np.mean(epoch_loss)))
            
        loss_list.append(np.mean(epoch_loss))
        loss_alpha_list.append(np.mean(epoch_loss_alpha_))

        for epoch in range(nos_epochs):
            classification.train()

            epoch_loss = [] 
            epoch_loss_alpha_ = []

            for i,data in enumerate(train_loader):
                classification,classification_optimizer=train_model_cin(data,alphas[run_no],
                                                                        classification,
                                                                        classification_optimizer,
                                                                        Criterion)

                with torch.no_grad():
                    images,labels,fore_idx = data
                    batch = images.size(0)
                    patches = images.size(1)
                    images,labels = images.to(device),labels.to(device)
                    
                    alpha_weights = torch.ones((batch,patches,3,32,32),device=device)*((1-alphas[run_no])/(patches-1))
                    alpha_weights[:,0] = torch.ones((batch,3,32,32),device=device)*alphas[run_no]
                    
                    
                    images_1 = torch.mul(alpha_weights,images)

                            
                    outputs = classification(torch.sum(images_1,dim=1))
                    
                    loss = Criterion(outputs,labels)
                    
                    alpha_weights_ = torch.ones((batch,patches,3,32,32),device=device)*((1-alpha_)/(patches-1))
                    alpha_weights_[:,0] = torch.ones((batch,3,32,32),device=device)*alpha_
                    
                    images_ = torch.mul(alpha_weights_,images)
                
    
                    outputs_ = classification(torch.sum(images_,dim=1))                              
                                                  
                    loss_alpha_ = Criterion(outputs_,labels,)

                epoch_loss.append(loss.item())
                epoch_loss_alpha_.append(loss_alpha_.item())
            print('[%d] loss: %.3f' %(epoch+1,np.mean(epoch_loss)))
            
            loss_list.append(np.mean(epoch_loss))
            loss_alpha_list.append(np.mean(epoch_loss_alpha_))
        loss_dict[str(alphas[run_no])] = np.array(loss_list)
        loss_diff_dict[str(alphas[run_no])] = np.array(loss_list) - np.array(loss_alpha_list)

print("Finished Training")


# In[9]:


k = 0 

color_list = ["#e41a1c", "#377eb8","#4daf4a","#984ea3","#ff7f00"]
plt.figure(figsize=(8,6))
for alpha_no in alphas:
    plt.plot(loss_dict[str(alpha_no)][:100],label = r"$\bf alpha=$"+str(alpha_no),c=color_list[k],linewidth=3.0)
    k = k+1

plt.xlabel("Epochs",weight="bold",fontsize=15)
plt.ylabel(r"$\bf L^{\mathrm{\bf SA}}$($ \alpha$)",weight="bold",fontsize=15)
plt.xticks([0,20,40,60,80,100],weight="bold",fontsize=15)
plt.yticks([1,0.5,0],weight="bold",fontsize=15)
plt.legend(bbox_to_anchor=[1., 1],prop={'size': 14})
plt.savefig("SA_m_"+str(m)+"_size_"+str(tr/1000)+"k.pdf",bbox_inches='tight')
plt.show()


# In[10]:


k = 0 
color_list = ["#e41a1c", "#377eb8","#4daf4a","#984ea3","#ff7f00"]
plt.figure(figsize=(8,6))
for alpha_no in alphas:
    plt.plot(loss_diff_dict[str(alpha_no)][:100],label = r"$\bf alpha=$"+str(alpha_no),c=color_list[k],linewidth=3.0)
    k = k+1

plt.xlabel("Epochs",weight="bold",fontsize=15)
plt.ylabel(r"$\bf L^{\mathrm{\bf SA}}$($ \alpha$)",weight="bold",fontsize=15)
plt.xticks([0,20,40,60,80,100],weight="bold",fontsize=15)
plt.yticks([0.03,0.02,0.01,0],weight="bold",fontsize=15)
plt.legend(bbox_to_anchor=[1., 1],prop={'size': 14})
plt.savefig("SA_m_"+str(m)+"_size_"+str(tr/1000)+"k.pdf",bbox_inches='tight')
plt.show()


# In[ ]:




