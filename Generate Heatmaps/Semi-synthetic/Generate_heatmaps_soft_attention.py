#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append('/home/malt/Documents/Notebooks')
# sys.path.append(path)
from models import *
from train import *
from generate_data import *
import numpy as np
import pandas as pd
from tqdm import tqdm as tqdm
from matplotlib import pyplot as plt
import torch
import torchvision
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import torchvision.transforms as transforms
from matplotlib.colors import ListedColormap
import seaborn as sns

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)


# In[2]:


# Change for Different Settings here
#------------------------------------------------------------------------------------
m = 20  # number of patches
desired_num = 60000 # number of training points train + test
batch = 256  # batch size for mosaic data
tr = 50000 # number of training points

algorithm = "soft" # soft, hard or LVML here
which_data = "CIFAR100" # CIFAR10 or CIFAR100
#------------------------------------------------------------------------------------------------


# In[3]:


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


# In[4]:


def labels_to_index(fg_labels):
    unique_foreground_labels = list(np.unique(fg_labels))
    foreground_labels = []
    for fg in fg_labels:
        foreground_labels.append(unique_foreground_labels.index(fg))
    return torch.tensor(foreground_labels,dtype=torch.int64)


# In[5]:


def Create_Mosaic_data_cifar10(desired_num,m,foreground_label,background_label,foreground_data,background_data,dataset="None"):

    if dataset =="training":
        n_bg = 35000
        n_fg = 15000
    elif dataset == "test":
        n_bg = 7000
        n_fg = 3000

    mosaic_data =[]      # list of mosaic images, each mosaic image is saved as list of 9 images
    fore_idx =[]                   # list of indexes at which foreground image is present in a mosaic image i.e from 0 to 9               
    mosaic_label=[]                # label of mosaic image = foreground class present in that mosaic
    list_set_labels = [] 
    for i in tqdm(range(desired_num)):
        set_idx = set()
        np.random.seed(i)
        bg_idx = np.random.randint(0,n_bg,m-1)
        set_idx = set(background_label[bg_idx].tolist())
        fg_idx = np.random.randint(0,n_fg)
        set_idx.add(foreground_label[fg_idx].item())
        fg = np.random.randint(0,m)
        fore_idx.append(fg)
        image_list,label = create_mosaic_img(foreground_data,background_data,foreground_label,bg_idx,fg_idx,fg,m)
        mosaic_data.append(image_list)
        mosaic_label.append(label)
        list_set_labels.append(set_idx)
    print("Mosaic Data Created")
    return mosaic_data,mosaic_label,fore_idx


# In[6]:


def Create_Mosaic_data_cifar100(desired_num,m,foreground_label,background_label,foreground_data,background_data,dataset="None"):

        if dataset =="training":
            n_bg = 25000
            n_fg = 25000
        elif dataset == "test":
            n_bg = 5000
            n_fg = 5000

        mosaic_data =[]      # list of mosaic images, each mosaic image is saved as list of 9 images
        fore_idx =[]                   # list of indexes at which foreground image is present in a mosaic image i.e from 0 to 9               
        mosaic_label=[]                # label of mosaic image = foreground class present in that mosaic
        list_set_labels = [] 
        for i in tqdm(range(desired_num)):
            set_idx = set()
            np.random.seed(i)
            bg_idx = np.random.randint(0,n_bg,m-1)
            set_idx = set(background_label[bg_idx].tolist())
            fg_idx = np.random.randint(0,n_fg)
            set_idx.add(foreground_label[fg_idx].item())
            fg = np.random.randint(0,m)
            fore_idx.append(fg)
            image_list,label = create_mosaic_img(foreground_data,background_data,foreground_label,bg_idx,fg_idx,fg,m)
            mosaic_data.append(image_list)
            mosaic_label.append(label)
            list_set_labels.append(set_idx)
        print("Mosaic Data Created")
        return mosaic_data,mosaic_label,fore_idx


# In[7]:


# # Generate Dataset

if which_data == "CIFAR100":


    fg1, fg2, fg3 = 0,1,2
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)


    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=10, shuffle=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=10, shuffle=False)


    classes = tuple(trainset.class_to_idx.keys())


    foreground_classes = {'beaver', 'dolphin', 'otter', 'seal', 'whale',
                         'aquarium_fish', 'flatfish', 'ray', 'shark', 'trout',
                         'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach',
                         'bear', 'leopard', 'lion', 'tiger', 'wolf',
                         'camel', 'cattle', 'chimpanzee', 'elephant','kangaroo',
                         'fox', 'porcupine', 'possum', 'raccoon', 'skunk',
                         'crab', 'lobster', 'snail', 'spider', 'worm',
                         'baby', 'boy', 'girl', 'man', 'woman',
                         'crocodile', 'dinosaur', 'lizard', 'snake', 'turtle',
                         'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'}

    background_classes = {'orchid', 'poppy', 'rose', 'sunflower', 'tulip',
                          'bottle', 'bowl', 'can', 'cup', 'plate',
                          'apple', 'mushroom', 'orange', 'pear', 'sweet_pepper',
                          'clock', 'keyboard', 'lamp', 'telephone', 'television',
                          'bed', 'chair', 'couch', 'table', 'wardrobe',
                          'bridge','castle', 'house', 'road', 'skyscraper',
                          'cloud', 'forest', 'mountain', 'plain', 'sea',
                          'maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree',
                          'bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train',
                          'lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor'}


    print("Total Classes",len(classes),
          "no of foreground classes:", len(foreground_classes),
          "no of background classes:",len(background_classes))


    # print(type(foreground_classes))

    dataiter = iter(trainloader)
    train_bg_data=[]
    train_bg_label=[]
    train_fg_data=[]
    train_fg_label=[]
    batch_size=10

    for i in tqdm(range(5000)):   #5000*batch_size = 50000 data points
        images, labels = dataiter.next()
        for j in range(batch_size):
            if(classes[labels[j]] in background_classes):
                img = images[j].tolist()
                train_bg_data.append(img)
                train_bg_label.append(labels[j])
            else:
                img = images[j].tolist()
                train_fg_data.append(img)
                train_fg_label.append(labels[j])

    train_fg_data = torch.tensor(train_fg_data)
    train_fg_label = torch.tensor(train_fg_label)
    train_bg_data = torch.tensor(train_bg_data)
    train_bg_label = torch.tensor(train_bg_label)
    print("Train Foreground Background Data created")

    train_fg_label = labels_to_index(train_fg_label)


    train_mosaic_data,train_mosaic_label,train_fore_idx = Create_Mosaic_data_cifar100(desired_num,m,train_fg_label,
                                         train_bg_label,train_fg_data,train_bg_data,"training")
elif which_data=="CIFAR10":

    fg1, fg2, fg3 = 0,1,2
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    print(mu)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=10, shuffle=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=10, shuffle=False)


    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    foreground_classes = {'plane', 'car', 'bird'}

    background_classes = {'cat', 'deer', 'dog', 'frog', 'horse','ship', 'truck'}

    # print(type(foreground_classes))

    dataiter = iter(trainloader)
    train_bg_data=[]
    train_bg_label=[]
    train_fg_data=[]
    train_fg_label=[]
    batch_size=10

    for i in tqdm(range(5000)):   #5000*batch_size = 50000 data points
        images, labels = dataiter.next()
        for j in range(batch_size):
            if(classes[labels[j]] in background_classes):
                img = images[j].tolist()
                train_bg_data.append(img)
                train_bg_label.append(labels[j])
            else:
                img = images[j].tolist()
                train_fg_data.append(img)
                train_fg_label.append(labels[j])

    train_fg_data = torch.tensor(train_fg_data)
    train_fg_label = torch.tensor(train_fg_label)
    train_bg_data = torch.tensor(train_bg_data)
    train_bg_label = torch.tensor(train_bg_label)
    print("Train Foreground Background Data created")



    train_mosaic_data,train_mosaic_label,train_fore_idx = Create_Mosaic_data_cifar10(desired_num,m,train_fg_label,
                                         train_bg_label,train_fg_data,train_bg_data,"training")


# In[8]:


msd = MosaicDataset(train_mosaic_data[0:tr], train_mosaic_label[0:tr] , train_fore_idx[0:tr])
train_loader = DataLoader( msd,batch_size= batch ,shuffle=False)

msd1 = MosaicDataset(train_mosaic_data[tr:],train_mosaic_label[tr:] , train_fore_idx[tr:])
test_loader = DataLoader( msd1,batch_size= batch ,shuffle=False)


# In[ ]:


if which_data == "CIFAR100":
    focus = Focus_cnn1()
    focus.load_state_dict(torch.load("focus.pth", map_location=device))
    focus = focus.to(device)
    classification = Classification_cnn1(50)
    classification.load_state_dict(torch.load("classification.pth", map_location=device))
    classification = classification.to(device)
elif which_data =="CIFAR10":
    torch.manual_seed(12)
    focus = Focus_cnn()
    focus.load_state_dict(torch.load("focus.pth", map_location=device))
    focus = focus.to(device)
    torch.manual_seed(12)
    classification = Classification_cnn(3)
    classification.load_state_dict(torch.load("classification.pth", map_location=device))
    classification = classification.to(device)

alpha_list = []
prediction_list1 = []
prediction_list2 = [] 
prediction_list3 = [] 
with torch.no_grad():
    for i,data in enumerate(train_loader):
        images,labels,fore_idx = data
        batch = images.size(0)
        
        patches = images.size(1)
        images,labels = images.to(device),labels.to(device)
        alphas = torch.softmax(focus(images),dim=1)
        
        
        # argmax based
        indexes  = torch.argmax(alphas,dim=1).cpu().numpy()
        outputs_1 = F.softmax(classification(images[np.arange(batch),indexes,:]),dim=1)
        prediction_list2.append(outputs_1[np.arange(len(labels)),labels].cpu().numpy())
        alpha_list.append(alphas[np.arange(len(fore_idx)),fore_idx].cpu().numpy())
        
        #averaging output
        images1 = images.reshape(batch*patches,3,32,32)
        classification_outputs = F.softmax(classification(images1),dim=1)
        classification_outputs = classification_outputs.reshape(batch,patches,50)
        outputs_3 = torch.sum(alphas[:,:,None]*classification_outputs,dim=1)
        prediction_list3.append(outputs_3[np.arange(len(labels)),labels].cpu().numpy())
        
        # averaging input
        images = torch.sum(alphas[:,:,None,None,None]*images,dim=1)
        outputs_2 = torch.softmax(classification(images),dim=1)
        
        prediction_list1.append(outputs_2[np.arange(len(labels)),labels].cpu().numpy())
alpha_list = np.hstack(alpha_list)
prediction_list = np.hstack(prediction_list1) #average input
prediction_list2 = np.hstack(prediction_list2) #argmax
prediction_list3 = np.hstack(prediction_list3) #average output
alpha_list.shape,prediction_list.shape,prediction_list2.shape,prediction_list3.shape


# In[ ]:


if which_data =="CIFAR10":
    num = 100
elif which_data=="CIFAR100":
    num= 500
    

    
plt.figure(figsize=(6,6))
im = plt.hist2d(alpha_list,prediction_list,(5,5))
ax = sns.heatmap(np.round(im[0].transpose()/num,1),vmin=5,vmax=70,annot=True,fmt="g",cmap=sns.color_palette("coolwarm"),
                 yticklabels=[0.2,0.4,0.6,0.8,1.],
                 xticklabels=[0.2,0.4,0.6,0.8,1],annot_kws={"size":18},cbar=False)
ax.invert_yaxis()

plt.xlabel(r"$\bf a_z$",fontweight="bold",fontsize=20)
plt.ylabel(r"$\bf s^{SA}_y$",fontweight="bold",fontsize=20) # change xlabel based on algo
plt.xticks([1,2,3,4,5],weight="bold",fontsize=18)
plt.yticks([1,2,3,4,5],weight="bold", va="top",fontsize=18)

plt.savefig("train_sa_m_20_50k.pdf",bbox_inches='tight')
#plt.savefig("ML_Loss_M_5_10K.pdf",bbox_inches='tight')


# In[ ]:


if which_data == "CIFAR100":
    focus = Focus_cnn1()
    focus.load_state_dict(torch.load("focus.pth", map_location=device))
    focus = focus.to(device)
    classification = Classification_cnn1(50)
    classification.load_state_dict(torch.load("classification.pth", map_location=device))
    classification = classification.to(device)
elif which_data =="CIFAR10":
    torch.manual_seed(12)
    focus = Focus_cnn()
    focus.load_state_dict(torch.load("focus.pth", map_location=device))
    focus = focus.to(device)
    torch.manual_seed(12)
    classification = Classification_cnn(3)
    classification.load_state_dict(torch.load("classification.pth", map_location=device))
    classification = classification.to(device)

alpha_list = []
prediction_list1 = []
prediction_list2 = [] 
prediction_list3 = [] 
with torch.no_grad():
    for i,data in enumerate(test_loader):
        images,labels,fore_idx = data
        batch = images.size(0)
        
        patches = images.size(1)
        images,labels = images.to(device),labels.to(device)
        alphas = torch.softmax(focus(images),dim=1)
        
        
        # argmax based
        indexes  = torch.argmax(alphas,dim=1).cpu().numpy()
        outputs_1 = F.softmax(classification(images[np.arange(batch),indexes,:]),dim=1)
        prediction_list2.append(outputs_1[np.arange(len(labels)),labels].cpu().numpy())
        alpha_list.append(alphas[np.arange(len(fore_idx)),fore_idx].cpu().numpy())
        
        #averaging output
        images1 = images.reshape(batch*patches,3,32,32)
        classification_outputs = F.softmax(classification(images1),dim=1)
        classification_outputs = classification_outputs.reshape(batch,patches,50)
        outputs_3 = torch.sum(alphas[:,:,None]*classification_outputs,dim=1)
        prediction_list3.append(outputs_3[np.arange(len(labels)),labels].cpu().numpy())
        
        # averaging input
        images = torch.sum(alphas[:,:,None,None,None]*images,dim=1)
        outputs_2 = torch.softmax(classification(images),dim=1)
        
        prediction_list1.append(outputs_2[np.arange(len(labels)),labels].cpu().numpy())
alpha_list = np.hstack(alpha_list)
prediction_list = np.hstack(prediction_list1) #average input
prediction_list2 = np.hstack(prediction_list2) #argmax
prediction_list3 = np.hstack(prediction_list3) #average output
alpha_list.shape,prediction_list.shape,prediction_list2.shape,prediction_list3.shape


# In[ ]:


if which_data =="CIFAR10":
    num = 100
elif which_data=="CIFAR100":
    num= 100
    

    
plt.figure(figsize=(6,6))
im = plt.hist2d(alpha_list,prediction_list,(5,5))
ax = sns.heatmap(np.round(im[0].transpose()/num,1),vmin=5,vmax=70,annot=True,fmt="g",cmap=sns.color_palette("coolwarm"),
                 yticklabels=[0.2,0.4,0.6,0.8,1.],
                 xticklabels=[0.2,0.4,0.6,0.8,1],annot_kws={"size":18},cbar=False)
ax.invert_yaxis()

plt.xlabel(r"$\bf a_z$",fontweight="bold",fontsize=20)
plt.ylabel(r"$\bf s^{SA}_y$",fontweight="bold",fontsize=20) # change algo based on algo
plt.xticks([1,2,3,4,5],weight="bold",fontsize=18)
plt.yticks([1,2,3,4,5],weight="bold", va="top",fontsize=18)

plt.savefig("test_sa_m_20_50k.pdf",bbox_inches='tight')
#plt.savefig("ML_Loss_M_5_10K.pdf",bbox_inches='tight')


# In[ ]:




