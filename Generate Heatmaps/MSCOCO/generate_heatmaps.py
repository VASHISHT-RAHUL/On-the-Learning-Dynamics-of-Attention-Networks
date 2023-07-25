
import seaborn as sns

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')

import torch
import torchvision
import torch.nn as nn

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm as tqdm
import unicodedata
import pycocotools

import re
import random 
import itertools
import time
import os 
import sys
from pycocotools.coco import COCO
import urllib
import zipfile
import json
from skimage.transform.pyramids import pyramid_expand
import skimage

from models import Encoder, Decoder, ShowAttendTell,DeterministicSpatialAttention,StochasticSpatialAttention


from dictionary import Vocabulary,EOS_token,PAD_token,SOS_token,UNK_token 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#Hyperparameters
batch_size = 16
val_batch_size = 16
feat_size = 512
feat_len = 196
embedding_size = 512
hidden_size = 512
attn_size = 256
#output_size = voc.num_words
rnn_dropout = 0.5
teacher_forcing_ratio = 0.5
     


# In[ ]:



voc = Vocabulary('COCO_VAL') #"TRAIN"
voc.load()
voc.trim(min_count=5) # remove words having frequency less than min_count
print('Vocabulary size :',voc.num_words)


voc.word2index['women']



enc = Encoder(batch_size).to(device)
torch.manual_seed(1234)
dec = Decoder(feat_size,feat_len,embedding_size,hidden_size,attn_size,voc.num_words,rnn_dropout,attn_type="stochastic").to(device)



model = ShowAttendTell(enc,dec,voc,teacher_forcing_ratio=teacher_forcing_ratio,batch_size=batch_size)
#model.load('Saved_Models/encoder_decoder_25.pt','encoder_decoder_25.pt')





data_transform = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])#transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     #std=[0.229, 0.224, 0.225])])

coco_val_detection = dset.CocoDetection(root="images/val2014",annFile="annotations/instances_val2014.json")#,transform=data_transform)
coco_val = dset.CocoDetection(root="images/val2014",annFile="annotations/instances_val2014.json",transform=data_transform)







category_in_image = {}
count = 0 
for data in tqdm(coco_val_detection):
    _,instance = data
    cat_id = []
    image_id = []
  
    for i in range(len(instance)):
        cat_id.append(instance[i]['category_id'])
        image_id.append(instance[i]['image_id'])
    try:
        category_in_image[image_id[0]] = list(set(cat_id))
    except:
        count = count+1
category_in_image
     




def bbox_transform_coco2cv(b,wratio,hratio):
    b = [b[0]*wratio,b[1]*hratio,b[2]*wratio,b[3]*hratio]
    temp = [b[0],b[1]]
    temp.append(b[0]+b[2])
    temp.append(b[1]+b[3])
    for i in range(len(temp)):
        temp[i] = int(temp[i])
    temp2 = [temp[1],temp[0],temp[3],temp[2]]
    return np.array(temp).astype(int)#, temp2 #(x1,y1,x2,y2)
     




bbox_dict = {}
for data in tqdm(coco_val_detection):
    image,instance= data
    original_shape = image.size
    wratio = (224*1.0)/original_shape[1]
    hratio = (224*1.0)/original_shape[0]
    for i in range(len(instance)):
        key_value = (instance[i]['image_id'],instance[i]['category_id'])
        if key_value in bbox_dict.keys():
            val = []
            for j in range(len(bbox_dict[key_value])):
                val.append(bbox_dict[key_value][j])
            val.append(bbox_transform_coco2cv(instance[i]['bbox'],hratio,wratio))
            bbox_dict[key_value] = val 
        else:
            bbox_dict[key_value] = [bbox_transform_coco2cv(instance[i]['bbox'],hratio,wratio)]







class COCO14Dataset(Dataset):
    def __init__(self,coco,voc,transforms=None):
        self.coco = coco
        self.voc = voc
        self.transforms = transforms
    def __len__(self):
        return len(self.coco)
    def __getitem__(self,idx):
        img,target = self.coco[idx]
        ide = self.coco.ids[idx]

        
        return img,ide

val_dset = COCO14Dataset(coco_val,voc)



def collate_fn(batch):
    data = [item[0] for item in batch]
    images = torch.stack(data,0)
    
    ides = torch.tensor([item[1] for item in batch])
    
    return images, ides



val_loader = DataLoader(val_dset,batch_size = val_batch_size, 
                        num_workers = 8,shuffle = False,
                        collate_fn = collate_fn,
                        drop_last=False)




data,ide = iter(val_loader).next()


model.load('Saved_Models/encoder_decoder_25_3.pt','Saved_Models/encoder_decoder_25_3.pt')  # Load Model here for which you want to generate the heatmaps


word_association=pd.read_csv("coco_word_association.csv")
word_association.fillna(value="dummy",inplace=True)


word_association_dict = {}
for i in tqdm(range(word_association.shape[0])):
    value = list(word_association.loc[i])
    try:
        idx = value.index("dummy")
        value = value[:idx]
        word_association_dict[value[0]] = value[1:]
    except:
        word_association_dict[value[0]] = value[1:]


count = 0 
s_y = [] #probabilities corresponding to true label
a_z = [] # attention weight given to true bbounding box
model.eval()
with torch.no_grad():
    for data in tqdm(val_loader):
        features, ides= data
        
        enc_output = model.encoder(features.to(device))
        batch_size = features.size()[0]
        decoder_hidden = (torch.zeros(1, batch_size, model.decoder.hidden_size).to(device),
                          torch.zeros(1, batch_size, model.decoder.hidden_size).to(device))
        
        decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]]).to(device)
        
        
        pred_probs = []
        attention_weights = []
        
        for _ in range(15):
            decoder_output, decoder_hidden, sampled_attn_values,attn_values = model.decoder(decoder_input, 
                                                                                           decoder_hidden, 
                                                                                           enc_output)

            pred_probs.append(decoder_output)
            attention_weights.append(attn_values[:,:,0])
            _, topi = decoder_output.topk(1)
            decoder_input = topi.permute(1,0).to(device)
        pred_probs = torch.stack(pred_probs)
        attention_weights = torch.stack(attention_weights)
        
        
   
        for idx,image_id in enumerate(ides):
            try:
                for word_category in category_in_image[image_id.item()]:
                    bbox_data = bbox_dict[(image_id.item(),word_category)]
                    if len(bbox_data) ==1:
                        word_categories = word_association.loc[word_association.Category_id == word_category].values[0]
                        word_categories = list(word_categories[1:])
                    
                        dummy_index = word_categories.index("dummy")
                        word_categories = word_categories[:dummy_index]
                        voc_indexes = [voc.word2index[x] for x in word_categories]
                    

                        
                        values,max_indexes = pred_probs[:,idx,voc_indexes].sum(dim=1).max(dim=0)    

            
                        s_y.append(values.detach().cpu().numpy())
                        
                        word_category_attn_wts = attention_weights[max_indexes,idx,:]

                        upsampled_wts = skimage.transform.resize(word_category_attn_wts.detach().cpu().numpy()
                                                                 ,[224, 224])/255
                        
                        temp= sum(sum(upsampled_wts[bbox_data[0][1]:bbox_data[0][3],
                                                     bbox_data[0][0]:bbox_data[0][2]]))
             
                        
                        a_z.append(temp)
            except:
                count = count +1



# Generate heatmap
file_name = "train_sa" # file name for heatmap
model_name = "sa" # 
plt.figure(figsize=(6,6))

im = plt.hist2d(np.array(a_z),s_y,[[0,0.2,0.3,1],[0,0.2,0.3,1]])
num = im[0].sum()/100
ax = sns.heatmap(np.round(im[0].transpose()/num,2),vmin=5,vmax=70,annot=True,fmt="g",cmap=sns.color_palette("coolwarm"),
                 yticklabels=[0,0.2,0.3,1.],
                 xticklabels=[0,0.2,0.3,1],annot_kws={"size":18},cbar=False)
ax.invert_yaxis()

plt.xlabel(r"$\bf a_z$",fontweight="bold",fontsize=20)
plt.ylabel(r"$\bf s^{" +model_name +"}_y$",fontweight="bold",fontsize=20) # change algo based on algo
plt.xticks([0,1,2,3],weight="bold",fontsize=18)
plt.yticks([0,1,2,3],weight="bold", va="top",fontsize=18)
plt.savefig(file_name+".pdf")







