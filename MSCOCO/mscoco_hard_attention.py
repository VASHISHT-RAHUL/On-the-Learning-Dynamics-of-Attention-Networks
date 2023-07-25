

# Before Running the code please do the following 
# Clone the git repository https://github.com/VASHISHT-RAHUL/pycocotools.git  
# Clone the git repository https://github.com/VASHISHT-RAHUL/coco_eval.git'
# install pickle5 and pycocotools

import sys




import numpy as np
import pandas as pd
import time
import math
import random
import tqdm
import os
import sys
import glob
import re
import unicodedata
from tqdm import tqdm
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import pickle
    
import itertools
import time
import json

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import models



import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
from torch.nn import functional as F

from pycocotools.coco import COCO
from coco_eval.pycocoevalcap.eval import COCOEvalCap


from dictionary_py import Vocabulary,EOS_token,PAD_token,SOS_token,UNK_token



if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
print('Device in Use: ',device)
print('Device Properties: ',torch.cuda.get_device_properties(device))



#Utility functions

# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def target_tensor_to_caption(target):
    gnd_trh = []
    lend = target.size()[1]
    for i in range(lend):
        tmp = ' '.join(voc.index2word[x.item()] for x in targets[:,i])
        gnd_trh.append(tmp)
    return gnd_trh

def maskNLLLoss(inp, target, mask):

    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(inp.squeeze(0), 1, target.view(-1, 1)).squeeze(1).float())
    loss = crossEntropy.masked_select(mask) #.mean()
    #print("Flag 2",loss.shape,mask.shape)
    loss = loss.to(device)
    return loss, nTotal.item()



train_image_path = os.path.join('.../train2014') # path to train images
val_image_path = os.path.join('.../val2014') # path to validation images


annotation_path = os.path.join('.../annotations') # annotation file path
train_annotation_file = os.path.join(annotation_path,'captions_train2014.json')  
val_annotation_file = os.path.join(annotation_path,'captions_val2014.json')


#Add image augmentation later
data_transform = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])

coco_train = dset.CocoCaptions(root=train_image_path,annFile=train_annotation_file,transform=data_transform)
coco_val = dset.CocoCaptions(root=val_image_path,annFile=val_annotation_file,transform=data_transform) 
img,target = coco_train[200]






voc = Vocabulary('COCO_TRAIN')
voc.load() # load saved vocabulary
voc.trim(min_count=5) # remove words having frequency less than min_count
print('Vocabulary size :',voc.num_words)




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
        lbl = normalizeString(random.choice(target))
        label = []
        for s in lbl.split(' '):
            if s in list(voc.word2index.keys()):
                label.append(voc.word2index[s])
            else:
                label.append(UNK_token)
        label = label +[EOS_token]
        
        return img, label,ide

train_dset = COCO14Dataset(coco_train,voc,transforms=data_transform)
val_dset = COCO14Dataset(coco_val,voc)




#Original Model Implementation Details
  # Encoder - VGG19 14×14×512 feature map of the fourth convolutional layer before max pooling. 196 × 512 
  # mini-batch - 64
  # stopping criterion - early stopping on BLEU score
  # model selection - BLEU on our validation set
  # vocabulary size - 10,000

#We observed a breakdown in correlation between the validation set log-likelihood and BLEU in the later stages of 
#training during our experiments

#Hyperparameters
batch_size = 100
val_batch_size = 64
feat_size = 512
feat_len = 196
embedding_size = 512
hidden_size = 512
attn_size = 256
output_size = voc.num_words
rnn_dropout = 0.5
teacher_forcing_ratio = 0.7





def collate_fn(batch):
    data = [item[0] for item in batch]
    images = torch.stack(data,0)
    
    ides = torch.tensor([item[2] for item in batch])
    
    label = [item[1] for item in batch]
    max_target_len = max([len(indexes) for indexes in label])
    padList = list(itertools.zip_longest(*label, fillvalue = 0))
    lengths = torch.tensor([len(p) for p in label])
    padVar = torch.LongTensor(padList)
    
    m = []
    for i, seq in enumerate(padVar):
        #m.append([])
        tmp = []
        for token in seq:
            if token == 0:
                tmp.append(int(0))
            else:
                tmp.append(1)
        m.append(tmp)
    m = torch.tensor(m)
    
    return images, padVar, m, max_target_len, ides

train_loader=DataLoader(train_dset,batch_size = batch_size, num_workers = 2,shuffle = True,
                    collate_fn = collate_fn, drop_last=True)

val_loader = DataLoader(val_dset,batch_size = val_batch_size, num_workers = 2,shuffle = False,collate_fn = collate_fn,
                     drop_last=False)





dataiter = iter(train_loader)
features, targets, mask, max_length,ides= dataiter.next()
features.size(),targets.size(),mask.size(),ides.size()




for i in targets[:,0]:
    print(voc.index2word[i.item()])




# Encoder Model
class Encoder(nn.Module):
    def __init__(self,batch_size):
        super(Encoder,self).__init__()
        base_model = models.vgg19(pretrained=True)
        layers_to_use = list(base_model.features.children())[:29]
        self.model = nn.Sequential(*layers_to_use)
        
    def forward(self,image_batch):
        batch_size = image_batch.size()[0]
        output = self.model(image_batch).view(batch_size,512,-1)
        output = output.permute(0,2,1)
        return output





class DeterministicSpatialAttention(nn.Module):
    def __init__(self,hidden_size, feat_size, bottleneck_size):
        super(DeterministicSpatialAttention,self).__init__()
        '''
        Spatial Attention module. It depends on previous hidden memory in the decoder(of shape hidden_size),
        feature at the source side ( of shape(196,feat_size) ).  
        at(s) = align(ht,hs)
              = exp(score(ht,hs)) / Sum(exp(score(ht,hs')))  
        where
        score(ht,hs) = ht.t * hs                         (dot)
                     = ht.t * Wa * hs                  (general)
                     = va.t * tanh(Wa[ht;hs])           (concat)  
        Here we have used concat formulae.
        Argumets:
          hidden_size : hidden memory size of decoder.
          feat_size : feature size of each grid (annotation vector) at encoder side.
          bottleneck_size : intermediate size.
        '''
        self.hidden_size = hidden_size
        self.feat_size = feat_size
        self.bottleneck_size = bottleneck_size
        
        self.decoder_projection = nn.Linear(hidden_size,bottleneck_size)
        self.encoder_projection = nn.Linear(feat_size, bottleneck_size)
        self.final_projection = nn.Linear(bottleneck_size,1)
     
    def forward(self,hidden,feats):
        '''
        shape of hidden (hidden_size)
        shape of feats (196,feat_size)
        '''
        Wh = self.decoder_projection(hidden)  # (256)
        Uv = self.encoder_projection(feats)   # (60,256)
        #print(' Wh(hidden to bottleneck)  Uv(Feats to bottleneck)',Wh.size(),Uv.size())
        Wh = Wh.unsqueeze(1).expand_as(Uv)
        #print('Wh size  : ',Wh.size())
        energies = self.final_projection(torch.tanh(Wh+Uv))
        #print('energies : ',Uv.size())
        weights = F.softmax(energies, dim=1)
        
        weighted_feats = feats *weights.expand_as(feats)
        attn_feats = weighted_feats.sum(dim=1)
        return attn_feats,weights



class StochasticSpatialAttention(nn.Module):
    def __init__(self,hidden_size, feat_size, bottleneck_size):
        super(StochasticSpatialAttention,self).__init__()
        '''
        Spatial Attention module. It depends on previous hidden memory in the decoder(of shape hidden_size),
        feature at the source side ( of shape(196,feat_size) ).  
        at(s) = align(ht,hs)
              = exp(score(ht,hs)) / Sum(exp(score(ht,hs')))  
        where
        score(ht,hs) = ht.t * hs                         (dot)
                     = ht.t * Wa * hs                  (general)
                     = va.t * tanh(Wa[ht;hs])           (concat)  
        Here we have used concat formulae.
        Argumets:
          hidden_size : hidden memory size of decoder.
          feat_size : feature size of each grid (annotation vector) at encoder side.
          bottleneck_size : intermediate size.
        '''
        self.hidden_size = hidden_size
        self.feat_size = feat_size
        self.bottleneck_size = bottleneck_size
        
        self.decoder_projection = nn.Linear(hidden_size,bottleneck_size)
        self.encoder_projection = nn.Linear(feat_size, bottleneck_size)
        self.final_projection = nn.Linear(bottleneck_size,1)
        #self.attn_feats = torch.tensor([0])
        #self.attn_weights = torch.tensor([0])
        
     
    def forward(self,hidden,feats):
        '''
        shape of hidden (hidden_size)
        shape of feats (196,feat_size)
        '''
        Wh = self.decoder_projection(hidden)  # (256)
        Uv = self.encoder_projection(feats)   # (60,256)
        #print(' Wh(hidden to bottleneck)  Uv(Feats to bottleneck)',Wh.size(),Uv.size())
        Wh = Wh.unsqueeze(1).expand_as(Uv)
        #print('Wh size  : ',Wh.size())
        energies = self.final_projection(torch.tanh(Wh+Uv))
        #print('energies : ',Uv.size())
        weights = F.softmax(energies, dim=1)
        
        #stochastic attention
        if self.training:
            m = torch.distributions.categorical.Categorical(logits = energies[:,:,0])
            
            sampled_weights = m.sample()
            
            #m.probs = weights[:,:,0] 
            
            
            log_action_prob = m.log_prob(sampled_weights)
            
            attn_weights = torch.nn.functional.one_hot(sampled_weights,num_classes=weights.shape[1]).float()
            attn_weights.requires_grad = True
            weighted_feats = feats * attn_weights[:,:,None].expand_as(feats)
            attn_feats = weighted_feats.sum(dim=1)
        else:
            #self.attn_weights = weights
            log_action_prob = None
            weighted_feats = feats *weights.expand_as(feats)
            attn_feats = weighted_feats.sum(dim=1)
            

        return attn_feats,weights,log_action_prob




class Decoder(nn.Module):
    
    def __init__(self, feat_size, feat_len, embedding_size, hidden_size, attn_size, output_size, rnn_dropout,
                num_layers = 1, num_directions = 1,attn_type="deterministic"):
        super(Decoder, self).__init__()
        '''
        Decoder, Basically a language model.
        
        Arguments:
        hidden_size : hidden memory size of LSTM/GRU
        output_size : output size. Its same as the vocabulary size.
        n_layers : 
        
        '''

        # Keep for reference
        self.feat_size = feat_size
        self.feat_len = feat_len
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.attn_size = attn_size
        self.output_size = output_size
        self.rnn_dropout = rnn_dropout
        
        self.num_layers = num_layers
        self.num_directions = num_directions
        self.attn_type = attn_type 

        # Define layers
        self.embedding = nn.Embedding(self.output_size, self.embedding_size)
        if self.attn_type == "deterministic":
            self.attention = DeterministicSpatialAttention(hidden_size = self.num_directions*self.hidden_size,
                                          feat_size=self.feat_size,
                                          bottleneck_size=self.attn_size)
        elif self.attn_type == "stochastic":
            self.attention = StochasticSpatialAttention(hidden_size = self.num_directions*self.hidden_size,
                                          feat_size=self.feat_size,
                                          bottleneck_size=self.attn_size)
        
            
        
        self.rnn = nn.LSTM(self.embedding_size+self.feat_size, self.hidden_size,
                           self.num_layers, dropout=self.rnn_dropout,batch_first=False, 
                          bidirectional=True if self.num_directions ==2 else False)
        
        self.out = nn.Linear(self.num_directions*self.hidden_size, self.output_size)

    def get_last_hidden(self, hidden):
        
        last_hidden = hidden[0] if isinstance(hidden,tuple) else hidden
        last_hidden = last_hidden.view(self.num_layers, self.num_directions,
                                       last_hidden.size(1),last_hidden.size(2))
        last_hidden = last_hidden.transpose(2,1).contiguous()
        last_hidden = last_hidden.view(self.num_layers,last_hidden.size(1),
                                       self.num_directions*last_hidden.size(3))
        last_hidden = last_hidden[-1]
        return last_hidden
    
    
    def forward(self, inputs, hidden, feats):
        # Note: we run this one step (word) at a time
        # Get embedding of current input word
        
        # inputs -  (1,batch)
        # hidden - (num_layers * num_directions, batch, hidden_size)
        # feats - (batch,attention_length,annotation_vector_size) (32,196,512)
        
        #print('input  hidden  feats :',inputs.size(),hidden[0].size(),feats.size())
        
        embedded = self.embedding(inputs)
        
        last_hidden = hidden[0]
        #print('embedded size :',embedded.size())
        
        feats, attn_weights,log_action_prob = self.attention(last_hidden.squeeze(0),feats)

        input_combined = torch.cat((embedded,feats.unsqueeze(0)),dim=2)
        #print('input combined :',input_combined.size())

        output, hidden = self.rnn(input_combined, hidden)

        output = output.squeeze(0)
        output = self.out(output)
        output = F.softmax(output, dim = 1)
        
        return output, hidden, attn_weights,log_action_prob




# instantiate encoder decoder model
enc = Encoder(batch_size).to(device)
dec = Decoder(feat_size,feat_len,embedding_size,hidden_size,attn_size,voc.num_words,rnn_dropout,attn_type="stochastic").to(device)




class ShowAttendTell(nn.Module):
    
    def __init__(self,encoder,decoder,vocabulary,teacher_forcing_ratio, batch_size=batch_size,):
        super(ShowAttendTell,self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.voc = vocabulary
        self.batch_size = batch_size
        self.enc_optimizer = optim.Adam(self.encoder.parameters(),lr=1e-5)
        self.dec_optimizer = optim.Adam(self.decoder.parameters(),lr=1e-3)
        
        #self.scheduler_enc = torch.optim.lr_scheduler.ExponentialLR(self.enc_optimizer, gamma=0.9)
        #self.scheduler_dec = torch.optim.lr_scheduler.ExponentialLR(self.dec_optimizer, gamma=0.9)
        
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.print_every = 200
        self.loss_ = 0 
        self.b = torch.tensor([0],device=device)#torch.zeros(200,device=device)#
        
    def update_hyperparam(self,encoder_lr,decoder_lr,teacher_forcing_ratio):
        self.encoder_lr = encoder_lr
        self.decoder_lr = decoder_lr
        self.enc_optimizer.param_groups[0]['lr'] = self.encoder_lr #= optim.Adam(self.encoder.parameters(),lr=self.encoder_lr)
        self.dec_optimizer.param_groups[0]['lr'] = self.decoder_lr #= optim.Adam(self.decoder.parameters(),lr=self.decoder_lr)
        self.teacher_forcing_ratio = teacher_forcing_ratio
 
        
    def load(self,encoder_path = 'Save/VGG_encoder_10.pt',decoder_path='Save/VGG_decoder_10.pt'):
        checkpoint = torch.load(encoder_path)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        self.enc_optimizer.load_state_dict(checkpoint['encoder_optimizer_state_dict'])
        self.dec_optimizer.load_state_dict(checkpoint['decoder_optimizer_state_dict'])
        self.b = checkpoint['b']
        
        
    def train_epoch(self,dataloader,clip=5):
        
        total_loss = 0
        start_iteration = 1
        print_loss = 0
        iteration = 1
        for data in dataloader:
            features, targets, mask, max_length,_ = data
            use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False
        

            loss = self.train_iter(features,targets,mask,max_length,use_teacher_forcing,clip)
            print_loss += loss
            total_loss += loss

        # Print progress
            if iteration % self.print_every == 0:
                print_loss_avg = print_loss / self.print_every
                print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".
                format(iteration, iteration / len(dataloader) * 100, print_loss_avg))
                print_loss = 0
            
            iteration += 1
        #self.scheduler_enc.step() 
        #self.scheduler_dec.step()
        return total_loss/len(dataloader)
            
        
    def train_iter(self,input_variable, target_variable, mask,max_target_len,use_teacher_forcing,clip=5):
        
        self.enc_optimizer.zero_grad()
        self.dec_optimizer.zero_grad()
        
        loss = 0
        print_losses = []
        n_totals = 0
        loss1 = 0
        entropy_ = 0 
        
        input_variable = input_variable.to(device)
        target_variable = target_variable.to(device)
        mask = mask.byte().to(device)
        
        
        
        enc_output = self.encoder(input_variable)
        decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
        decoder_input = decoder_input.to(device)
        decoder_hidden = (torch.zeros(1, batch_size, self.decoder.hidden_size).to(device),
                  torch.zeros(1, batch_size, self.decoder.hidden_size).to(device))
        
        # Forward batch of sequences through decoder one time step at a time
        if use_teacher_forcing:
            for t in range(max_target_len):
                decoder_output, decoder_hidden,attn_weights,log_action_prob = self.decoder(decoder_input, decoder_hidden,enc_output)
                decoder_input = target_variable[t].view(1, -1)
                mask_loss, nTotal = maskNLLLoss(decoder_output.unsqueeze(0), target_variable[t], mask[t])
    
                
                entropy_ += (torch.sum(-1.0* attn_weights * torch.log(attn_weights),dim=1)).mean()
                log_action_prob = log_action_prob.masked_select(mask[t])
                
                #loss1 += (log_action_prob*mask_loss.detach()).mean()
                
                #self.b[t] = (0.9*self.b[t]) + (0.1*mask_loss.detach().mean())
                
                #loss1 += log_action_prob.mean() #*( mask_loss.detach().mean() + self.b )
            
                
                loss += mask_loss.mean()
        
                
                print_losses.append(mask_loss.mean().item() * nTotal)
                n_totals += nTotal
        else:
            for t in range(max_target_len):
                #print('decoder hidden in sampling :',decoder_hidden.size())
                decoder_output, decoder_hidden,attn_weights,log_action_prob  = self.decoder(decoder_input, decoder_hidden,enc_output )
                # No teacher forcing: next input is decoder's own current output
                
                
                _, topi = decoder_output.squeeze(0).topk(1)
                decoder_input = torch.LongTensor([[topi[i][0] for i in range(self.batch_size)]])
                decoder_input = decoder_input.to(device)
                
                
                entropy_ += torch.sum(-1.0* attn_weights * torch.log(attn_weights),dim=1).mean()
                
                mask_loss, nTotal = maskNLLLoss(decoder_output.unsqueeze(0), target_variable[t], mask[t])
                log_action_prob = log_action_prob.masked_select(mask[t])
                
                

                
                loss += mask_loss.mean()
                print_losses.append(mask_loss.mean().item() * nTotal)
                n_totals += nTotal
        
        
        

        
        
        self.b = ( 0.9*self.b) + 0.1*(loss.detach())
        loss2 = loss + 0.5 *(log_action_prob.mean())*(loss.detach()  + self.b ) #+ 0.005*(entropy_)
        loss2.backward()


        # Clip gradients: gradients are modified in place
        _ = nn.utils.clip_grad_norm_(self.encoder.parameters(), clip)
        _ = nn.utils.clip_grad_norm_(self.decoder.parameters(), clip)

        # Adjust model weights
        self.enc_optimizer.step()
        self.dec_optimizer.step()

        return sum(print_losses) / n_totals
            
        
    def Greedy_Decoding(self,features,max_length=15):
        enc_output = self.encoder(features)
        batch_size = features.size()[0]
        decoder_hidden = (torch.zeros(1, batch_size, self.decoder.hidden_size).to(device),
                          torch.zeros(1, batch_size, self.decoder.hidden_size).to(device))
        
        decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]]).to(device)
        #print('Initial size',decoder_input.size())
        caption = []
        attn_weights = []
        for _ in range(max_length):
            decoder_output, decoder_hidden, attn_values,_ = self.decoder(decoder_input, decoder_hidden, enc_output)
            #print(attn_values.shape)

            #print('decoder hidden size :',decoder_hidden.size())
            #print('decoder output size :',decoder_output.size())
            _, topi = decoder_output.topk(1)
            #print(topi)
            #print('topi size',topi.size())
            decoder_input = topi.permute(1,0).to(device)
            caption.append(topi.squeeze(1).cpu())
            attn_weights.append(attn_values[:,:,0].cpu())
        caption = torch.stack(caption,0).permute(1,0)
        attn_weights= torch.stack(attn_weights,0)
        #print(attn_weights.size())

        caps_text = []
        for dta in caption:
            tmp = []
            for token in dta:
                if token.item() not in self.voc.index2word.keys() or token.item()==2:
                    pass
                else:
                    tmp.append(self.voc.index2word[token.item()])
            tmp = ' '.join(x for x in tmp)
            caps_text.append(tmp)
        return caption,caps_text,attn_weights




class Evaluator:
    
    def __init__(self,arch_name,prediction_filepath,reference_file,dataloader):
        self.arch_name = arch_name
        self.prediction_filepath = prediction_filepath
        self.dataloader = dataloader
        self.coco = COCO(reference_file)
        self.scores = {}
        self.bleu4 = 0
    
    def prediction_list(self,model):
        result = []
        ide_list = []
        caption_list =[]
        model.eval()
        with torch.no_grad():
            for data in self.dataloader:
                features, targets, mask, max_length,ides= data
                cap,cap_txt,_ = model.Greedy_Decoding(features.to(device))
                ide_list += list(ides.cpu().numpy())
                caption_list += cap_txt
        for a in zip(ide_list,caption_list):
            result.append({'image_id':a[0].item(),'caption':a[1].strip()})      
        return result
    
    def prediction_file_generation(self,result,prediction_filename):
    
        self.predicted_file = os.path.join(self.prediction_filepath,prediction_filename) 
        with open(self.predicted_file, 'w') as fp:
            json.dump(result,fp)
            
    def evaluate(self,model,epoch):
        model.eval()
        prediction_filename = self.arch_name+str(epoch)+'.json'
        result = self.prediction_list(model)
        self.prediction_file_generation(result,prediction_filename)
        
        cocoRes = self.coco.loadRes(self.predicted_file)
        cocoEval = COCOEvalCap(self.coco,cocoRes)
        scores = cocoEval.evaluate()
        self.scores[epoch] = scores
        print(scores[0])
        if scores[0][1][3] > self.bleu4:
            self.bleu4 = scores[0][1][3]
        self.save_model(model,epoch)
        return scores
    def save_model(self,model,epoch):
        print('Better result saving models....')
        encoder_filename = 'encoder_decoder_'+str(epoch)+'.pt'
        #decoder_filename = 'decoder_'+str(epoch)+'.pt'
        model.train()
        torch.save({'encoder_state_dict':model.encoder.state_dict(),
                   'decoder_state_dict':model.decoder.state_dict(),
                   'encoder_optimizer_state_dict':model.enc_optimizer.state_dict(),
                   'decoder_optimizer_state_dict':model.dec_optimizer.state_dict(),
                   'b':model.b},encoder_filename)
        #torch.save(model.decoder,decoder_filename)
        
        
        print("model saved")
        #torch.save(model.encoder,encoder_filename)
        #torch.save(model.decoder,decoder_filename)



model = ShowAttendTell(enc,dec,voc,teacher_forcing_ratio=1,batch_size=batch_size)

#------- For Hybrid model Load soft attention model here and reinitialize the optimizers and baseline-------------------------------



#model.load('/kaggle/input/load-model/encoder_decoder_20.pt','/kaggle/input/load-model/encoder_decoder_20.pt')
#model.enc_optimizer = optim.Adam(model.encoder.parameters(),lr=1e-7)
#model.dec_optimizer = optim.Adam(model.decoder.parameters(),lr=1e-5)
#model.b = torch.tensor([0],device=device) #torch.zeros(200,device=device) #torch.tensor([0],device=device)


#-----------------------------------------------------------------------------------------------------------------
val_evaluator = Evaluator('epoch_',"",val_annotation_file,val_loader)





model.train()
k=20
# model.update_hyperparam(1e-5,1e-3,1) # use to change the hyperparameters of optimizer
for epoch in tqdm(range(5)):
    #if (epoch+1) == 4:
        #model.update_hyperparam(1e-6,1e-5,1)
    loss = model.train_epoch(train_loader)
    print(' Epoch :',epoch+k+1,' Loss :',loss)
    scores = val_evaluator.evaluate(model,epoch+k+1)
    print(scores)
    model.train()


