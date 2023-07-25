import numpy as np
import pandas as pd
from tqdm import tqdm as tqdm
import math
import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from datasets import load_dataset
from emoji import UNICODE_EMOJI

from torch.nn.utils.rnn import pad_sequence

#import wandb

#wandb.login()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


import warnings
warnings.filterwarnings("ignore")





dataset = load_dataset("hatexplain")

trainloader = torch.utils.data.DataLoader(dataset["train"], batch_size=1, shuffle=False)

testloader = torch.utils.data.DataLoader(dataset["test"], batch_size=1, shuffle=False)
valloader = torch.utils.data.DataLoader(dataset["validation"], batch_size=1, shuffle=False)





# Train Data
count = 0 
train_sentences = []
train_labels = []
train_rationales = []
for i,item in tqdm(enumerate(trainloader)):
    Sentence = ' '.join(''.join(tup) for tup in item["post_tokens"])
    Label = item["annotators"]["label"][0]
    try:
        True_Attention = torch.hstack(item["rationales"][0])
    except:
        True_Attention = torch.zeros(len(item["post_tokens"]))
    train_sentences.append(Sentence)
    train_labels.append(Label)
    train_rationales.append(True_Attention)




# Validation Data
val_sentences = []
val_labels = []
val_rationales = []
for i,item in tqdm(enumerate(valloader)):
    Sentence = ' '.join(''.join(tup) for tup in item["post_tokens"])
    Label = item["annotators"]["label"][0]
    try:
        True_Attention = torch.hstack(item["rationales"][0])
    except:
        True_Attention = torch.zeros(len(item["post_tokens"]))
    val_sentences.append(Sentence)
    val_labels.append(Label)
    val_rationales.append(True_Attention)




# Test Data
test_sentences = []
test_labels = []
test_rationales = []
for i,item in tqdm(enumerate(testloader)):
    Sentence = ' '.join(''.join(tup) for tup in item["post_tokens"])
    Label = item["annotators"]["label"][0]
    try:
        True_Attention = torch.hstack(item["rationales"][0])
    except:
        True_Attention = torch.zeros(len(item["post_tokens"]))
    
    test_sentences.append(Sentence)
    test_labels.append(Label)
    test_rationales.append(True_Attention)







# Create Vocabulary

PAD_token = 0
EOS_token = 1
UNK_token = 2

class Vocabulary:
    
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {"PAD":PAD_token,"EOS":EOS_token, "UNK":UNK_token}
        self.word2count = {}
        self.index2word = {PAD_token:"PAD",EOS_token:"EOS", UNK_token:"UNK"}
        self.num_words = 3
        
    def addSentence(self,sentence): #Add Sentence to vocabulary
        for word in sentence.split(' '):
            self.addWord(word)
            
    def addWord(self, word):  # Add words to vocabulary
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            if self.trimmed == False:
                self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            if self.trimmed == False:
                self.word2count[word] += 1
            
    def save(self,word2index_dic = 'word2index_dic', index2word_dic = 'index2word_dic',
         word2count_dic = 'word2count_dic'):

        with open('/content/Save/'+word2index_dic+'.p', 'wb') as fp:
            pickle5.dump(self.word2index, fp, protocol=pickle.HIGHEST_PROTOCOL)

        with open('/content/Save/'+index2word_dic+'.p', 'wb') as fp:
            pickle5.dump(self.index2word, fp, protocol=pickle.HIGHEST_PROTOCOL)

        with open('/content/Save/'+word2count_dic+'.p', 'wb') as fp:
            pickle5.dump(self.word2count, fp, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, word2index_dic = 'word2index_dic', index2word_dic = 'index2word_dic',
             word2count_dic = 'word2count_dic'):
        
        with open('/content/Save/'+word2index_dic+'.p', 'rb') as fp:
            self.word2index = pickle5.load(fp)
            
        with open('/content/Save/'+index2word_dic+'.p', 'rb') as fp:
            self.index2word = pickle5.load(fp)
            
        with open('/content/Save/'+word2count_dic+'.p', 'rb') as fp:
            self.word2count = pickle5.load(fp)
            
        self.num_words = len(self.word2index)
        
    def trim(self, min_count):  # Trim Rare words with frequency less than min_count
        if self.trimmed:
            print('Already trimmed before')
            return 0
        self.trimmed = True
        
        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.word2index = {"PAD":PAD_token,"EOS":EOS_token}
        #self.word2count = {}
        self.index2word = {PAD_token:"PAD",EOS_token:"EOS"}
        self.num_words = 3

        for word in keep_words:
            self.addWord(word)
            if word not in self.word2count:
                del self.word2count[word]




voc = Vocabulary("Hatexplain_data")
for sentence in train_sentences:
    voc.addSentence(sentence)
for sentence in test_sentences:
    voc.addSentence(sentence)
for sentence in val_sentences:
    voc.addSentence(sentence)
voc.trim(1)





def custom_collate_fn(data):
    sentence_index = []
    sentence_length = []
    sentence_rationales = []
    sentence_labels = []
    for d in data:
        sentence = d[0].split(" ")
        sentence_index.append(torch.tensor([voc.word2index[x] for x in sentence ]))
        sentence_length.append(len(sentence))
        sentence_rationales.append(torch.tensor(d[2]))
        sentence_labels.append(d[1].item())
    sentence_index = pad_sequence(sentence_index,batch_first=True)
    sentence_rationales = pad_sequence(sentence_rationales,batch_first=True,padding_value = 2) 
    return sentence_index,sentence_rationales,sentence_length,sentence_labels



class textDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, inputs,labels,rationales, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.inputs = inputs
        self.labels = labels
        self.rationales = rationales
        self.transform = transform

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
 

        return self.inputs[idx], self.labels[idx],self.rationales[idx]



trainset = textDataset(train_sentences,train_labels,train_rationales)
valset = textDataset(val_sentences,val_labels,val_rationales)
testset = textDataset(test_sentences,test_labels,test_rationales)



trainloader = DataLoader(trainset, batch_size=16000, shuffle=False, collate_fn=custom_collate_fn)
valloader = DataLoader(valset,batch_size=3000,shuffle=False,collate_fn= custom_collate_fn)
testloader = DataLoader(testset,batch_size=3000,shuffle=False,collate_fn=custom_collate_fn)


# Models


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)




class MultiAttention(nn.Module):
    def __init__(self,heads):
        super().__init__()
        self.heads = heads         
        
        
        self.query = nn.Linear(100,50*self.heads,bias=False)
        self.key = nn.Linear(100,50*self.heads,bias=False)
        self.value = nn.Linear(100,50*self.heads,bias=False)
        self.scaling = torch.tensor([100**(1/4)],device=device)
        self.unifyheads = nn.Linear(heads * 50, 100)
        

    def forward(self,x,mask):
        b = x.shape[0]
        t = x.shape[1]
        k = 50
        queries = self.query(x).view(b, t, self.heads, k)
        values  = self.value(x).view(b,t,self.heads,k)
        keys = self.key(x).view(b,t,self.heads,k)
        
        
        # - fold heads into the batch dimension
        keys = keys.transpose(1, 2).contiguous().view(b * self.heads, t, k)
        queries = queries.transpose(1, 2).contiguous().view(b * self.heads, t, k)
        values = values.transpose(1, 2).contiguous().view(b * self.heads, t, k)
        
        queries = queries /self.scaling
        keys    = keys / self.scaling
        
        attention_scores = torch.bmm(queries, keys.transpose(1, 2))
        
        attention_scores  = attention_scores + torch.cat([mask,mask],dim=0)[:,None,:]


        attention_weights = F.softmax(attention_scores, dim=2)
        
        context = torch.bmm(attention_weights, values).view(b, self.heads, t, k)
        context = context.transpose(1, 2).contiguous().view(b, t, self.heads* k)
        
        context = self.unifyheads(context)
        
        return x+context




class Focus(nn.Module):
    def __init__(self,):
        super(Focus,self).__init__()
        self.embedding = nn.Embedding(28041,100,padding_idx=0)
        

        self.positional_embedding = PositionalEncoding(100)
        self.attention_head = MultiAttention(heads= 2)
    
        self.fc1 = nn.Linear(100,50)
        self.fc2 = nn.Linear(50,1)
        #torch.nn.init.zeros_(self.fc1.weight)
        #torch.nn.init.zeros_(self.fc1.bias)
    
        
    def forward(self,x,mask):
        x  = self.embedding(x)
        x =self.positional_embedding(x)
        
        x = self.attention_head(x,mask)
        
           
        
    
        x1  = x
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        #x = torch.ones(x.size(0),x.size(1),1,device=device)
        x = x[:,:,0] + mask
        alpha = F.softmax(x,dim=1)
        #print(alpha)
        context  = x1 #torch.sum(alpha[:,:,None] *x1,dim=1)
        return alpha,context
    
        
        




class Classification(nn.Module):
    def __init__(self,):
        super(Classification,self).__init__()
        self.fc1 = nn.Linear(100,200)
        self.fc2 = nn.Linear(200,3)
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x




def my_cross_entropy(output,target,alpha,criterion):
    
    batch = alpha.size(0)
    patches = alpha.size(1)
    target = target.repeat_interleave(patches)
    loss = criterion(output,target)
    loss = torch.reshape(loss,(batch,patches))
    if len(alpha.shape)>2:
        final_loss = torch.sum(loss*alpha[:,:,0],dim=1)
    else:
        final_loss = torch.sum(loss*alpha,dim=1)
    final_loss = torch.mean(final_loss,dim=0)
    return final_loss




Criterion = nn.CrossEntropyLoss(reduction="none")

instance,instance_rationale, instance_length, instance_labels = iter(trainloader).next()


masks = torch.zeros((len(instance_length),np.max(instance_length)))
for i in range(len(masks)):
    masks[i,instance_length[i]:] = -np.inf
    
    
    
val_instance,val_instance_rationale, val_instance_length, val_instance_labels = iter(valloader).next()


val_masks = torch.zeros((len(val_instance_length),np.max(val_instance_length)))
for i in range(len(val_masks)):
    val_masks[i,val_instance_length[i]:] = -np.inf
    
    
test_instance,test_instance_rationale, test_instance_length, test_instance_labels = iter(testloader).next()


test_masks = torch.zeros((len(test_instance_length),np.max(test_instance_length)))
for i in range(len(test_masks)):
    test_masks[i,test_instance_length[i]:] = -np.inf



class textprocessedDataset(Dataset):


    def __init__(self, inputs,labels,rationales,masks, transform=None):
  
        self.inputs = inputs
        self.labels = labels
        self.rationales = rationales
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
 

        return self.inputs[idx], self.labels[idx],self.rationales[idx],self.masks[idx]




processedtrainset = textprocessedDataset(instance,instance_labels,instance_rationale,masks)
processedtrainloader = DataLoader(processedtrainset, batch_size=256, shuffle=False)


processedvalset = textprocessedDataset(val_instance,val_instance_labels,val_instance_rationale,val_masks)
processedvalloader = DataLoader(processedvalset, batch_size=256, shuffle=False)

processedtestset = textprocessedDataset(test_instance,test_instance_labels,test_instance_rationale,test_masks)
processedtestloader = DataLoader(processedtestset, batch_size=256, shuffle=False)





torch.manual_seed(12)
focus = Focus().to(device)
focus.load_state_dict(torch.load("focus_sa.pth", map_location=device))
torch.manual_seed(12)
classification = Classification().to(device)
classification.load_state_dict(torch.load("classification_sa.pth", map_location=device))

true_class = 0
total = 0
ft = 0 
with torch.no_grad():
    for i,data in enumerate(processedtrainloader):
        inputs,labels,labels_rationales,rationales_mask = data
        inputs,labels, = inputs.to(device),labels.to(device)
        labels_rationales,rationales_mask = labels_rationales.to(device),rationales_mask.to(device)

        alpha,context = focus(inputs,rationales_mask)
        
        outputs = classification(context)
        batch = outputs.shape[0]
        patches = outputs.shape[1]
        outputs = outputs.reshape(batch*patches,outputs.shape[2])
        
        loss = my_cross_entropy(outputs,labels,alpha,Criterion)
       
        outputs = classification(torch.sum(alpha[:,:,None] *context,dim=1))
        _,prediction = outputs.max(1)
        #print(prediction.shape,labels.shape)
        true_class += np.sum(prediction.cpu().numpy() == labels.cpu().numpy()).item()
        total += len(labels)
        indexes = torch.argmax(alpha,dim=1)
        ft += sum(labels_rationales[np.arange(len(labels)),indexes]).item()
print(true_class/total,ft/total)


# Training Starts


seeds = 12
nos_epochs = 50

run_no = 0
learning_rates =[0.008] #[0.2,0.1,0.05,0.02,0.01,0.005,0.002,0.001,0.0005]
#run = wandb.init(project="HateXplain_data",
#               name ="Self_Attn_Hybrid_Final",config = {
#                   "learning rate ":learning_rates,
#                   "epochs":50
#               },save_code=True)

# wandb.run.log_code(".") 


for j in range(len(learning_rates)):
    torch.manual_seed(12)
    focus = Focus().to(device)
    focus.load_state_dict(torch.load("focus_sa.pth", map_location=device))
    torch.manual_seed(12)
    classification = Classification().to(device)
    classification.load_state_dict(torch.load("classification_sa.pth", map_location=device))
    lr = learning_rates[j]
    
    focus_optimizer = optim.SGD(focus.parameters(), lr=lr,momentum=0.9)
    classification_optimizer = optim.SGD(classification.parameters(),lr=lr,momentum=0.9)
    loss_list = []
    for epoch in tqdm(range(nos_epochs)):
        focus.train()
        classification.train()
        epoch_loss = [] 
        ft1 = 0 
        total1 = 0
        for i,data in enumerate(processedtrainloader):
            inputs,labels,labels_rationales,rationales_mask = data
            inputs,labels, = inputs.to(device),labels.to(device)
            labels_rationales,rationales_mask = labels_rationales.to(device),rationales_mask.to(device)
            
            focus_optimizer.zero_grad()
            classification_optimizer.zero_grad()
            alpha,context = focus(inputs,rationales_mask)

            #print(images.shape)
            outputs = classification(context)
            batch = outputs.shape[0]
            patches = outputs.shape[1]
            outputs = outputs.reshape(batch*patches,outputs.shape[2])
            loss = my_cross_entropy(outputs,labels,alpha,Criterion)
            loss.backward()
            focus_optimizer.step()
            classification_optimizer.step()
            with torch.no_grad():
                alpha,context = focus(inputs,rationales_mask)
                
                outputs = classification(context)
                batch = outputs.shape[0]
                patches = outputs.shape[1]
                outputs = outputs.reshape(batch*patches,outputs.shape[2])
                loss = my_cross_entropy(outputs,labels,alpha,Criterion)
                 
                total1 += len(labels)
                indexes = torch.argmax(alpha,dim=1)
                ft1 += sum(labels_rationales[np.arange(len(labels)),indexes]).item()

            epoch_loss.append(loss.item())
            #print('[%d] loss: %.3f' %(epoch+1,np.mean(epoch_loss)))
        #run.log({"loss_lr_"+str(learning_rates[j])+"_"+str(seeds):np.mean(epoch_loss).item(),
        #         "Epoch":epoch+1,
        #        "Ft"+str(learning_rates[j])+"_"+str(seeds):ft1/total1})
        #run.log({"Epoch":epoch+1})
        loss_list.append(np.mean(epoch_loss))   

    print("Finished Training")
    torch.save(focus.state_dict(), 'focus_'+str(learning_rates[j])+'.pth')  
    torch.save(classification.state_dict(), 'classification_'+str(learning_rates[j])+'.pth')  
    #artifact = wandb.Artifact('model', type='model')
    #artifact.add_file('focus_'+str(learning_rates[j])+'.pth')
    #artifact.add_file('classification_'+str(learning_rates[j])+'.pth')
    #run.log_artifact(artifact)

    true_class = 0
    total = 0
    ft = 0 

    true_class = 0
    total = 0
    ft = 0 
    
    with torch.no_grad():
        for i,data in enumerate(processedtrainloader):
            inputs,labels,labels_rationales,rationales_mask = data
            inputs,labels = inputs.to(device),labels.to(device)
            labels_rationales,rationales_mask = labels_rationales.to(device),rationales_mask.to(device)
            alpha,context = focus(inputs,rationales_mask)
            outputs = classification(torch.sum(alpha[:,:,None] *context,dim=1))
            
            _,prediction = outputs.max(1)
            #print(prediction.shape,labels.shape)
            true_class += np.sum(prediction.cpu().numpy() == labels.cpu().numpy()).item()
            total += len(labels)
            indexes = torch.argmax(alpha,dim=1)
            ft += sum(labels_rationales[np.arange(len(labels)),indexes]).item()



    print("Training Accuracy_"+str(learning_rates[j]),true_class/total,ft/total) 
    
    true_class = 0
    total = 0
    ft = 0 
    
    with torch.no_grad():
        for i,data in enumerate(processedvalloader):
            inputs,labels,labels_rationales,rationales_mask = data
            inputs,labels = inputs.to(device),labels.to(device)
            labels_rationales,rationales_mask = labels_rationales.to(device),rationales_mask.to(device)
            alpha,context = focus(inputs,rationales_mask)
            outputs = classification(torch.sum(alpha[:,:,None] *context,dim=1))
            loss = Criterion(outputs,labels) 
            _,prediction = outputs.max(1)
            #print(prediction.shape,labels.shape)
            true_class += np.sum(prediction.cpu().numpy() == labels.cpu().numpy()).item()
            total += len(labels)
            indexes = torch.argmax(alpha,dim=1)
            ft += sum(labels_rationales[np.arange(len(labels)),indexes]).item()




    print("Validation Accuracy_"+str(learning_rates[j]),true_class/total,ft/total) 


    
    with torch.no_grad():
        for i,data in enumerate(processedtestloader):
            inputs,labels,labels_rationales,rationales_mask = data
            inputs,labels = inputs.to(device),labels.to(device)
            labels_rationales,rationales_mask = labels_rationales.to(device),rationales_mask.to(device)
            alpha,context = focus(inputs,rationales_mask)
            outputs = classification(torch.sum(alpha[:,:,None] *context,dim=1))
            loss = Criterion(outputs,labels) 
            _,prediction = outputs.max(1)
            #print(prediction.shape,labels.shape)
            true_class += np.sum(prediction.cpu().numpy() == labels.cpu().numpy()).item()
            total += len(labels)
            indexes = torch.argmax(alpha,dim=1)
            ft += sum(labels_rationales[np.arange(len(labels)),indexes]).item()




    print("Test Accuracy_"+str(learning_rates[j]),true_class/total,ft/total) 



alpha_list = []
prediction_list1 = []
true_class = 0
total = 0
ft =0 

with torch.no_grad():
    for i,data in enumerate(processedtestloader ):
        inputs,labels,labels_rationales,rationales_mask = data
        inputs,labels = inputs.to(device),labels.to(device)
        labels_rationales,rationales_mask = labels_rationales.to(device),rationales_mask.to(device)
        alpha,context = focus(inputs,rationales_mask)
        #print(context.shape,alpha.shape)
         
        outputs = torch.softmax(classification(torch.sum(alpha[:,:,None] * context,dim=1)),dim=1)
       
        _,prediction = outputs.max(1)
        #print(prediction.shape,labels.shape)
        true_class += np.sum(prediction.cpu().numpy() == labels.cpu().numpy()).item()
        total += len(labels)
        indexes = torch.argmax(alpha,dim=1)
        ft += sum(labels_rationales[np.arange(len(labels)),indexes]).item()
        
        #fore_idx = []
        list_alpha = []
        list_prediction = []
        for j in range(len(labels_rationales)):
            if 1 in labels_rationales[j]:
                #print(1)
                fore_idx = torch.where(labels_rationales[j]==1)
                list_alpha.append(torch.sum(alpha[j,fore_idx[0]]).cpu().numpy())
                list_prediction.append(outputs[j,labels[j]].cpu().numpy())
        alpha_list.append(list_alpha)
        prediction_list1.append(list_prediction)



alpha_list = np.hstack(alpha_list)
prediction_list = np.hstack(prediction_list1)

alpha_list.shape,prediction_list.shape


# In[36]:


from matplotlib.colors import ListedColormap
from matplotlib import pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns


# In[37]:


#path1 = "/home/malt/Documents/Notebooks/Soft_Attention/CIFAR/inference_methods_for_Attention/CIFAR10/averaging_input/"
plt.figure(figsize=(6,6))
im = plt.hist2d(alpha_list,prediction_list,(5,5))
num = im[0].sum()/100
ax = sns.heatmap(np.round(im[0].transpose()/num,1),vmin=5,vmax=70,annot=True,fmt="g",cmap=sns.color_palette("coolwarm"),
                 yticklabels=[0.2,0.4,0.6,0.8,1.],
                 xticklabels=[0.2,0.4,0.6,0.8,1],annot_kws={"size":18},cbar=False)
ax.invert_yaxis()

plt.xlabel(r"$\bf a_z$",fontweight="bold",fontsize=20)
plt.ylabel(r"$\bf s^{Hard}_y$",fontweight="bold",fontsize=20) # change algo based on algo
plt.xticks([1,2,3,4,5],weight="bold",fontsize=18)
plt.yticks([1,2,3,4,5],weight="bold", va="top",fontsize=18)

#plt.savefig("lvml_sa.png")
plt.savefig("Hard.pdf")



#wandb.finish()





