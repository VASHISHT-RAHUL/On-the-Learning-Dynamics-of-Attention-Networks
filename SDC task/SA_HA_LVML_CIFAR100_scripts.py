import sys
path = "/" # add path to scripts here 
sys.path.append(path)
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
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Change for Different Settings here
#------------------------------------------------------------------------------------
m = 5  # number of patches
desired_num = 40000 # number of training points train + test
batch = 256  # batch size for mosaic data
tr = 30000 # number of training points

nos_epochs = 150 # number of epochs to train the model

learning_rates = [0.05] #[0.2,0.1,0.05,0.02,0.01,0.005,0.002,0.001,0.0005]import wandb

algorithm = "soft" # soft, hard or LVML here
#------------------------------------------------------------------------------------------------
# # Generate Dataset



def labels_to_index(fg_labels):
    unique_foreground_labels = list(np.unique(fg_labels))
    foreground_labels = []
    for fg in fg_labels:
        foreground_labels.append(unique_foreground_labels.index(fg))
    return torch.tensor(foreground_labels,dtype=torch.int64)


# In[ ]:


def Create_Mosaic_data(desired_num,m,foreground_label,background_label,foreground_data,background_data,dataset="None"):
    
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


# In[ ]:


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


train_mosaic_data,train_mosaic_label,train_fore_idx = Create_Mosaic_data(desired_num,m,train_fg_label,
                                     train_bg_label,train_fg_data,train_bg_data,"training")


# In[ ]:


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


# In[ ]:



msd = MosaicDataset(train_mosaic_data[0:tr], train_mosaic_label[0:tr] , train_fore_idx[0:tr])
train_loader = DataLoader( msd,batch_size= batch ,shuffle=False)

msd1 = MosaicDataset(train_mosaic_data[tr:],train_mosaic_label[tr:] , train_fore_idx[tr:])
test_loader = DataLoader( msd1,batch_size= batch ,shuffle=False)


# In[ ]:


def print_analysis(data_loader,focus,classification,dataset="None"):
    ftpt_1,ffpt_1,ftpf_1,ffpf_1,accuracy_1 = evaluation_method_1(data_loader,focus,classification)
    ftpt_2,ffpt_2,ftpf_2,ffpf_2,accuracy_2 = evaluation_method_2(data_loader,focus,classification)
    ftpt_3,ffpt_3,ftpf_3,ffpf_3,accuracy_3 = evaluation_method_3(data_loader,focus,classification)
    
    print(str(dataset)+"_Evaluation Method 1")
    print("*"*60)
    print("FTPT",ftpt_1)
    print("FFPT",ffpt_1)
    print("FTPF",ftpf_1)
    print("FFPF",ffpf_1)
    print("Accuracy",accuracy_1)
    
    print(str(dataset)+"_Evaluation Method 2")
    print("*"*60)
    print("FTPT",ftpt_2)
    print("FFPT",ffpt_2)
    print("FTPF",ftpf_2)
    print("FFPF",ffpf_2)
    print("Accuracy",accuracy_2)
    
    print(str(dataset)+"_Evaluation Method 3")
    print("*"*60)
    print("FTPT",ftpt_3)
    print("FFPT",ffpt_3)
    print("FTPF",ftpf_3)
    print("FFPF",ffpf_3)
    print("Accuracy",accuracy_3)
    
    
    


# In[ ]:


if algorithm =="LVML":
    print("---- training LVML model -----")
    torch.manual_seed(12)
    focus = Focus_cnn1()
    focus = focus.to(device)
    torch.manual_seed(12)
    classification = Classification_cnn1(50)
    classification = classification.to(device)

    lr = learning_rates[0] 
    #Criterion = nn.CrossEntropyLoss(reduction="none") #nn.BCELoss(reduction="none")
    optimizer_focus = optim.SGD(focus.parameters(), lr=lr,momentum=0.9)
    optimizer_classification = optim.SGD(classification.parameters(), lr=lr,momentum=0.9)
    loss_list = []
    for i in tqdm(range(nos_epochs)):
        epoch_loss = []
        classification_epoch_loss = []
        for j,data in enumerate(train_loader):
            images,labels,foreground_index = data
            images = images.to(device)
            labels = labels.to(device)

            optimizer_focus.zero_grad()
            optimizer_classification.zero_grad()

            loss = marginal_loss(focus,classification,images,labels)
            loss.backward()
            optimizer_focus.step()
            optimizer_classification.step()

            with torch.no_grad():
                loss = marginal_loss(focus,classification,images,labels)
                epoch_loss.append(loss.item())
        print("*"*60)
        print("Epoch: " + str(i+1)+", Loss: "+str(np.mean(epoch_loss)))

        loss_list.append(np.mean(epoch_loss))
    print_analysis(train_loader,focus,classification,dataset="training")
    print_analysis(test_loader,focus,classification,dataset="testing")
    print_analysis(test_data_loader,focus,classification,dataset="cifar test")
    print("Finished Training")
    torch.save(focus.state_dict(), 'focus.pth')  # save the model in current directory
    torch.save(classification.state_dict(), 'classification.pth')   # save the model in current directory
elif algorithm == "soft":
    print("----- training soft attention model -----")
        torch.manual_seed(seeds)
        focus = Focus_cnn1()
        focus = focus.to(device)

        torch.manual_seed(seeds)
        classification = Classification_cnn1(50)
        classification = classification.to(device)


        lr = learning_rates[run_no] 

        Criterion = nn.CrossEntropyLoss() #nn.BCELoss(reduction="none")
        focus_optimizer = optim.SGD(focus.parameters(), lr=lr,momentum=0.9)
        classification_optimizer = optim.SGD(classification.parameters(),lr=lr,momentum=0.9)
        loss_list = []
        for epoch in tqdm(range(nos_epochs)):
            focus.train()
            classification.train()
            epoch_loss = [] 
            for i,data in enumerate(train_loader):
                focus,classification,focus_optimizer,classification_optimizer=train_model_sa(data,
                                                                                          focus,
                                                                                          classification,
                                                                                          focus_optimizer,
                                                                                          classification_optimizer,
                                                                                          Criterion)
                with torch.no_grad():
                    images,labels,fore_idx = data
                    batch = images.size(0)
                    patches = images.size(1)
                    images,labels = images.to(device),labels.to(device)
                    alphas = torch.softmax(focus(images),dim=1)
                    images = torch.sum(alphas[:,:,None,None,None]*images,dim=1)
                    outputs = classification(images)
                    loss = Criterion(outputs,labels)

                epoch_loss.append(loss.item())
            print('[%d] loss: %.3f' %(epoch+1,np.mean(epoch_loss)))
            loss_list.append(np.mean(epoch_loss))   

        print_analysis(train_loader,focus,classification,dataset="training")
        print_analysis(test_loader,focus,classification,dataset="testing")
        print_analysis(test_data_loader,focus,classification,dataset="cifar test")


    print("Finished Training")
    torch.save(focus.state_dict(), 'focus_'+str(seeds)+'.pth')  
    torch.save(classification.state_dict(), 'classification_'+str(seeds)+'.pth')  

elif algorithm =="hard":
    print("---- training hard attention model----")
    torch.manual_seed(seeds)
    focus = Focus_cnn1()
    focus = focus.to(device)
        
    torch.manual_seed(seeds)
    classification = Classification_cnn1(50)
    classification = classification.to(device)
    lr = learning_rates[run_no] 
        
    Criterion = nn.CrossEntropyLoss(reduction="none") #nn.BCELoss(reduction="none")
    focus_optimizer = optim.SGD(focus.parameters(), lr=lr,momentum=0.9)
    classification_optimizer = optim.SGD(classification.parameters(),lr=lr,momentum=0.9)
    loss_list = []
        

    for epoch in tqdm(range(nos_epochs)):
        focus.train()
        classification.train()

        epoch_loss = [] 

        for i,data in enumerate(train_loader):
            focus,classification,focus_optimizer,classification_optimizer=train_model(data,
                                                                                      focus,
                                                                                      classification,
                                                                                      focus_optimizer,
                                                                                      classification_optimizer,
                                                                                      Criterion)

            with torch.no_grad():
                images,labels,fore_idx = data
                batch = images.size(0)
                patches = images.size(1)
                images,labels = images.to(device),labels.to(device)
                alphas = torch.softmax(focus(images),dim=1)
                images =  images.reshape(batch*patches,3,32,32)
                outputs = classification(images)
                loss = my_cross_entropy(outputs,labels,alphas,Criterion)

                epoch_loss.append(loss.item())
            print('[%d] loss: %.3f' %(epoch+1,np.mean(epoch_loss)))

            loss_list.append(np.mean(epoch_loss))

        print_analysis(train_loader,focus,classification,dataset="training")
        print_analysis(test_loader,focus,classification,dataset="testing")
        print_analysis(test_data_loader,focus,classification,dataset="cifar test")


    print("Finished Training")
    torch.save(focus.state_dict(), 'focus_'+str(seeds)+'.pth')  
    torch.save(classification.state_dict(), 'classification_'+str(seeds)+'.pth')  

