# -*- coding: utf-8 -*-

"""
Created on Sun Sep 20 23:38:32 2020

@author: deviantpadam
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
from PIL import Image, ImageFilter
import tqdm.auto as tqdm
import kornia


class ImageClassifier(nn.Module):
    def __init__(self):
        super(ImageClassifier,self).__init__()
        num_classes=10
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.feature_extractor = models.mobilenet_v2(pretrained=True)
        self.feature_extractor.classifier[1] = nn.Linear(in_features=self.feature_extractor.classifier[1].in_features,out_features=num_classes)
#         self.feature_extractor = models.resnet50(pretrained=True)
#         self.feature_extractor.eval()
#         self.feature_extractor.fc = nn.Linear(self.feature_extractor.fc.in_features,out_features=num_classes)
#         self.classifier = nn.Linear(in_features=1000,out_features=num_classes)
        
        
        self.data_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                                   transforms.RandomHorizontalFlip(),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize([0.485, 0.456, 0.406], 
                                                                         [0.229, 0.224, 0.225])])
        
        self.translate = {'cane': 'dog',
                          'cavallo': 'horse',
                          'elefante': 'elephant',
                          'farfalla': 'butterfly',
                          'gallina': 'chicken',
                          'gatto': 'cat',
                          'mucca': 'cow',
                          'pecora': 'sheep', 
                          'scoiattolo': 'squirrel',
                          'ragno': 'spider'}
        
    def forward(self,x):
        embeddings = self.feature_extractor(x)
        x = embeddings
#         x = self.classifier(embeddings)
        return x
    
    def fit(self,root_dir,num_epochs,batch_size,val_split=0.1,num_workers=4):
        image_datasets = datasets.ImageFolder(root_dir,self.data_transforms)
        image_datasets.classes = [self.translate[i] for i in image_datasets.classes]
        train_size = int((1-val_split)*len(image_datasets))
        train, valid = torch.utils.data.random_split(image_datasets,[train_size,len(image_datasets)-train_size])
        dataloaders = {'train': torch.utils.data.DataLoader(train, batch_size=batch_size,
                                             shuffle=True, num_workers=num_workers),
                       'val': torch.utils.data.DataLoader(valid, batch_size=batch_size,
                                             shuffle=True, num_workers=num_workers)}
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(),lr=0.0001)
        epoch_bar = tqdm.tqdm(range(num_epochs),desc='Epochs---',position=0)
        self.train_epoch_loss = []
        self.val_epoch_loss = []
        train_acc = 0.
        val_acc = 0.
        for epoch in epoch_bar:
            self.train()
            pbar = tqdm.tqdm(dataloaders['train'],desc='Epoch= {}'.format(epoch+1),position=1)
            running_loss = 0.0
            val_loss = 0.0
            total = 0
            correct = 0
            for inputs, labels in pbar:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # zero the parameter gradients
                optimizer.zero_grad()
                labels_hat = self(inputs)
                _, predicted = torch.max(labels_hat.data, 1)
                loss = criterion(labels_hat,labels)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                pbar.set_postfix(train_run_loss='{:.4f}'.format(loss.item()))
                running_loss += loss.item()
                loss.backward()
                optimizer.step()
            self.train_epoch_loss.append(running_loss / len(dataloaders['train']))
            train_acc = 100 * correct / total
            epoch_bar.set_postfix(train_acc='{:.4f}'.format(train_acc),val_acc='{:.4f}'.format(val_acc))    
#             pbar = tqdm.tqdm(dataloaders['val'],desc='Epoch={}, train_epoch_loss= {}'.format(epoch+1,epoch_loss))
            total = 0
            correct = 0
            with torch.no_grad():
                self.eval()
                for inputs, labels in dataloaders['val']:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    labels_hat = self(inputs)
                    _, predicted = torch.max(labels_hat.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    loss = criterion(labels_hat,labels)
                    val_loss += loss.item()
                self.val_epoch_loss.append(val_loss / len(dataloaders['val']))
                val_acc = 100 * correct / total
                epoch_bar.set_postfix(train_acc='{:.4f}'.format(train_acc),val_acc='{:.4f}'.format(val_acc))
                
    def fit_on_blur(self,root_dir,num_epochs,batch_size,val_split=0.1,num_workers=4):
        image_datasets = datasets.ImageFolder(root_dir,self.data_transforms)
        image_datasets.classes = [self.translate[i] for i in image_datasets.classes]
        train_size = int((1-val_split)*len(image_datasets))
        train, valid = torch.utils.data.random_split(image_datasets,[train_size,len(image_datasets)-train_size])
        dataloaders = {'train': torch.utils.data.DataLoader(train, batch_size=batch_size,
                                             shuffle=True, num_workers=num_workers),
                       'val': torch.utils.data.DataLoader(valid, batch_size=batch_size,
                                             shuffle=True, num_workers=num_workers)}
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(),lr=0.0001)
        epoch_bar = tqdm.tqdm(range(num_epochs),desc='Epochs---',position=0)
        self.train_epoch_loss = []
        self.val_epoch_loss = []
        train_acc = 0.
        val_acc = 0.
        for epoch in epoch_bar:
            self.train()
            pbar = tqdm.tqdm(dataloaders['train'],desc='Epoch= {}'.format(epoch+1),position=1)
            running_loss = 0.0
            val_loss = 0.0
            total = 0
            correct = 0
            for inputs, labels in pbar:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                inputs = torch.stack([kornia.gaussian_blur2d(inputs[i].view(1,3,224,224),kernel_size=(25,25),sigma=(4,4)).view(3,224,224) for i in range(len(inputs))])
                # zero the parameter gradients
                optimizer.zero_grad()
                labels_hat = self(inputs)
                _, predicted = torch.max(labels_hat.data, 1)
                loss = criterion(labels_hat,labels)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                pbar.set_postfix(train_run_loss='{:.4f}'.format(loss.item()))
                running_loss += loss.item()
                loss.backward()
                optimizer.step()
            self.train_epoch_loss.append(running_loss / len(dataloaders['train']))
            train_acc = 100 * correct / total
            epoch_bar.set_postfix(train_acc='{:.4f}'.format(train_acc),val_acc='{:.4f}'.format(val_acc))    
#             pbar = tqdm.tqdm(dataloaders['val'],desc='Epoch={}, train_epoch_loss= {}'.format(epoch+1,epoch_loss))
            total = 0
            correct = 0
            with torch.no_grad():
                self.eval()
                for inputs, labels in dataloaders['val']:
                    inputs = inputs.to(self.device)
                    inputs = torch.stack([kornia.gaussian_blur2d(inputs[i].view(1,3,224,224),kernel_size=(25,25),sigma=(4,4)).view(3,224,224) for i in range(len(inputs))])
                    labels = labels.to(self.device)
                    labels_hat = self(inputs)
                    _, predicted = torch.max(labels_hat.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    loss = criterion(labels_hat,labels)
                    val_loss += loss.item()
                self.val_epoch_loss.append(val_loss / len(dataloaders['val']))
                val_acc = 100 * correct / total
                epoch_bar.set_postfix(train_acc='{:.4f}'.format(train_acc),val_acc='{:.4f}'.format(val_acc))   

    def fit_on_edges(self,root_dir,num_epochs,batch_size,val_split=0.1,num_workers=4):
        image_datasets = datasets.ImageFolder(root_dir,self.data_transforms)
        image_datasets.classes = [self.translate[i] for i in image_datasets.classes]
        train_size = int((1-val_split)*len(image_datasets))
        train, valid = torch.utils.data.random_split(image_datasets,[train_size,len(image_datasets)-train_size])
        dataloaders = {'train': torch.utils.data.DataLoader(train, batch_size=batch_size,
                                             shuffle=True, num_workers=num_workers),
                       'val': torch.utils.data.DataLoader(valid, batch_size=batch_size,
                                             shuffle=True, num_workers=num_workers)}
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(),lr=0.0001)
        epoch_bar = tqdm.tqdm(range(num_epochs),desc='Epochs---',position=0)
        self.train_epoch_loss = []
        self.val_epoch_loss = []
        train_acc = 0.
        val_acc = 0.
#         self.to(self.device)
        for epoch in epoch_bar:
            self.train()
            pbar = tqdm.tqdm(dataloaders['train'],desc='Epoch= {}'.format(epoch+1),position=1)
            running_loss = 0.0
            val_loss = 0.0
            total = 0
            correct = 0
            for inputs, labels in pbar:
                inputs = inputs.to(self.device)
                inputs = torch.stack([kornia.sobel(inputs[i].view(1,3,224,224)).view(3,224,224) for i in range(len(inputs))])
                labels = labels.to(self.device)
                
                # zero the parameter gradients
                optimizer.zero_grad()
                labels_hat = self(inputs)
                _, predicted = torch.max(labels_hat.data, 1)
                loss = criterion(labels_hat,labels)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                pbar.set_postfix(train_run_loss='{:.4f}'.format(loss.item()))
                running_loss += loss.item()
                loss.backward()
                optimizer.step()
            self.train_epoch_loss.append(running_loss / len(dataloaders['train']))
            train_acc = 100 * correct / total
            epoch_bar.set_postfix(train_acc='{:.4f}'.format(train_acc),val_acc='{:.4f}'.format(val_acc))    
#             pbar = tqdm.tqdm(dataloaders['val'],desc='Epoch={}, train_epoch_loss= {}'.format(epoch+1,epoch_loss))
            total = 0
            correct = 0
            with torch.no_grad():
                self.eval()
                for inputs, labels in dataloaders['val']:
                    inputs = inputs.to(self.device)
                    inputs = torch.stack([kornia.sobel(inputs[i].view(1,3,224,224)).view(3,224,224) for i in range(len(inputs))])
                    labels = labels.to(self.device)
                    labels_hat = self(inputs)
                    _, predicted = torch.max(labels_hat.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    loss = criterion(labels_hat,labels)
                    val_loss += loss.item()
                self.val_epoch_loss.append(val_loss / len(dataloaders['val']))
                val_acc = 100 * correct / total
                epoch_bar.set_postfix(train_acc='{:.4f}'.format(train_acc),val_acc='{:.4f}'.format(val_acc))
