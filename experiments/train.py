log_folder = 'rn18_imagenet_32r4-0r2-0r0_avgpFC-A'

# Definición de la topología
# Arquitectura ResNet
# swap define si se permutan los pesos. Solo para probar. No tiene sentido no hacerlo.
topo = {'conv1': {'r4': 32,   'r2': 0,   'r0': 0,                },
        'mod1':  {'r4': 32,   'r2': 0,   'r0': 0,   "swap": True },
        'mod2':  {'r4': 32*2, 'r2': 0*2, 'r0': 0*2, "swap": True },
        'mod3':  {'r4': 32*4, 'r2': 0*4, 'r0': 0*4, "swap": True },
        'mod4':  {'r4': 32*8, 'r2': 0*8, 'r0': 0*8, "swap": True }}

# batch size por epoch
bsdict = {100: 256,
          200: 256,
          9999999: 256} # train batchsize para epochs < key

# cantidad de epochs hasta la 1er parada
str_epochs = "300"

#-------------------------------------------------------------------------------

import numpy as np
import os
import time
import random
from datetime import datetime, timedelta
import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18, ResNet18_Weights
import pickle
from typing import Union, Tuple
from models.rotk import Conv2d_rotk, NeuralNetwork

parent_log_folder = './logs/'
parent_log_folderR = './logsR/'
datadir = './data/imagenet1k/'
trainpath = datadir + 'train_images/'
testpath = datadir + 'val_images/'

codelist = [ord(c) for c in log_folder]
seed = np.sum(codelist)
np.random.seed(seed=seed)
random.seed(10)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

log_folderR = parent_log_folderR + log_folder
log_folder = parent_log_folder + log_folder

if not os.path.isdir(log_folder):
  os.makedirs(log_folder)

# DATA

def getbs(x, bsdict):
  bs = 1
  epochth = list(bsdict.keys())
  epochth.reverse()
  for th in epochth:
    if x < th:
      bs = bsdict[th]
  return(bs)

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
         transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

train_dataset = datasets.ImageFolder(trainpath, transform=data_transforms['train'])

test_dataset = datasets.ImageFolder(testpath, transform=data_transforms['test'])
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=bsdict[100], shuffle=False, pin_memory=True, num_workers=5)

# MODEL

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device: " + device)

model = NeuralNetwork(topo=topo, device=device).to(device)

if not os.path.isfile(log_folder + '/model_state_dict.pth'):
  print('Inicializando modelo...')
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters())
  train_loss=[]
  train_accuary=[]
  test_loss=[]
  test_accuary=[]
  train_acc_steps=[]
  train_loss_steps=[]
  times=[]
  epochs_ini = -1
  steps = steps_ini = 0
else:
  print('Cargando modelo...')
  model.load_state_dict(torch.load(log_folder + '/model_state_dict.pth', map_location=device))
  # ver https://pytorch.org/tutorials/recipes/recipes/save_load_across_devices.html
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters())
  with open(log_folder + '/train_loss.pickle', 'rb') as handle:
    train_loss = pickle.load(handle)
  with open(log_folder + '/train_accuary.pickle', 'rb') as handle:
    train_accuary = pickle.load(handle)
  with open(log_folder + '/test_loss.pickle', 'rb') as handle:
    test_loss = pickle.load(handle)
  with open(log_folder + '/test_accuary.pickle', 'rb') as handle:
    test_accuary = pickle.load(handle)
  with open(log_folder + '/train_acc_steps.pickle', 'rb') as handle:
    train_acc_steps = pickle.load(handle)
  with open(log_folder + '/train_loss_steps.pickle', 'rb') as handle:
    train_loss_steps = pickle.load(handle)
  with open(log_folder + '/epochs_ini.pickle', 'rb') as handle:
    epochs_ini = pickle.load(handle)
  with open(log_folder + '/steps_ini.pickle', 'rb') as handle:
    steps_ini = pickle.load(handle)
    steps = steps_ini
  with open(log_folder + '/times.pickle', 'rb') as handle:
    times = pickle.load(handle)

total_params = sum(p.numel() for p in model.parameters())
total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('params:    ' + str(total_params))
print('trainable: ' + str(total_trainable_params))

# TRAINING
currentbs = -1
inpute = str_epochs
while not inpute.isnumeric():
  inpute = input("epochs (num): ")
num_epochs = int(inpute)
print("Training for " + str(num_epochs) + " epochs...")
# while(num_epochs > 0):
#     epochs_ini_ini = epochs_ini + 1
#     for epoch in range(epochs_ini + 1, num_epochs + epochs_ini + 1):
#         start_time = time.time()
#         if len(times) == 0:
#             secs_estim = 0
#         else:
#             secs_estim = int(np.array(times).mean())* (num_epochs+epochs_ini)
raise NotImplementedError("Training loop missing from the provided script.")
