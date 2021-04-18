from utils.data_loader import DataLoader4Signature
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as utils
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from torch.autograd import Variable
import json
from vehicle_signature.contrastive import ContrastiveLoss
from vehicle_signature.siamese import SiameseNetwork

data_root = "/content/drive/MyDrive/21Spring/COMS 6998/Project/6998-sec11-final-project/CNR-EXT-Patches-150x150/"
annotation_files = data_root + 'LABELS/all.txt'
EPOCHS = 10
BATCH_SIZE = 32

net = SiameseNetwork().cuda()
criterion = ContrastiveLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=0.0005)

# train the model
def train():
    loss = []
    data_loader = DataLoader4Signature(data_root,
                                   annotation_files)
    train_loader, val = data_loader.load(batch_size=BATCH_SIZE, repeat=True)

    for epoch in range(1, EPOCHS + 1):
        for idx, (img1, img2, labels) in enumerate(train_loader):
            img1, img2, labels = img1.numpy(), img2.numpy(), labels.numpy()
            img1, img2, labels = torch.from_numpy(img1), torch.from_numpy(img2), \
              torch.from_numpy(labels)
            img1, img2, labels = torch.reshape(img1, (img1.size(0), 3, 150, 150)), \
              torch.reshape(img2, (img1.size(0), 3, 150, 150)), torch.reshape(labels, (labels.size(0), 1)) 
            img1, img2, labels = img1.cuda(), img2.cuda(), labels.cuda()
            optimizer.zero_grad()
            output1, output2 = net(img1, img2)
            loss_contrastive = criterion(output1, output2, labels)
            loss_contrastive.backward()
            optimizer.step()
            if idx % 100 == 0:
                print(idx, loss_contrastive.item())
        print("Epoch {} Current loss {}".format(epoch, loss_contrastive.item()))
        loss.append(loss_contrastive.item())
    
    return net

# set the device to cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Start training")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
model = train()
torch.save(model.state_dict(), "model.pt")
print("Model Saved Successfully")
