import constants
from utils.data_loader import DataLoader4Detector
from utils.callbacks import Logger, LrStepDecay
from vehicle_detection import detector
import tensorflow.keras as tfk
import matplotlib.pyplot as plt
import numpy as np

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as utils
import matplotlib.pyplot as plt
import torchvision.utils
from functools import reduce
from operator import __add__

def compute_padding(k):
    return (k - 1) // 2


class TorchDetector(nn.Module):
    def __init__(self):
        super(TorchDetector, self).__init__()

        # Setting up the Sequential of CNN Layers
        self.cnn1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=7, stride=1, padding=compute_padding(7)),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=compute_padding(3)),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=compute_padding(3)),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=compute_padding(3)),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=compute_padding(3)),
            nn.MaxPool2d(2, stride=2)
        )

        # Defining the fully connected layers
        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(92416, 20),
            nn.ReLU(inplace=True),
            nn.Linear(20, 1), 
            nn.Sigmoid()
        )

    def forward(self, x):
        # Forward pass
        output = self.cnn1(x)
        # output = torch.reshape(output, (output.size(0), -1))
        output = self.fc1(output)
        return output


def prepare_data(imgs, labels):
    imgs, labels = imgs.numpy(), labels.numpy()
    imgs = np.transpose(imgs, axes=(0, -1, 1, 2))
    labels = np.reshape(labels, (labels.shape[0], 1))
    imgs, labels = torch.from_numpy(imgs), torch.from_numpy(labels)
    imgs, labels = imgs.cuda(), labels.cuda()
    return imgs, labels


EPOCHS = 2
BATCH_SIZE = 32
data_root = "/content/drive/MyDrive/21Spring/COMS 6998/Project/6998-sec11-final-project/CNR-EXT-Patches-150x150/"
annotation_files = data_root + 'LABELS/all.txt'
model = TorchDetector().cuda()
model.train()
# model = TorchDetector().cuda()
# model.load_state_dict(torch.load("torch_detector_retrained.pt"))
# model.eval()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=0.0005)
criterion = nn.BCELoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1, \
                                                verbose=True)
print(model)

data_loader = DataLoader4Detector(data_root,
                                  annotation_files)
train, val = data_loader.load(batch_size=BATCH_SIZE, repeat=True)
train_data = []
i = 0

for epoch in range(1, EPOCHS + 1):
    running_valloss, running_valacc = 0.0, 0.0
    for idx, (imgs, labels) in enumerate(train.take(6000)):
        imgs, labels = prepare_data(imgs, labels)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        model.eval()
        for _, (val_imgs, val_labels) in enumerate(val.take(1), i):
            val_imgs, val_labels = prepare_data(val_imgs, val_labels)

            pred_labels = model(val_imgs)
            val_loss = criterion(pred_labels, val_labels)

            pred_labels, val_labels = pred_labels.cpu().detach().numpy(), \
                          val_labels.cpu().detach().numpy()
            pred_labels = np.around(pred_labels)
            val_acc = (pred_labels == val_labels).mean()
            running_valacc += val_acc
            running_valloss += val_loss.item()
        model.train()

        if idx % 200 == 199:
            print(idx, running_valloss / 200.0, running_valacc / 200.0)
            running_valloss = 0.0
            running_valacc = 0.0

        i += 1

    print("Epoch {} Current loss {}".format(epoch, loss.item()))
    scheduler.step()
    i = 0

torch.save(model.state_dict(), "torch_detector_retrained.pt")
print("Model Saved Successfully")

# running_acc = 0.0
# with torch.no_grad():
#     for idx, (imgs, labels) in enumerate(train, 7000):
#         imgs, labels = prepare_data(imgs, labels)

#         imgs = imgs.cpu().detach().numpy()
#         example = np.transpose(imgs[0], axes=(1, 2, 0)) + 0.5
#         print(example)
#         plt.imshow(example)
#         plt.show()
#         break

#         pred_labels = model(imgs)
#         pred_labels, labels = pred_labels.cpu().detach().numpy(), \
#                                 labels.cpu().detach().numpy()
#         pred_labels = np.around(pred_labels)
#         acc = (pred_labels == labels).mean()
#         print(acc)
#         running_acc += acc
#         if idx % 200 == 199:
#           print(idx, running_acc / 200.0)
#           running_acc = 0.0
