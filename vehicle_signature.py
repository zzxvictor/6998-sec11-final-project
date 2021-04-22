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
# from vehicle_signature.contrastive import ContrastiveLoss
# from vehicle_signature.siamese import SiameseNetwork

def prepare_data(img1, img2, labels):
  img1, img2, labels = img1.numpy(), img2.numpy(), labels.numpy()
  img1, img2 = np.transpose(img1, axes=(0, -1, 1, 2)), \
                np.transpose(img2, axes=(0, -1, 1, 2))
  labels = np.reshape(labels, (labels.shape[0], 1))
  img1, img2, labels = torch.from_numpy(img1), torch.from_numpy(img2), \
    torch.from_numpy(labels)
  img1, img2, labels = img1.cuda(), img2.cuda(), labels.cuda()
  return img1, img2, labels


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()

        # Setting up the Sequential of CNN Layers
        self.cnn1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=1),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout2d(p=0.3),
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout2d(p=0.3)
        )

        # Defining the fully connected layers
        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(65536, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),
            nn.Linear(1024, 128)
            # nn.ReLU(inplace=True),
            # nn.Linear(128, 2),
        )

    def forward_once(self, x):
        # Forward pass
        output = self.cnn1(x)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        # forward pass of input 1
        output1 = self.forward_once(input1)
        # forward pass of input 2
        output2 = self.forward_once(input2)
        return output1, output2


class ContrastiveLoss(torch.nn.Module):
    "Contrastive loss function"

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = 0.5 * torch.mean(
            (label) * torch.pow(euclidean_distance, 2)
            + (1 - label)
            * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )

        return loss_contrastive


data_root = "/content/drive/MyDrive/21Spring/COMS 6998/Project/6998-sec11-final-project/CNR-EXT-Patches-150x150/"
annotation_files = data_root + 'LABELS/all.txt'
EPOCHS = 3
BATCH_SIZE = 64

# train the model
def train():
    net = SiameseNetwork().cuda()
    print(net)
    criterion = ContrastiveLoss(margin=1.5)
    print(criterion.margin)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5, \
                                                verbose=True)
    
    train_losses, val_losses = [], []
    data_loader = DataLoader4Signature(data_root,
                                   annotation_files)
    train_loader, val = data_loader.load(batch_size=BATCH_SIZE, repeat=True)
    i = 0
    
    for epoch in range(1, EPOCHS + 1):
        running_trainloss, running_valloss = 0.0, 0.0
        for idx, (img1, img2, labels) in enumerate(train_loader):
            img1, img2, labels = prepare_data(img1, img2, labels)
            optimizer.zero_grad()
            output1, output2 = net(img1, img2)
            train_loss = criterion(output1, output2, labels)
            train_losses.append(train_loss.item())
            running_trainloss += train_loss.item()
            train_loss.backward()
            optimizer.step()

            net.eval()
            with torch.no_grad():
                for _, (val_img1, val_img2, val_labels) in enumerate(val.take(1), i):
                    val_img1, val_img2, val_labels = prepare_data(val_img1, \
                                              val_img2, val_labels)
                    val_output1, val_output2 = net(val_img1, val_img2)
                    val_loss = criterion(val_output1, val_output2, val_labels)
                    val_losses.append(val_loss.item())
                    running_valloss += val_loss.item()
            net.train()

            if idx % 200 == 199:
                print("Iteration {}; Train loss: {}; Val loss: {}".format(idx, \
                        running_trainloss / 200.0, running_valloss / 200.0))
                running_trainloss = 0.0
                running_valloss = 0.0
            i += 1
        print("Epoch {} Current loss {}".format(epoch, val_loss.item()))
        scheduler.step()
        i = 0
    
    return net, train_losses, val_losses


print("Start training")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
model, train_losses, val_losses = train()
torch.save(model.state_dict(), "siamese.pt")
print("Model Saved Successfully")
with open("siamese_train_loss.txt", "w+") as filehandle:
    json.dump(train_losses, filehandle)
with open("siamese_val_loss.txt", "w+") as filehandle:
    json.dump(val_losses, filehandle)









# siamese = SiameseNetwork().cuda()
# siamese.load_state_dict(torch.load("siamese.pt"))
# print(siamese)

# data_loader = DataLoader4Signature(data_root,
#                                    annotation_files)
# train_loader, val_loader = data_loader.load(batch_size=32, repeat=True)

# candidates = np.arange(1.30, 1.50001, 0.005)

# for threshold in candidates:
#   print("Current threshold:", threshold)
#   running_acc = 0.0
#   for idx, (img1, img2, labels) in enumerate(val_loader):
#     img1, img2, labels = prepare_data(img1, img2, labels)
#     embed1, embed2 = siamese(img1, img2)
#     embed_diff = embed1 - embed2
#     embed_norm = torch.norm(embed_diff, dim=1)
#     pred_labels = (embed_norm > threshold).float().cpu().detach().numpy()
#     labels = torch.flatten(labels).cpu().detach().numpy()
#     acc = (labels == pred_labels).mean()
#     running_acc += acc
#     if idx % 100 == 99:
#       print(threshold, running_acc / 100.0)
#       running_acc = 0.0


    # for idx, (img1, img2, labels) in enumerate(train_loader, 7000):
    #     img1, img2, labels = prepare_data(img1, img2, labels)

    #     img1 = img1.cpu().detach().numpy()
    #     example = np.transpose(img1[6], axes=(1, 2, 0)) + 0.5
    #     plt.imshow(example)
    #     plt.show()
    #     break