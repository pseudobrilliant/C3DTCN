import os
import torch
import gc
import time
import matplotlib.pyplot as plt
from torch import load, save
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from .util import download_url
from .util import euclidean
from .ixmas_dataset import IXMASDataset
from datetime import timedelta


class C3DTCN(nn.Module):

    def __init__(self, pretrain=True):
        super(C3DTCN, self).__init__()

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.fc1 = nn.Linear(8192, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 512)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

        if pretrain:
            self.c3d_pretrained_weights()

    def forward(self, x):

        l = self.relu(self.conv1(x))
        l = self.pool1(l)

        l = self.relu(self.conv2(l))
        l = self.pool2(l)

        l = self.relu(self.conv3a(l))
        l = self.relu(self.conv3b(l))
        l = self.pool3(l)

        l = self.relu(self.conv4a(l))
        l = self.relu(self.conv4b(l))
        l = self.pool4(l)

        l = self.relu(self.conv5a(l))
        l = self.relu(self.conv5b(l))
        l = self.pool5(l)

        l = l.view(-1, 8192)
        l = self.relu(self.fc1(l))
        l = self.dropout(l)
        l = self.relu(self.fc2(l))
        l = self.dropout(l)
        l = self.relu(self.fc3(l))

        return l

    def c3d_pretrained_weights(self):
        if not os.path.exists("./dataset/c3d.pickle"):
            print("-----Retrieving C3D Pretrained Weights-----")
            download_url("https://pseudobrilliant.com/files/c3d.pickle", "./dataset/c3d.pickle")
            print("-----Completed C3D Pretrained Weights-----")

        pretrained = load("./dataset/c3d.pickle")
        pretrained.pop("fc6.weight")
        pretrained.pop("fc6.bias")
        pretrained.pop("fc7.weight")
        pretrained.pop("fc7.bias")
        pretrained.pop("fc8.weight")
        pretrained.pop("fc8.bias")

        new_params = self.state_dict()
        new_params.update(pretrained)
        self.load_state_dict(new_params)

    def save_model(self, path):
        if not os.path.exists("./saves"):
            os.mkdir("./saves")
        save(self, "./saves/"+path)

    @staticmethod
    def load_tcn(tcn_settings):

        tcn = C3DTCN(pretrain=False)
        tcn.cuda()
        data = torch.load("./saves/tcnc3d.pt")
        tcn.load_state_dict(data)

        return tcn

    @staticmethod
    def tcn_batch(batch, tcn, cuda):

        frames = Variable(batch)

        if cuda:
            frames = frames.cuda()

        anchor_results = tcn(frames[:, 0, :, :, :])
        positive_results = tcn(frames[:, 1, :, :, :])
        negative_results = tcn(frames[:, 2, :, :, :])

        positive_distance = euclidean(anchor_results, positive_results)
        negative_distance = euclidean(anchor_results, negative_results)

        return positive_distance, negative_distance

    @staticmethod
    def validate_tcn(tcn_settings, tcn, dataset, verbose=False):
        cuda = torch.cuda.is_available()

        batch_size = int(tcn_settings["batch"])
        margin = float(tcn_settings["margin"])

        data_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=cuda
        )

        torch.cuda.empty_cache()
        gc.collect()

        num_correct = 0
        for batch in data_loader:

            batch_positive, batch_negative = C3DTCN.tcn_batch(batch, tcn, cuda)

            loss = torch.clamp(margin + batch_positive - batch_negative, min=0.0).mean()
            num_correct += (loss <= 0.0).data.cpu().numpy().sum()

            del batch_positive, batch_negative, loss
            torch.cuda.empty_cache()
            gc.collect()

        loss_validation = (1.0 - float(num_correct) / len(dataset))

        return loss_validation

    @staticmethod
    def train_tcn(tcn_settings, verbose=False):
        cuda = torch.cuda.is_available()

        transform = transforms.Compose([transforms.CenterCrop(224), transforms.Resize(56), transforms.ToTensor()])

        training_collections = tcn_settings["training_collections"].split(",")
        path = os.path.abspath("./")

        training = IXMASDataset(path, training_collections, transform=transform, verbose=False)
        training.set_triplet_flag(True)

        validation_collection = tcn_settings["validation_collection"]
        validation = IXMASDataset(path, [validation_collection], transform=transform, verbose=False)
        validation.set_triplet_flag(True)

        learning = float(tcn_settings["learning"])
        momentum = float(tcn_settings["momentum"])
        batch_size = int(tcn_settings["batch"])
        epochs = int(tcn_settings["epochs"])
        margin = float(tcn_settings["margin"])

        print("-----Starting TCN Training-----")
        data_loader = DataLoader(
            dataset=training,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=cuda
        )

        tcn = C3DTCN()

        if cuda:
            tcn = tcn.cuda()

        optimizer = optim.SGD(tcn.parameters(), lr=learning, momentum=momentum)
        learning_rate_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[5, 15, 20], gamma=0.1)

        historical = []
        accuracy = []
        total_time = 0

        for i in range(epochs):
            print("Epoch {}:".format(i))
            start_time = time.time()
            losses = []
            learning_rate_scheduler.step()

            for batch in data_loader:

                batch_positive, batch_negative = C3DTCN.tcn_batch(batch, tcn, cuda)

                loss = torch.clamp(margin + batch_positive - batch_negative, min=0.0).mean()

                loss_data = loss.data.cpu().numpy()[0]
                if verbose:
                    print("\tPositive Distance: " + str(batch_positive.data.cpu().numpy()[0]))
                    print("\tNegative Distance: " + str(batch_negative.data.cpu().numpy()[0]))
                    print("\tLoss: " + str(loss_data))
                losses.append(loss_data)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                del loss, batch_positive, batch_negative
                torch.cuda.empty_cache()
                gc.collect()

            tcn.save_model("tcnc3d.pt")
            mean_loss = np.mean(losses)
            print("\tDistance Loss: " + str(mean_loss))
            historical.append(mean_loss)

            val_loss = C3DTCN.validate_tcn(tcn_settings, tcn, validation, verbose)
            print("\tValidation Loss: " + str(val_loss))
            accuracy.append(val_loss)

            end_time = time.time() - start_time
            total_time += end_time
            print("\tEpoch Time:{}".format(timedelta(seconds=end_time)))

            if i+1 % 5 == 0 and i != 0:
                x = [j for j in range(0, i+1)]
                plt.title("Iterations vs Average Distance Loss")
                plt.xlabel("Iterations")
                plt.ylabel("Distance Loss")
                plt.plot(x, historical, marker='o')
                plt.show()

                plt.title("Iterations vs Validation Loss")
                plt.xlabel("Iterations")
                plt.ylabel("Validation Loss")
                plt.plot(x, accuracy, marker='o')
                plt.show()

        print("Total Time: {}".format(timedelta(seconds=total_time)))
        print("----Completed TCN Training-----")

        return tcn


