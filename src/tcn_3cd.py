import os
import torch
import gc
import time
import warnings
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
from .util import get_config_list
from .ixmas_dataset import IXMASDataset
from datetime import timedelta


class C3DTCN(nn.Module):

    def __init__(self, tcn_settings, verbose=False):
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
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, 32)

        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

        self.is_cuda = torch.cuda.is_available()

        self.verbose = verbose
        self.training = tcn_settings.getboolean("train")

        self.pretrain = tcn_settings.getboolean("c3d_pretrained")
        if self.pretrain:
            self.c3d_pretrained_weights()

        if self.training:
            self.training_collections = get_config_list(tcn_settings["training_collections"])
            self.validation_collection = get_config_list(tcn_settings["validation_collection"])
            self.learning = float(tcn_settings["learning"])
            self.frames = int(tcn_settings["num_frames"])
            self.momentum = float(tcn_settings["momentum"])
            self.batch_size = int(tcn_settings["batch"])
            self.epochs = int(tcn_settings["epochs"])
            self.margin = float(tcn_settings["margin"])

            self.train()
        else:
            self.saved_model = tcn_settings["saved_model"]
            self.load_tcn()

    def forward(self, x):

        h = self.relu(self.conv1(x))
        h = self.pool1(h)

        h = self.relu(self.conv2(h))
        h = self.pool2(h)

        h = self.relu(self.conv3a(h))
        h = self.relu(self.conv3b(h))
        h = self.pool3(h)

        h = self.relu(self.conv4a(h))
        h = self.relu(self.conv4b(h))
        h = self.pool4(h)

        h = self.relu(self.conv5a(h))
        h = self.relu(self.conv5b(h))
        h = self.pool5(h)

        h = h.view(-1, 8192)
        h = self.relu(self.fc6(h))
        h = self.dropout(h)
        h = self.relu(self.fc7(h))
        h = self.dropout(h)
        h = self.fc8(h)

        return h

    def c3d_pretrained_weights(self):
        if not os.path.exists("./dataset/c3d.pickle"):
            print("-----Retrieving C3D Pretrained Weights-----")
            download_url("https://pseudobrilliant.com/files/c3d.pickle", "./dataset/c3d.pickle")
            print("-----Completed C3D Pretrained Weights-----")

        pretrained = load("./dataset/c3d.pickle")
        pretrained.pop("fc8.weight")
        pretrained.pop("fc8.bias")

        new_params = self.state_dict()
        new_params.update(pretrained)
        self.load_state_dict(new_params)

    def save_model(self, temp_epoch=None):
        if not os.path.exists("./saves"):
            os.mkdir("./saves")

        if temp_epoch is None:
            temp_epoch = self.epochs

        learning_string = str(self.learning)
        learning_string = learning_string[learning_string.find('.') + 1:]

        margin_string = str(self.margin)
        margin_string = margin_string[margin_string.find('.') + 1:]

        filename = "tcn_epochs_{}_batch_{}_frames_{}_learning_{}_margin_{}.pt".format(temp_epoch, self.batch_size,
                                                                                      self.frames, learning_string,
                                                                                      margin_string)

        save(self, "./saves/"+filename)

    def load_tcn(self):

        path = "./saves/"+self.saved_model
        if not os.path.exists(path):
            raise ValueError("Saved model not provided, unable to load tcn!")
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                data = load(path)
                self.load_state_dict(data.state_dict())


    def tcn_batch(self, batch):

        frames = Variable(batch)

        if self.is_cuda:
            frames = frames.cuda()

        anchor_results = self.forward(frames[:, 0, :, :, :])
        positive_results = self.forward(frames[:, 1, :, :, :])
        negative_results = self.forward(frames[:, 2, :, :, :])

        positive_distance = euclidean(anchor_results, positive_results)
        negative_distance = euclidean(anchor_results, negative_results)

        return positive_distance, negative_distance

    def validate(self, dataset):

        data_loader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.is_cuda
        )

        torch.cuda.empty_cache()
        gc.collect()

        total = 0
        num_correct = 0
        for batch in data_loader:

            batch_positive, batch_negative = self.tcn_batch(batch)

            loss = torch.clamp(self.margin + batch_positive - batch_negative, min=0.0)
            num_correct += (loss <= 0.0).data.cpu().numpy().sum()
            total += batch.size(0)

            del batch_positive, batch_negative, loss
            torch.cuda.empty_cache()
            gc.collect()

        loss_validation = (1.0 - (float(num_correct) / total))

        return loss_validation

    def train(self):
        transform = transforms.Compose([transforms.CenterCrop(224), transforms.Resize(112), transforms.ToTensor()])
        path = os.path.abspath("./")

        training = IXMASDataset(path, self.training_collections, transform=transform, verbose=False)
        training.set_triplet_flag(True)

        validation = IXMASDataset(path, self.validation_collection, transform=transform, verbose=False)
        validation.set_triplet_flag(True)

        print("-----Starting TCN Training-----")
        data_loader = DataLoader(
            dataset=training,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=self.is_cuda
        )

        if self.is_cuda:
            self.cuda()

        optimizer = optim.Adam(self.parameters(), lr=self.learning)

        historical = []
        accuracy = []
        total_time = 0

        for i in range(self.epochs):
            print("Epoch {}:".format(i))
            start_time = time.time()
            losses = []

            for batch in data_loader:

                batch_positive, batch_negative = self.tcn_batch(batch)

                loss = torch.clamp(self.margin + batch_positive - batch_negative, min=0.0).mean()

                loss_data = loss.data.cpu().numpy()[0]
                if self.verbose:
                    print("\tPositive Distance: " + str(batch_positive.data.cpu().numpy()))
                    print("\tNegative Distance: " + str(batch_negative.data.cpu().numpy()))
                    print("\tLoss: " + str(loss_data))
                losses.append(loss_data)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                del loss, batch_positive, batch_negative
                torch.cuda.empty_cache()
                gc.collect()

            mean_loss = np.mean(losses)
            print("\tDistance Loss: " + str(mean_loss))
            historical.append(mean_loss)

            val_loss = self.validate(validation)
            print("\tValidation Loss: " + str(val_loss))
            accuracy.append(val_loss)

            end_time = time.time() - start_time
            total_time += end_time
            print("\tEpoch Time:{}".format(timedelta(seconds=end_time)))

            next_epoch = i + 1
            if next_epoch % 5 == 0 and next_epoch != 0:
                self.save_model(temp_epoch=next_epoch)

                x = [j for j in range(0, next_epoch)]
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



