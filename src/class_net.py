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
from .ixmas_dataset import IXMASDataset
from .util import get_config_list
from datetime import timedelta
import warnings


class ClassNet(nn.Module):

    def __init__(self, cnet_settings, tcn, class_size=15, verbose=False):
        super(ClassNet, self).__init__()

        self.class_size = class_size
        self.layer1 = nn.Linear(32, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, class_size)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        self.tcn = tcn

        self.is_cuda = torch.cuda.is_available()

        self.training = cnet_settings.getboolean("train")
        if self.training:
            self.training_collections = get_config_list(cnet_settings["training_collections"])
            self.validation_collection = get_config_list(cnet_settings["validation_collection"])
            self.learning = float(cnet_settings["learning"])
            self.momentum = float(cnet_settings["momentum"])
            self.batch_size = int(cnet_settings["batch"])
            self.frames = int(cnet_settings["num_frames"])
            self.epochs = int(cnet_settings["epochs"])
            self.train_cnet()
        else:
            self.saved_model = cnet_settings["saved_model"]
            self.load_cnet()


    def forward(self, layer):
        layer = self.layer1(layer)
        layer = self.relu(layer)
        layer = self.dropout(layer)
        layer = self.layer2(layer)
        layer = self.relu(layer)
        layer = self.dropout(layer)
        layer = self.layer3(layer)
        predictions = self.softmax(layer)

        return predictions

    def save_model(self, temp_epoch=None):
        if not os.path.exists("./saves"):
            os.mkdir("./saves")

        if temp_epoch is None:
            temp_epoch = self.epochs

        learning_string = str(self.learning)
        learning_string = learning_string[learning_string.find('.'):]

        filename = "cnet_epochs_{}_batch_{}_frames_{}_learning_{}.pt".format(temp_epoch, self.batch_size,
                                                                             self.frames, learning_string)

        save(self, "./saves/"+filename)

    def load_net(self):

        path = "./saves/"+self.saved_model
        if not os.path.exists(path):
            raise ValueError("Saved model not provided, unable to load cnet!")
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                data = load(path)
                self.load_state_dict(data.state_dict())

    def process_batch(self, batch):

        frames = Variable(batch)

        if self.is_cuda:
            frames = frames.cuda()

        tcn_embedding = self.tcn(frames)
        batch_results = self(tcn_embedding)

        del frames
        torch.cuda.empty_cache()
        gc.collect()

        return batch_results

    def validate_cnet(self, dataset):

        data_loader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=self.is_cuda
        )

        total = 0
        num_correct = 0
        for batch, labels in data_loader:

            output = self.process_batch(batch)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            num_correct += (predicted.cpu() == labels).sum()

            del output
            torch.cuda.empty_cache()
            gc.collect()

        loss_validation = (1.0 - (float(num_correct) / total))
        return loss_validation

    def train_cnet(self):
        transform = transforms.Compose([transforms.CenterCrop(224), transforms.Resize(112), transforms.ToTensor()])
        path = os.path.abspath("./")


        training = IXMASDataset(path, self.training_collections, transform=transform, verbose=False)
        training.set_triplet_flag(False)

        validation = IXMASDataset(path, self.validation_collection, transform=transform, verbose=False)
        validation.set_triplet_flag(False)

        print("-----Starting CNET Training-----")
        data_loader = DataLoader(
            dataset=training,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=self.is_cuda
        )

        if self.is_cuda:
            self.cuda()

        optimizer = optim.Adam(self.parameters(), lr=self.learning)
        classifier = nn.CrossEntropyLoss()

        historical = []
        accuracy = []
        total_time = 0

        for i in range(self.epochs):
            print("Epoch {}:".format(i))
            start_time = time.time()
            losses = []

            for (batch, labels) in data_loader:

                output = self.process_batch(batch)

                labels_tensor = Variable(labels).cuda()

                optimizer.zero_grad()
                loss = classifier(output, labels_tensor)
                loss_data = loss.data.cpu().numpy()[0]
                losses.append(loss_data)
                loss.backward()
                optimizer.step()

                del output
                torch.cuda.empty_cache()
                gc.collect()

            mean_loss = np.mean(losses)
            print("\tDistance Loss: " + str(mean_loss))
            historical.append(mean_loss)

            val_loss = self.validate_cnet(validation)
            print("\tValidation Loss: " + str(val_loss))
            accuracy.append(val_loss)

            end_time = time.time() - start_time
            total_time += end_time
            print("\tEpoch Time:{}".format(timedelta(seconds=end_time)))

            next_epoch = i + 1
            if next_epoch % 5 == 0 and next_epoch != 0:
                self.save_model()

                x = [j for j in range(0, next_epoch)]
                plt.title("Iterations vs Loss")
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
        print("----Completed CNET Training-----")