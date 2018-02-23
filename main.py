import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from src.ixmas_dataset import IXMASDataset
from src.tcn_3cd import C3DTCN


def main():

    path = os.path.abspath("../")
    cuda = torch.cuda.is_available()

    transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(200)])

    data_set = IXMASDataset(path, ["julien1", "alba1", "alba2"], transform=transform)
    data_loader = DataLoader(
        dataset=data_set,
        batch_size=1,
        shuffle=True,
        pin_memory=cuda
    )

    tcn = C3DTCN()

    if cuda:
        tcn = tcn.cuda()

    optimizer = optim.SGD(tcn.parameters(), lr=0.01, momentum=0.9)
    learning_rate_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[100, 500, 1000], gamma=0.1)

    for i in range(25):
        print("Epoch {}:".format(i))
        losses = []
        learning_rate_scheduler.step()

        for i in range(5):
            for batch in data_loader:
                frames = Variable(batch)

                if cuda:
                    frames = frames.cuda()

                anchor = frames[:, 0, :, :, :]
                positive = frames[:, 1, :, :, :]
                negative = frames[:, 2, :, :, :]

                anchor_results = tcn(anchor)
                positive_results = tcn(positive)
                negative_results = tcn(negative)

                positive_distance = euclidean(anchor_results, positive_results)
                negative_distance = euclidean(anchor_results, negative_results)

                loss = torch.clamp(2.0 + positive_distance - negative_distance, 0.0).mean()
                losses.append(loss)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    x = [i for i in range(0, 25)]
    plt.plot(x, loss)
    plt.show()

def euclidean(a, b):
    dist = torch.pow(torch.abs(a - b), 2).sum(dim=1)
    return dist


if __name__ == '__main__':
    main()
