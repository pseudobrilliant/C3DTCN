import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from src.ixmas_dataset import IXMASDataset

def main():
    path = os.path.abspath("../")
    cuda = torch.cuda.is_available()
    data_set = IXMASDataset(path, ["julien1", "alba1", "alba2"])
    data_loader = DataLoader(
        dataset=data_set,
        batch_size=10,
        shuffle=True,
        pin_memory=cuda
    )

    for batch in data_loader:
        frames = Variable(batch)
        print(frames)


if __name__ == '__main__':
    main()