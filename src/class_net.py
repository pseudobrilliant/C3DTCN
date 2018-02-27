import torch.nn as nn

class ClassNet(nn.Module):

    def __init__(self, class_size):
        super(ClassNet, self).__init__()

        self.class_size = class_size
        self.hidden1 = nn.Linear(512, 256, bias=False)
        self.norm1 = nn.BatchNorm1d(256)
        self.out = nn.Linear(256, class_size, bias=False)
        self.relu - nn.ReLU()
        self.logits = nn.Softmax()

    def forward(self, x):
        layer = self.hidden1(x)
        layer = self.relu(layer)
        layer = self.norm1(layer)

        layer = self.logits(layer)
        prob = self.softmax(layer)

        return prob
