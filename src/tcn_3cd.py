import torch.nn as nn

class C3DTCN(nn.Module):

    def __init__(self):
        super(C3DTCN, self).__init__()

        self.layer1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), padding=(1, 2, 2))

        self.layer2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), padding=(2, 2, 2))

        self.layer3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.layer3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), padding=(2, 2, 2))

        self.layer4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.layer4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), padding=(2, 2, 2))

        self.layer5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.layer5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), padding=(2, 2, 2))

        self.fc1 = nn.Linear(8192, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 487)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):

        l = self.relu(self.layer1(x))
        l = self.pool1(l)

        l = self.relu(self.layer2(l))
        l = self.pool2(l)

        l = self.relu(self.layer3a(l))
        l = self.relu(self.layer3b(l))
        l = self.pool3(l)

        l = self.relu(self.layer4a(l))
        l = self.relu(self.layer4b(l))
        l = self.pool4(l)

        l = self.relu(self.layer5a(l))
        l = self.relu(self.layer5b(l))
        l = self.pool5(l)

        l = l.view(-1,8192)
        l = self.relu(self.fc1(l))
        l = self.dropout(l)
        l = self.relu(self.fc2(l))
        l = self.dropout(l)
        l = self.relu(self.fc3(l))

        return l