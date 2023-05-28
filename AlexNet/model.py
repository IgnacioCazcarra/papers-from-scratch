from torch import nn 
import torch.nn.functional as F


class AlexNet(nn.Module):

    def __init__(self):
        super(AlexNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3)
        
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.response_normalization = ResponseNormalizationLayer()
        self.dropout = nn.Dropout(p=0.5)

        self.fc1 = nn.Linear(4096, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 1000)

        # Init weights
        self.apply(self.init_weights)
        # Override weights of the other layers. Conv 1 and 3 should have bias 0
        self.conv1.bias.data.fill_(0)
        self.conv3.bias.data.fill_(0)


    def init_weights(m):
        nn.init.normal_(m.weight, mean=0.0, std=0.01)
        m.bias.data.fill_(1)


    def forward(self, x):
        # CNN
        x = self.pool(self.response_normalization(F.relu(self.conv1(x))))
        x = self.pool(self.response_normalization(F.relu(self.conv2(x))))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(F.relu(self.conv5(x)))
        # FC
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return F.softmax(x)


class ResponseNormalizationLayer(nn.Module):
    def __init__(self, k=2, n=5, alpha=10e-4, beta=0.75):
        super(ResponseNormalizationLayer, self).__init__()
        self.k = k
        self.n = n
        self.alpha = alpha
        self.beta = beta

    def forward(self, x):
        squared = x.pow(2)
        sum_squared = F.avg_pool2d(squared, self.n, stride=1, padding=self.n // 2)
        divider = sum_squared.mul(self.alpha).add(self.k).pow(self.beta)
        return x / divider