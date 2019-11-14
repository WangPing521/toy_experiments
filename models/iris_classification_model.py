import torch.nn as nn


class iris_classifier(nn.Module):
    def __init__(self, num_class=3):
        super(iris_classifier, self).__init__()
        self.conv1 = nn.Linear(4, 3)
        self.conv2 = nn.Linear(3, 3)
        self.outlayer = nn.Softmax(3)

    def forward(self, x):
        out = self.conv1(x)
        out = nn.ReLU(out)
        out = self.conv2(out)
        out = self.outlayer(out)
        return out
