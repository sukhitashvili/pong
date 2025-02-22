import torch.nn as nn


class MoveClassifier(nn.Module):
    def __init__(self, num_features=6400, hidden_size=512):
        super().__init__()
        self.linear1 = nn.Linear(num_features, hidden_size, bias=False)
        self.leaky_relu = nn.LeakyReLU()
        self.linear2 = nn.Linear(hidden_size, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        out = self.linear1(X)
        out = self.leaky_relu(out)
        out = self.linear2(out)
        out = self.sigmoid(out)
        return out
