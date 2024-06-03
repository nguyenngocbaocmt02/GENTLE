import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
class MLPM(nn.Module):
    def __init__(self, input_dim=3):
        super(MLPM, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256 + input_dim, 256)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(256, 128)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(128 + input_dim, 128)
        self.relu4 = nn.ReLU()
        self.fc5 = nn.Linear(128, 64)
        self.relu5 = nn.ReLU()
        self.fc6 = nn.Linear(64 + input_dim, 64)
        self.relu6 = nn.ReLU()
        self.fc7 = nn.Linear(64, 1)
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                init.zeros_(m.bias)

    def forward(self, input):
        x = self.fc1(input)
        x = self.relu1(x)
        x = self.fc2(torch.cat((x, input), dim=1))
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(torch.cat((x, input), dim=1))
        x = self.relu4(x)
        x = self.fc5(x)
        x = self.relu5(x)
        x = self.fc6(torch.cat((x, input), dim=1))
        x = self.relu6(x)
        x = self.fc7(x)

        return x.squeeze()
