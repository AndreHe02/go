from torch import nn
import torch.nn.functional as F

class BaseBoardEncoder(nn.Module):

    def __init__(self, board_size=19, in_channel=2, num_channels=256, num_layers=4, out_dim=256):
        super(BaseBoardEncoder, self).__init__()
        self.board_size, self.num_layers = board_size, num_layers
        self.in_channel, self.num_channels, self.out_dim = in_channel, num_channels, out_dim
        self.lyrs = nn.ModuleList()
        self.init_params()

    def init_params(self):
        # self.num_layers of convolution layers
        for lyr_idx in range(self.num_layers):
            in_dimension = self.in_channel if lyr_idx == 0 else self.num_channels
            lyr = nn.Sequential(
                nn.Conv2d(in_dimension, self.num_channels, kernel_size=3, stride=1, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            self.lyrs.append(lyr)

        # fully connected layer
        self.fc = nn.Linear(in_features=self.num_channels, out_features=self.out_dim)

        # non-linear activation for input feature
        self.final_nonlin_act = nn.Tanh()

    def forward(self, board):
        out = board
        # passing through the convolution layers
        for lyr in self.lyrs:
            out = lyr(out)

        # global max pool after convolution
        # out_dim = batch_size x num_channels
        out = F.max_pool2d(out, kernel_size=out.size()[2]).squeeze()

        # fully connected + activation
        out = self.fc(out)
        out = self.final_nonlin_act(out)

        # out_dim = batch_size x out_dimension
        return out



