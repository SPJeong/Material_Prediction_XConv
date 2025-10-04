##### model.py (XConv)

import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.nn import XConv


class XConv_Model(nn.Module):
    def __init__(self, in_channels=57, out_channels=1024, hidden_dim=300, descriptor_ecfp2_size=1217, dropout=0.2):
        super().__init__()

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Define the arguments for the XConv layers
        kernel_size = 3
        dim = 3  # for 3D coordinates

        # XConv layers
        # The first layer needs `in_channels` as input and will output `in_channels * 2`
        self.conv1 = XConv(in_channels, in_channels * 2, dim, kernel_size)
        self.bn1 = torch.nn.BatchNorm1d(in_channels * 2)

        # The second layer takes `in_channels * 2` and outputs `in_channels * 4`
        self.conv2 = XConv(in_channels * 2, in_channels * 4, dim, kernel_size)
        self.bn2 = torch.nn.BatchNorm1d(in_channels * 4)

        # The third layer takes `in_channels * 4` and outputs `hidden_dim`
        self.conv3 = XConv(in_channels * 4, hidden_dim, dim, kernel_size)
        self.bn3 = torch.nn.BatchNorm1d(hidden_dim)

        # The fourth layer takes `in_channels * 4` and outputs `hidden_dim`
        self.conv4 = XConv(hidden_dim, hidden_dim, dim, kernel_size)
        self.bn4 = torch.nn.BatchNorm1d(hidden_dim)

        # xconv last layer
        self.xconv_fc = nn.Linear(hidden_dim, out_channels)

        # final FC
        self.final_fc1 = nn.Linear(out_channels + descriptor_ecfp2_size, 2048)  # (1024+1217) -> 1024
        self.final_fc2 = nn.Linear(2048, 1024)
        self.final_fc3 = nn.Linear(1024, 512)
        self.final_fc4 = nn.Linear(512, 256)
        self.final_fc5 = nn.Linear(256, 1)

    def forward(self, data_graph, data_descriptor_ECFP):
        x, pos, batch = data_graph.x, data_graph.pos, data_graph.batch

        # XConv forward
        x = self.relu(self.bn1(self.conv1(x, pos)))
        x = self.relu(self.bn2(self.conv2(x, pos)))
        x = self.relu(self.bn3(self.conv3(x, pos)))
        x = self.relu(self.bn4(self.conv4(x, pos)))

        # Global pooling (can choose one btween two)
        x = torch_geometric.nn.global_add_pool(x, batch)  # better for molecule's property prediction
        # x = torch_geometric.nn.global_mean_pool(x, batch)

        # XConv fc
        x = self.xconv_fc(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)

        # concat XConv + descriptor_ECFP
        combined_features = torch.cat([x, data_descriptor_ECFP], dim=1)

        # final FC
        out = self.relu(self.final_fc1(combined_features))
        out = self.dropout(out)
        out = self.relu(self.final_fc2(out))
        out = self.dropout(out)
        out = self.relu(self.final_fc3(out))
        out = self.dropout(out)
        out = self.relu(self.final_fc4(out))
        out = self.final_fc5(out)

        return out


def count_parameters(model):
    """
    Calculates the total number of trainable parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Calculate and print the total parameters
if __name__ == '__main__':
    my_deep_model = XConv_Model()
    total_params = count_parameters(my_deep_model)
    print(f"Total trainable parameters: {total_params:,}")