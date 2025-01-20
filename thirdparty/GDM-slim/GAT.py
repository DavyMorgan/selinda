#   This file is part of GDM (Graph Dismantling with Machine learning),
#   proposed in the paper "Machine learning dismantling and
#   early-warning signals of disintegration in complex systems"
#   by M. Grassia, M. De Domenico and G. Mangioni.
#
#   GDM is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   GDM is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with GDM.  If not, see <http://www.gnu.org/licenses/>.

# from collections import defaultdict

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_geometric.nn import GATConv


class GAT_Model(nn.Module):

    def __init__(self):

        super(GAT_Model, self).__init__()

        self.features = ['chi_degree', 'clustering_coefficienct', 'degree', 'k_core']
        self.num_features = len(self.features)
        self.conv_layers = [20, 20, 20, 20]
        self.heads = [1, 1, 1, 1]
        self.fc_layers = [40, 30, 20, 1]
        self.concat = [True, True, True, True]
        self.negative_slope = [0.2, 0.2, 0.2, 0.2]
        self.dropout = [0.3, 0.3, 0.3, 0.3]
        self.bias = [True, True, True, True]
        self.seed_train = 0

        self.convolutional_layers = torch.nn.ModuleList()
        self.linear_layers = torch.nn.ModuleList()
        self.fullyconnected_layers = torch.nn.ModuleList()

        # TODO support non constant concat values
        for i in range(len(self.conv_layers)):
            num_heads = self.heads[i - 1] if ((self.concat[i - 1] is True) and (i > 0)) else 1
            in_channels = self.conv_layers[i - 1] * num_heads if i > 0 else self.num_features
            self.convolutional_layers.append(
                GATConv(in_channels=in_channels,
                        out_channels=self.conv_layers[i],
                        heads=self.heads[i],
                        concat=self.concat[i],
                        negative_slope=self.negative_slope[i],
                        dropout=self.dropout[i],
                        bias=self.bias[i])
            )

            num_out_heads = self.heads[i] if self.concat[i] is True else 1
            self.linear_layers.append(
                torch.nn.Linear(in_features=in_channels, out_features=self.conv_layers[i] * num_out_heads)
            )

        # Regressor

        # If last layer output is not a regressor, append a layer
        if self.fc_layers[-1] != 1:
            self.fc_layers.append(1)

        for i in range(len(self.fc_layers)):
            num_heads = self.heads[-1] if ((self.concat[-1] is True) and (i == 0)) else 1
            in_channels = self.fc_layers[i - 1] if i > 0 else self.conv_layers[-1] * num_heads
            self.fullyconnected_layers.append(
                torch.nn.Linear(in_features=in_channels, out_features=self.fc_layers[i])
            )

    def forward(self, x, edge_index):

        for i in range(len(self.convolutional_layers)):
            x = F.elu(self.convolutional_layers[i](x, edge_index) + self.linear_layers[i](x))

        x = x.view(x.size(0), -1)
        for i in range(len(self.fullyconnected_layers)):
            # TODO ELU?
            x = F.elu(self.fullyconnected_layers[i](x))

        x = x.view(x.size(0))
        x = torch.sigmoid(x)
        # print(x.size())
        return x
