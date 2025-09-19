import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes, num_layers=2, dropout_rate=0.1, fc_layer_channels=0):
        super(GCN, self).__init__()
        self.num_layers = num_layers
        self.fc_layer_channels = fc_layer_channels

        self.conv_layers = torch.nn.ModuleList()
        self.conv_layers.append(GCNConv(num_features, hidden_channels))  # Input layer
        self.conv_layers.append(torch.nn.Dropout(p=dropout_rate))

        for _ in range(num_layers - 2):  # We already have one layer above
            self.conv_layers.append(GCNConv(hidden_channels, hidden_channels))  # Hidden layers
            self.conv_layers.append(torch.nn.Dropout(p=dropout_rate))  # Dropout

        # Determine output size of the last GCN layer
        final_gcn_output_size = hidden_channels if self.fc_layer_channels != 0 else num_classes
        self.conv_layers.append(GCNConv(hidden_channels, final_gcn_output_size))  # Last GCN layer

        # Add a fully connected layer if `fc_layer_channels` is set
        if self.fc_layer_channels != 0:
            self.fc_layer = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        for i in range(self.num_layers - 1):  # Apply all layers except the last one
            x = self.conv_layers[i * 2](x, edge_index)  # Get the GCNConv layer
            x = F.relu(x)
            x = self.conv_layers[i * 2 + 1](x)  # Get the Dropout layer

        x = self.conv_layers[-1](x, edge_index)  # Apply the last GCNConv layer

        if self.fc_layer_channels != 0:
            x = self.fc_layer(x)  # Apply the fully connected layer if it exists

        return F.log_softmax(x, dim=1)