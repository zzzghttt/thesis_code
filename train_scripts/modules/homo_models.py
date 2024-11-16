import torch
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, Linear

class Graph_Model(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, out_channels, num_layers, num_heads=4, model_type="GAT"):
        super().__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers

        # 卷积层
        self.layer_convs = torch.nn.ModuleList()
        
        # 异质卷积
        for _ in range(num_layers):
            if (model_type == "GAT"):
                conv = GATConv(in_channels=-1, out_channels=self.hidden_channels, heads=num_heads)
            if (model_type == "SAGE"):
                conv = SAGEConv(in_channels=-1, out_channels=self.hidden_channels)
            if (model_type == "GCN"):
                conv = GCNConv(in_channels=-1, out_channels=self.hidden_channels)
            self.layer_convs.append(conv)
        
        # 输出层MLP
        self.output_layer_lin1 = Linear(-1, self.hidden_channels)
        self.output_layer_lin2 = Linear(self.hidden_channels, self.out_channels)
        self.sigmoid = torch.nn.Sigmoid()

    # 模型前向计算方法定义
    def forward(self, x, edge_index,):
        # 原始输入
        x_o = x

        # 逐层卷积
        for conv in self.layer_convs:
            x = conv(x, edge_index)

        # 输出变换
        return self.sigmoid(
            self.output_layer_lin2(
                self.output_layer_lin1(
                    torch.concat((x, x_o), dim=-1)
                    ).relu()
                )
            )