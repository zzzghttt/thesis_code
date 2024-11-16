import torch
from torch_geometric.nn import HGTConv, Linear

class Graph_Model(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, out_channels, num_heads, num_layers, meta_data):
        super().__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.meta_data = meta_data

        # 输入映射层
        self.lin_dict_zero_layer = torch.nn.ModuleDict()
        for node_type in self.meta_data[0]:
            self.lin_dict_zero_layer[node_type] = Linear(-1, self.hidden_channels)

        # 卷积层
        self.layer_convs = torch.nn.ModuleList()
        
        # 异质卷积
        for _ in range(num_layers):
            conv = HGTConv(in_channels=self.hidden_channels, out_channels=self.hidden_channels, metadata=self.meta_data, heads=num_heads)
            self.layer_convs.append(conv)
        
        # 输出层MLP
        self.output_layer_lin1 = Linear(-1, self.hidden_channels)
        self.output_layer_lin2 = Linear(self.hidden_channels, self.out_channels)
        self.sigmoid = torch.nn.Sigmoid()

    # 模型前向计算方法定义
    def forward(self, x_dict, edge_index_dict,):
        # 原始输入
        x_o = x_dict
        # 初始映射
        x_dict = {
            node_type: self.lin_dict_zero_layer[node_type](x).relu()
            for node_type, x in x_dict.items()
        }
        # 逐层卷积
        for conv in self.layer_convs:
            x_dict = conv(x_dict, edge_index_dict)
        # 输出变换
        return self.sigmoid(
            self.output_layer_lin2(
                self.output_layer_lin1(
                    torch.concat((x_dict['uid'], x_o['uid']), dim=-1)
                    ).relu()
                )
            )