import torch
from prompt_graph.model import GAT, GCN, GCov, GIN, GraphSAGE, GraphTransformer


class PreTrain(torch.nn.Module):
    def __init__(self, gnn_type='TransformerConv', dataset_name = 'Cora', hid_dim=256, out_dim=128, gln=2, num_epoch=100, num_heads=1, device : int=5):
        super().__init__()
        self.device = torch.device('cuda:' + str(device) if torch.cuda.is_available() else 'cpu')
        self.dataset_name = dataset_name
        self.gnn_type = gnn_type
        self.num_layer = gln
        self.epochs = num_epoch
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
    
    def initialize_gnn(self, input_dim, hid_dim, out_dim):
        if self.gnn_type == 'GAT':
                self.gnn = GAT(input_dim=input_dim, hid_dim=hid_dim, out_dim=out_dim, num_layer=self.num_layer, heads=self.num_heads)
        elif self.gnn_type == 'GCN':
                self.gnn = GCN(input_dim=input_dim, hid_dim=hid_dim, out_dim=out_dim, num_layer=self.num_layer)
        elif self.gnn_type == 'GraphSAGE':
                self.gnn = GraphSAGE(input_dim=input_dim, hid_dim=hid_dim, out_dim=out_dim, num_layer=self.num_layer)
        elif self.gnn_type == 'GIN':
                self.gnn = GIN(input_dim=input_dim, hid_dim=hid_dim, out_dim=out_dim, num_layer=self.num_layer)
        elif self.gnn_type == 'GCov':
                self.gnn = GCov(input_dim=input_dim, hid_dim=hid_dim, out_dim=out_dim, num_layer=self.num_layer)
        elif self.gnn_type == 'GraphTransformer':
                self.gnn = GraphTransformer(input_dim=input_dim, hid_dim=hid_dim, out_dim=out_dim, num_layer=self.num_layer)
        else:
                raise ValueError(f"Unsupported GNN type: {self.gnn_type}")
        print(self.gnn)
        self.gnn.to(self.device)

