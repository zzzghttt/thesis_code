import torch
from prompt_graph.model import GAT, GCN, GCov, GIN, GraphSAGE, GraphTransformer


class BaseInfer:
    def __init__(self, pre_train_model_path=None, gnn_type='TransformerConv', hid_dim=128, out_dim=128, num_layer=2, num_heads=1, dataset_name='Cora', device : int=5):
        self.pre_train_model_path = pre_train_model_path
        self.device = torch.device('cuda:'+ str(device) if torch.cuda.is_available() else 'cpu')
        self.hid_dim = hid_dim
        self.num_layer = num_layer
        self.dataset_name = dataset_name
        self.gnn_type = gnn_type
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

        self.gnn.to(self.device)

        if self.pre_train_model_path != 'None':
            self.gnn.load_state_dict(torch.load(self.pre_train_model_path, map_location=self.device))
            print("Successfully loaded pre-trained weights!")

         
      
 
            
      
