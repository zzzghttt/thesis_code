from prompt_graph.data import SubGraphPretrainDataset
from torch_geometric.loader import DataLoader
from .base import BaseInfer
import pandas as pd 
import torch


class GraphInference(BaseInfer):
    def __init__(self, *args, **kwargs):    
        super().__init__(*args, **kwargs)
        self.task_type = 'GraphTask'
        self.load_graph_data()
        self.initialize_gnn(self.input_dim, self.hid_dim, self.out_dim) 

    def load_graph_data(self):
        subgraph = SubGraphPretrainDataset(datapath=self.dataset_name)
        self.graph_list, self.input_dim = subgraph.get_graph_list()

    def inference(self, batch_size):
        test_loader = DataLoader(self.graph_list, batch_size=batch_size, shuffle=False)
        self.gnn.eval() 

        group_ids = []
        embeddings = []
        with torch.no_grad():
            for batch_data in test_loader:  
                batch_data = batch_data.to(self.device)
                batch_size = batch_data.batch_size
                out = self.gnn(batch_data.x, batch_data.edge_index, batch_data.batch)

                embedding = out.cpu().detach().numpy()
                group_ids += [int(i) for i in batch_data.group_id]
                embeddings += [row for row in embedding]

        df = pd.DataFrame({'group_id': group_ids, 'embedding': embeddings})

        return df
        

        
