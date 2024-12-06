import torch
import time
import os
import torch.nn.functional as F
import numpy as np
import logging
import wandb
import numpy as np
from .base import PreTrain
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch import nn
from prompt_graph.data import SubGraphPretrainDataset
from itertools import chain
from functools import partial
from prompt_graph.model import GAT, GCN, GCov, GIN, GraphSAGE, GraphTransformer


def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
    loss = loss.mean()
    return loss

# loss function: sig
def sig_loss(x, y):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    loss = (x * y).sum(1)
    loss = torch.sigmoid(-loss)
    loss = loss.mean()
    return loss

def mask_edge(graph, mask_prob):
    E = graph.num_edges()
    mask_rates = torch.FloatTensor(np.ones(E) * mask_prob)
    masks = torch.bernoulli(1 - mask_rates)
    mask_idx = masks.nonzero().squeeze(1)
    return mask_idx

# graph transformation: drop edge
def drop_edge(graph, drop_rate, return_edges=False):
    if drop_rate <= 0:
        return graph
    edge_mask = mask_edge(graph, drop_rate)
    src = graph.edges()[0]
    dst = graph.edges()[1]

    nsrc = src[edge_mask]
    ndst = dst[edge_mask]

    ng = Data(edge_index=torch.concat((nsrc, ndst), 0))
    dsrc = src[~edge_mask]
    ddst = dst[~edge_mask]

    if return_edges:
        return ng, (dsrc, ddst)
    return ng

def initialize_gnn_decoder(gnn_type, input_dim, hid_dim, out_dim, num_layer, num_heads=4):
    if gnn_type == 'GAT':
        gnn = GAT(input_dim=input_dim, hid_dim=hid_dim, out_dim=out_dim, num_layer=num_layer, heads=num_heads)
    elif gnn_type == 'GCN':
        gnn = GCN(input_dim=input_dim, hid_dim=hid_dim, out_dim=out_dim, num_layer=num_layer)
    elif gnn_type == 'GraphSAGE':
        gnn = GraphSAGE(input_dim=input_dim, hid_dim=hid_dim, out_dim=out_dim, num_layer=num_layer)
    elif gnn_type == 'GIN':
        gnn = GIN(input_dim=input_dim, hid_dim=hid_dim, out_dim=out_dim, num_layer=num_layer)
    elif gnn_type == 'GCov':
        gnn = GCov(input_dim=input_dim, hid_dim=hid_dim, out_dim=out_dim, num_layer=num_layer)
    elif gnn_type == 'GraphTransformer':
        gnn = GraphTransformer(input_dim=input_dim, hid_dim=hid_dim, out_dim=out_dim, num_layer=num_layer)
    else:
        raise ValueError(f"Unsupported GNN type: {gnn_type}")
    return gnn
    

class PreModel(nn.Module):
    def __init__(self, encoder, decoder, hidden_dim, enc_in_dim, dec_in_dim, mask_rate=0.75, drop_edge_rate=0.0, replace_rate=0.1, loss_fn='sce', alpha_l=2):
        super(PreModel, self).__init__()
        self._mask_rate = mask_rate
        self._drop_edge_rate = drop_edge_rate
        self._replace_rate = replace_rate
        self._mask_token_rate = 1 - self._replace_rate
        self.hidden_dim = hidden_dim

        # build encoder
        self.encoder = encoder

        # build decoder
        self.decoder = decoder
        
        self.enc_mask_token = nn.Parameter(torch.zeros(1, enc_in_dim))
        self.encoder_to_decoder = nn.Linear(dec_in_dim, dec_in_dim, bias=False)
        # setup loss function
        self.criterion = self.setup_loss_fn(loss_fn, alpha_l)

    def forward(self, data):
        loss, x_hidden = self.mask_attr_prediction(data)
        loss_item = {"loss": loss.item()}

        return loss, loss_item, x_hidden

    def setup_loss_fn(self, loss_fn, alpha_l):
        if loss_fn == "mse":
            criterion = nn.MSELoss()
        elif loss_fn == "sce":
            criterion = partial(sce_loss, alpha=alpha_l)
        else:
            raise NotImplementedError
        return criterion

    def encoding_mask_noise(self, g, x, mask_rate=0.3):
        num_nodes = g.num_nodes
        perm = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = int(mask_rate * num_nodes)

        # random masking
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes: ]

        if self._replace_rate > 0:
            num_noise_nodes = int(self._replace_rate * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[perm_mask[: int(self._mask_token_rate * num_mask_nodes)]]
            noise_nodes = mask_nodes[perm_mask[-int(self._replace_rate * num_mask_nodes): ]]
            noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]
            out_x = x.clone()
            out_x[token_nodes] = 0.0
            out_x[noise_nodes] = x[noise_to_be_chosen]
        else:
            out_x = x.clone()
            token_nodes = mask_nodes
            out_x[mask_nodes] = 0.0

        out_x[token_nodes] += self.enc_mask_token
        use_g = g.clone()

        return use_g, out_x, (mask_nodes, keep_nodes)
    
    def mask_attr_prediction(self, data, pretrain_method='graphmae'):
        g = data
        x = data.x

        pre_use_g, use_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(g, x, self._mask_rate)
        
        if self._drop_edge_rate > 0:
            use_g, masked_edges = drop_edge(pre_use_g, self._drop_edge_rate, return_edges=True)
        else:
            use_g = pre_use_g
        
        # if there are noise nodes before reconstruction, then execture this line
        all_hidden = self.encoder(x=use_x, edge_index=use_g.edge_index)

        # if there are none noise nodes before reconstruction, please execture this line
        # all_hidden = self.encoder(data.x, data.edge_index)

        # ---- attribute reconstruction ----

        node_reps = self.encoder_to_decoder(all_hidden)
        node_reps[mask_nodes] = 0
        recon_node_reps = self.decoder(node_reps, use_g.edge_index)

        x_init = x[mask_nodes]
        x_rec = recon_node_reps[mask_nodes]
        loss = self.criterion(x_rec, x_init)

        return loss, all_hidden

    def embed(self, g, x):
        rep = self.encoder(x=x, edge_index=g.edge_index)
        return rep

    @property
    def enc_params(self):
        return self.encoder.parameters()

    @property
    def dec_params(self):
        return chain(*[self.encoder_to_decoder.parameters(), self.decoder.parameters()])


class GraphMAE(PreTrain):
    def __init__(self, *args, mask_rate=0.75, drop_edge_rate=0.0, replace_rate=0.1, loss_fn='sce', alpha_l=2, **kwargs):    # hid_dim=16
        super().__init__(*args, **kwargs)
        self.mask_rate = mask_rate       
        self.graph_list = self.load_graph_data()
        encoder = initialize_gnn_decoder(self.gnn_type, self.input_dim, self.hid_dim, self.out_dim, self.num_layer, num_heads=self.num_heads)
        decoder = initialize_gnn_decoder(self.gnn_type, self.out_dim, self.hid_dim, self.input_dim, self.num_layer, num_heads=self.num_heads)
        self.model = PreModel(encoder, decoder, self.hid_dim, self.input_dim, self.out_dim, mask_rate, drop_edge_rate, replace_rate, loss_fn, alpha_l).to(self.device)

    def load_graph_data(self):
        subgraph = SubGraphPretrainDataset(datapath=self.dataset_name)
        self.graph_list, self.input_dim = subgraph.get_graph_list()

        return self.graph_list
    
    def pretrain(self, lr=0.01, decay=0.0001, epochs=100, batch_size=10, model_save_dir='./Experiment/pre_trained_model'):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=decay)
        scheduler = lambda epoch :( 1 + np.cos((epoch) * np.pi / epochs) ) * 0.5
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)

        graph_dataloader = DataLoader(self.graph_list, batch_size=batch_size, shuffle=True)
        train_loss_min = np.inf

        for epoch in range(epochs):
            st_time = time.time()
            loss_list = []

            for step, batch in enumerate(graph_dataloader):
                optimizer.zero_grad()
                batch = batch.to(self.device)

                loss, loss_item, x_hidden = self.model(batch)              
                loss.backward()
                optimizer.step() 
                loss_list.append(loss.item())

                scheduler.step()

            logging.info(f"GraphMAE [Pretrain] Epoch {epoch}/{self.epochs} | Train Loss {np.mean(loss_list):.5f} | "
                  f"Cost Time {time.time() - st_time:.3}s")

            logging.info(f'''Current learning rate: {optimizer.state_dict()["param_groups"][0]["lr"]}''')

            wandb.log(
                {'train/loss/epoch': np.mean(loss_list)}
                )
            
            if train_loss_min > np.mean(loss_list):
                train_loss_min = np.mean(loss_list)

                folder_path = model_save_dir
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)

                torch.save(self.model.encoder.state_dict(),
                            "{}/{}.{}.{}.{}.{}.{}.{}.{}.pth".format(folder_path, 'GraphMAE', self.gnn_type, batch_size, self.hid_dim, self.out_dim, self.mask_rate, lr, decay))
                
                logging.info("+++model saved ! {}.{}.{}.{}.{}.{}.{}.{}.pth".format('GraphMAE', self.gnn_type, batch_size, self.hid_dim, self.out_dim, self.mask_rate, lr, decay))
            
        logging.info("minimum epoch loss: {}".format(train_loss_min))