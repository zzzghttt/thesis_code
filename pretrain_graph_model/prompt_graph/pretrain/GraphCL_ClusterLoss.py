import os
import random
import torch
import wandb
import logging
import numpy as np
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from random import shuffle
from prompt_graph.utils import graph_views
from prompt_graph.data import SubGraphPretrainDataset
from torch.optim import Adam
from .base import PreTrain


logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(filename)s: %(levelname)s: %(message)s'
)


class GraphEmbedding(torch.nn.Module):
    def __init__(self, graph_num=10, graph_emb_dim=64, graph_hidden_dim=32, graph_output_dim=64):
        super(GraphEmbedding, self).__init__()
        self.graph_num = graph_num
        self.graph_emb_dim = graph_emb_dim
        self.graph_nn_emb = torch.nn.Embedding(graph_num, graph_emb_dim)
        torch.nn.init.xavier_uniform_(self.graph_nn_emb.weight)
        # self.graph_linear = torch.nn.Sequential(
        #         torch.nn.Linear(graph_emb_dim, graph_hidden_dim),
        #         torch.nn.ReLU(),
        #         torch.nn.Linear(graph_hidden_dim, graph_output_dim)
        #     )
        self.graph_linear = torch.nn.Linear(graph_emb_dim, graph_output_dim, bias=False)

    def forward(self, graph_id_list):
        graph_emb = self.graph_nn_emb(graph_id_list)
        graph_emb = self.graph_linear(graph_emb)
        graph_emb = F.normalize(graph_emb, p=2, dim=-1)
        return graph_emb


class ClusterContrastiveLossV2(torch.nn.Module):
    def __init__(self, device='cuda'):
        super(ClusterContrastiveLossV2, self).__init__()
        self.device = device

    def forward(self, z, index_list, graph_emb):
        # z = z.to(device)
        # start_time = time.time()
        z_shape = z.shape[0]
        cluster_shape = len(index_list)
        cluster_z = F.normalize(graph_emb, p=2, dim=1)
        dis = torch.matmul(z, cluster_z.t())
        mask = torch.zeros([z_shape, cluster_shape], dtype=torch.bool)
        for index, se in enumerate(index_list):
            mask[se[0]:se[1], index] = True
        pos_dis = F.relu(0.99 - dis[mask]).mean()
        pos_dis = torch.tensor(0.0) if torch.isnan(pos_dis).any() else pos_dis
        neg_dis = F.relu(dis[~mask] - 0.2).mean()
        neg_dis = torch.tensor(0.0) if torch.isnan(neg_dis).any() else neg_dis
        # logging.info(f'Rank: {self.device}, ClusterContrastiveLossV2 pos_dis: {pos_dis:.5f}, neg_dis: {neg_dis:.5f}, dis: {dis[0][0:5].cpu().detach().numpy()}, cost: {(time.time()-start_time):.5f}')
        return pos_dis + neg_dis


class GraphclClusterLoss(PreTrain):
    def __init__(self, *args, **kwargs):    # hid_dim=16
        super().__init__(*args, **kwargs)
        self.load_graph_data()
        self.initialize_gnn(self.input_dim, self.hid_dim, self.out_dim) 
        self.group_id_list = [subgraph.group_id for subgraph in self.graph_list] 
        self.group_id_map = {subgraph_id: id for id, subgraph_id in enumerate(self.group_id_list)}
        self.group_size_map = {subgraph.group_id: subgraph.num_nodes for subgraph in self.graph_list}
        self.graph_num = len(self.group_id_list)

        self.nn_graph_emb = GraphEmbedding(graph_num=self.graph_num)
        self.cluster_loss = ClusterContrastiveLossV2(self.device)
        self.projection_head = torch.nn.Sequential(
            torch.nn.Linear(self.out_dim, self.out_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(self.out_dim, self.out_dim)
        ).to(self.device)

    def load_graph_data(self):
        subgraph = SubGraphPretrainDataset(datapath=self.dataset_name)
        self.graph_list, self.input_dim = subgraph.get_graph_list()
    
    def get_loader(self, graph_list, batch_size,aug1=None, aug2=None, aug_ratio=None):

        if len(graph_list) % batch_size == 1:
            raise KeyError(
                "batch_size {} makes the last batch only contain 1 graph, \n which will trigger a zero bug in GraphCL!")
        
        shuffle(graph_list)
        if aug1 is None:
            aug1 = random.sample(['dropN', 'permE', 'maskN'], k=1)
        if aug2 is None:
            aug2 = random.sample(['dropN', 'permE', 'maskN'], k=1)
        if aug_ratio is None:
            aug_ratio = random.randint(1, 3) * 1.0 / 10  # 0.1,0.2,0.3

        logging.info("===graph views: {} and {} with aug_ratio: {}".format(aug1, aug2, aug_ratio))

        view_list_1 = []
        view_list_2 = []
        for g in graph_list:
            view_g = graph_views(data=g.clone(), aug=aug1, aug_ratio=aug_ratio)
            view_g = Data(x=view_g.x, edge_index=view_g.edge_index)
            view_list_1.append(view_g)
            view_g = graph_views(data=g.clone(), aug=aug2, aug_ratio=aug_ratio)
            view_g = Data(x=view_g.x, edge_index=view_g.edge_index)
            view_list_2.append(view_g)
        loader1 = DataLoader(view_list_1, batch_size=batch_size, shuffle=False,
                                num_workers=1)  # you must set shuffle=False !
        loader2 = DataLoader(view_list_2, batch_size=batch_size, shuffle=False,
                                num_workers=1)  # you must set shuffle=False !
        loader3 = DataLoader(graph_list, batch_size=batch_size, shuffle=False,
                                num_workers=1) 
        return loader1, loader2, loader3
    
    def forward_cl(self, x, edge_index, batch):
        x = self.gnn(x, edge_index, batch)
        x = self.projection_head(x)
        return x

    def loss_cl(self, x1, x2):
        T = 0.1
        batch_size, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)
        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = - torch.log(pos_sim / (sim_matrix.sum(dim=1) + 1e-4)).mean()
        # loss = pos_sim / ((sim_matrix.sum(dim=1) - pos_sim) + 1e-4)
        # loss = - torch.log(loss).mean() 
        return loss

    def train_graphcl(self, loader1, loader2, loader3, optimizer, scheduler):
        self.train()
        train_loss_accum = 0
        total_step = 0
        for step, batch in enumerate(zip(loader1, loader2, loader3)):
            batch1, batch2, batch3 = batch
            optimizer.zero_grad()
            x1 = self.forward_cl(batch1.x.to(self.device), batch1.edge_index.to(self.device), batch1.batch.to(self.device))
            x2 = self.forward_cl(batch2.x.to(self.device), batch2.edge_index.to(self.device), batch2.batch.to(self.device))
            x3 = self.gnn(batch3.x.to(self.device), batch3.edge_index.to(self.device))

            contrast_loss = self.loss_cl(x1, x2)

            graph_id_list = torch.tensor([self.group_id_map[group_id] for group_id in batch3.group_id]).to(self.device)
            graph_emb = self.nn_graph_emb(graph_id_list)

            index_list = []
            start_index = 0
            for group_id in batch3.group_id:
                graph_node_num = self.group_size_map[group_id]
                end_index = start_index + graph_node_num
                index_list.append((start_index, end_index)) 
                start_index += graph_node_num

            cluster_loss = self.cluster_loss(x3, index_list, graph_emb)
            loss = cluster_loss + 0.6 * contrast_loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            train_loss_accum += float(loss.detach().cpu().item())
            total_step = total_step + 1
            logging.info('proc: {}/{}, contrast_loss: {}. cluster_loss: {}, in_batch_loss: {};'.format(step+1, len(loader1), contrast_loss.item(), cluster_loss.item(), loss.item()))

        return train_loss_accum / total_step

    def pretrain(self, batch_size=10, aug1='dropN', aug2="permE", aug_ratio=None, lr=0.01, decay=0.0001, epochs=100, model_save_dir='./Experiment/pre_trained_model'):
        self.to(self.device)
        loader1, loader2, loader3 = self.get_loader(self.graph_list, batch_size, aug1=aug1, aug2=aug2, aug_ratio=aug_ratio)
        logging.info('start training {} | {} | {}...'.format(self.dataset_name, 'GraphclClusterLoss', self.gnn_type))
        optimizer = Adam(self.parameters(), lr=lr, weight_decay=decay)
        scheduler = lambda epoch :( 1 + np.cos((epoch) * np.pi / epochs) ) * 0.5
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)

        train_loss_min = 1000000
        for epoch in range(1, epochs + 1):  # 1..100
            train_loss = self.train_graphcl(loader1, loader2, loader3, optimizer, scheduler)

            logging.info("***epoch: {}/{} | train_loss: {:.8}".format(epoch, epochs, train_loss))
            wandb.log(
                {'train/loss/epoch': train_loss}
                )
    
            if train_loss_min > train_loss:
                train_loss_min = train_loss
                folder_path = model_save_dir
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)

                torch.save(self.gnn.state_dict(),
                           "{}/{}.{}.{}.{}.{}.{}.{}.pth".format(folder_path, 'GraphclClusterLoss', self.gnn_type, batch_size, self.hid_dim, self.out_dim, lr, decay))
                logging.info("+++model saved ! {}.{}.{}.{}.{}.{}.{}.pth".format('GraphclClusterLoss', self.gnn_type, batch_size, self.hid_dim, self.out_dim, lr, decay))