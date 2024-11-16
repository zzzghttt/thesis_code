# -*- coding: utf-8 -*-
import os 
import sys
import torch
import yaml
import random
import numpy as np
import logging
from modules.hetero_models import Graph_Model
from torch_geometric.loader import NeighborLoader
from utils.metrics import calculate_auc, calculate_pr_auc, calculate_ks, compute_recall_at_accuracy, compute_accuracy_at_recall
from utils.utility import seed_everything


logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(filename)s: %(levelname)s: %(message)s'
)
    
def evaluate(val_dataloader, model, device):
    model.eval()
    sellers, probs, eval_labels = [], [], []

    with torch.no_grad():
        for sample_data in val_dataloader:
            sample_data.to(device)
            batch_node_num = sample_data['node'].batch_size

            y_pre = model(x_dict=sample_data.x_dict, edge_index_dict=sample_data.edge_index_dict)[: batch_node_num]
            y_ture = sample_data['node'].y[: batch_node_num]
            y_id = sample_data['node'].seller_id[: batch_node_num]

            sellers += [i for i in y_id.cpu().numpy()]
            eval_labels += [i for i in y_ture.cpu().numpy()]
            probs += [i for i in y_pre.squeeze(dim=-1).cpu().detach().numpy()]

        # metrics evaluate 
        auc = calculate_auc(eval_labels, probs)
        pr_auc = calculate_pr_auc(eval_labels, probs)
        ks = calculate_ks(eval_labels, probs)
        rec_95pre, rec_95pre_thr = compute_recall_at_accuracy(eval_labels, probs, 0.95)
        pre_95rec, pre_95rec_thr = compute_accuracy_at_recall(eval_labels, probs, 0.95)

        return auc, pr_auc, ks, rec_95pre, rec_95pre_thr, pre_95rec, pre_95rec_thr

def inference(
    config_file_path,
    stopper,
    ):
    logging.info(f'config file path is {config_file_path}')

    # set seeds
    seed_everything(1)
    
    # logging configs
    with open(config_file_path, 'r') as file:
        all_config = yaml.safe_load(file)
    
    logging.info(
        '\n{}\n\nDATASET_CONFIG: \n{}\n{}'.format(
            '**' * 95, yaml.dump(all_config, sort_keys=False, indent=4), '**' * 95
            )
        )
    
    # device env
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load pyg graph
    pyg_graph_load_path = all_config['DATASET_PATH']['PYG_GRAPH_SAVE2FILE_PATH'].format(DATA_TYPE='eval')

    logging.info('\n\nloading data...')
    hetero_graph_data = torch.load(f'{pyg_graph_load_path}/PYG_HETERO_GRAPH_FOR_TRAIN.pth')
    
    train_id = hetero_graph_data['node'].train_idx
    
    graph_metadata = list(hetero_graph_data.metadata())
    fan_outs = all_config['NETWORK_CONFIG']['FAN_OUTS']
    
    model = Graph_Model(
            input_channels=-1, out_channels=1, 
            hidden_channels=all_config['NETWORK_CONFIG']['HIDEN_SIZE'], num_heads=all_config['NETWORK_CONFIG']['N_HEADS'], 
            num_layers=all_config['NETWORK_CONFIG']['N_LAYERS'], meta_data=graph_metadata
        ).to(device)

    stopper.load_checkpoint(model)
    
    logging.info('\n\nloading model success...')
        
    train_dataloader = NeighborLoader(
        hetero_graph_data,
        num_neighbors={key: fan_outs for key in hetero_graph_data.edge_types},
        batch_size=all_config['TRAIN_CONF']['BATCH_SIZE'],
        input_nodes=('node', train_id),
        directed=False
    )
    
    auc, pr_auc, ks, rec_95pre, rec_95pre_thr, pre_95rec, pre_95rec_thr = evaluate(train_dataloader, model, device)
    print(f'auc:{auc}\npr_auc:{pr_auc}\nks:{ks}\nrec_95pre:{rec_95pre}\nrec_95pre_thr:{rec_95pre_thr}\npre_95rec:{pre_95rec}\npre_95rec_thr:{pre_95rec_thr}')

if __name__ == '__main__':
    # inference_date = sys.argv[1]
    # country = sys.argv[2]
    # config_file_path = sys.argv[1]
    # inference(config_file_path)
    pass
    
