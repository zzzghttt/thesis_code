# -*- coding: utf-8 -*-
import os 
import sys
import torch
import yaml
import random
import numpy as np
import logging
from modules.homo_models import Graph_Model
from torch_geometric.loader import NeighborLoader
from utils.metrics import calculate_auc, calculate_pr_auc, calculate_ks, compute_recall_at_accuracy, compute_accuracy_at_recall
from utils.utility import seed_everything


logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(filename)s: %(levelname)s: %(message)s'
)
    
def evaluate(val_dataloader, model, device):
    model.eval()
    probs, eval_labels = [], []

    with torch.no_grad():
        for sample_data in val_dataloader:
            sample_data.to(device)
            batch_node_num = sample_data.batch_size

            y_pre = model(x=sample_data.x, edge_index=sample_data.edge_index)[: batch_node_num]
            y_ture = sample_data.y[: batch_node_num]
            # y_id = sample_data['uid'].seller_id[: batch_node_num]

            # sellers += [i for i in y_id.cpu().numpy()]
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
    homo_graph_data = torch.load(f'{pyg_graph_load_path}/PYG_HETERO_GRAPH_FOR_TRAIN.pth').to_homogeneous()
    
    fan_outs = all_config['NETWORK_CONFIG']['FAN_OUTS']
    
    model = Graph_Model(
            input_channels=-1, out_channels=1, 
            hidden_channels=all_config['NETWORK_CONFIG']['HIDEN_SIZE'], num_heads=all_config['NETWORK_CONFIG']['N_HEADS'], 
            num_layers=all_config['NETWORK_CONFIG']['N_LAYERS'], model_type=all_config['TRAIN_CONF']['CONV_TYPE']
        ).to(device)

    stopper.load_checkpoint(model)
    
    logging.info('\n\nloading model success...')
        
    train_dataloader = NeighborLoader(
        homo_graph_data,
        num_neighbors=fan_outs,
        batch_size=all_config['TRAIN_CONF']['BATCH_SIZE'],
        input_nodes=homo_graph_data.train_mask,
        directed=False
    )
    
    auc, pr_auc, ks, rec_95pre, rec_95pre_thr, pre_95rec, pre_95rec_thr = evaluate(train_dataloader, model, device)
    print(f'auc:{auc}\npr_auc:{pr_auc}\nks:{ks}\nrec_95pre:{rec_95pre}\nrec_95pre_thr:{rec_95pre_thr}\npre_95rec:{pre_95rec}\npre_95rec_thr:{pre_95rec_thr}')

if __name__ == '__main__':
    config_file_path = sys.argv[1]
    
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
    GRAPH_TYPE = all_config['TRAIN_CONF']['GRAPH_TYPE']
    MODEL_TYPE = all_config['TRAIN_CONF']['CONV_TYPE']
    assert (GRAPH_TYPE == "Homo") and (MODEL_TYPE in ["GCN", "SAGE", "GAT"])
    
    # load pyg graph
    pyg_graph_load_path = all_config['DATASET_PATH']['PYG_GRAPH_SAVE2FILE_PATH'].format(DATA_TYPE='eval')

    logging.info('\n\nloading data...')
    homo_graph_data = torch.load(f'{pyg_graph_load_path}/PYG_HETERO_GRAPH_FOR_TRAIN.pth').to_homogeneous()
    
    train_mask = homo_graph_data.train_mask
    
    # graph_metadata = list(homo_graph_data.metadata())
    fan_outs = all_config['NETWORK_CONFIG']['FAN_OUTS']
    
    model_path = all_config['DATASET_PATH']['MODEL_SAVE_PATH']
    state_dict = torch.load(model_path)
    
    model = Graph_Model(
        input_channels=-1, hidden_channels=all_config['NETWORK_CONFIG']['HIDEN_SIZE'], 
        out_channels=1, num_layers=all_config['NETWORK_CONFIG']['N_LAYERS'], num_heads=all_config['NETWORK_CONFIG']['N_HEADS'], model_type=MODEL_TYPE
    ).to(device)
    logging.info('\n{}\n\nMODEL ARCHITECTURE:\n{}\n{}'.format('**' * 95, model,'**' * 95))
    
    model.load_state_dict(state_dict)
    logging.info('\n\nloading model success...')
        
    train_dataloader = NeighborLoader(
        homo_graph_data,
        num_neighbors=fan_outs,
        batch_size=all_config['TRAIN_CONF']['BATCH_SIZE'],
        input_nodes=homo_graph_data.train_mask,
    )
    
    auc, pr_auc, ks, rec_95pre, rec_95pre_thr, pre_95rec, pre_95rec_thr = evaluate(train_dataloader, model, device)
    print(f'auc:{auc}\npr_auc:{pr_auc}\nks:{ks}\nrec_95pre:{rec_95pre}\nrec_95pre_thr:{rec_95pre_thr}\npre_95rec:{pre_95rec}\npre_95rec_thr:{pre_95rec_thr}')