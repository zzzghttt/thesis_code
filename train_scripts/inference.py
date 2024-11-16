# -*- coding: utf-8 -*-
import os 
import sys
import torch
import yaml
import logging
import pandas as pd
from modules.hetero_models import Graph_Model
from torch_geometric.loader import NeighborLoader
from utils.utility import run_command, seed_everything


logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(filename)s: %(levelname)s: %(message)s'
)

def inference(val_dataloader, model, device):
    model.eval()
    sellers, probs = [], []

    with torch.no_grad():
        for sample_data in val_dataloader:
            sample_data.to(device)
            batch_node_num = sample_data['uid'].batch_size

            y_pre = model(x_dict=sample_data.x_dict, edge_index_dict=sample_data.edge_index_dict)[: batch_node_num]
            y_id = sample_data['uid'].seller_id[: batch_node_num]

            sellers += [i for i in y_id.cpu().numpy()]
            probs += [i for i in y_pre.squeeze(dim=-1).cpu().detach().numpy()]
        
        return sellers, probs

    
    
if __name__ == '__main__':
    inference_date = sys.argv[1]
    country = sys.argv[2]
    config_file_path = sys.argv[3]
    
    logging.info(f'inference_date is {inference_date}, config file path is {config_file_path}')

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
    pyg_graph_load_path = all_config['DATASET_PATH']['PYG_GRAPH_SAVE2FILE_PATH'].format(DATE=inference_date, COUNTRY=country)

    logging.info('\n\nloading data...')
    hetero_graph_data = torch.load(f'{pyg_graph_load_path}/PYG_HETERO_GRAPH_FOR_TRAIN.pth')
    
    train_id = hetero_graph_data['uid'].train_idx
    
    graph_metadata = list(hetero_graph_data.metadata())
    fan_outs = all_config['NETWORK_CONFIG']['FAN_OUTS']
    
    model_path = all_config['DATASET_PATH']['MODEL_SAVE2FILE_PATH']
    state_dict = torch.load(model_path)
    
    model = Graph_Model(
            input_channels=-1, out_channels=1, 
            hidden_channels=all_config['NETWORK_CONFIG']['HIDEN_SIZE'], num_heads=all_config['NETWORK_CONFIG']['N_HEADS'], 
            num_layers=all_config['NETWORK_CONFIG']['N_LAYERS'], meta_data=graph_metadata
        ).to(device)
    
    model.load_state_dict(state_dict)
    logging.info('\n\nloading model success...')
        
    train_dataloader = NeighborLoader(
        hetero_graph_data,
        num_neighbors={key: fan_outs for key in hetero_graph_data.edge_types},
        batch_size=all_config['TRAIN_CONF']['BATCH_SIZE'],
        input_nodes=('uid', train_id),
        directed=False
    )
    
    sellers, probs = inference(train_dataloader, model, device)
    df = pd.DataFrame({'seller_id': sellers, 'probs': probs})
    df.to_csv(f'model_result_{inference_date}.csv')
    
    run_command(f'''hdfs dfs -put 'model_result_{inference_date}.csv' {all_config['DATASET_PATH']['PREDICT_SAVE2FILE_PATH']}''')