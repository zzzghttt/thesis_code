# -*- coding: utf-8 -*-
import os 
import sys
import torch
import yaml
import logging
import time
import pandas as pd
from modules.homo_models import Graph_Model
from torch_geometric.loader import NeighborLoader
from utils.utility import run_command, seed_everything


logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(filename)s: %(levelname)s: %(message)s'
)

def inference(val_dataloader, model, device):
    model.eval()
    group_id, nodes, probs = [], [], []

    with torch.no_grad():
        for sample_data in val_dataloader:
            sample_data.to(device)
            batch_node_num = sample_data.batch_size

            y_pre = model(x=sample_data.x, edge_index=sample_data.edge_index)[: batch_node_num]
            y_id = sample_data.n_id[: batch_node_num]
            y_gid = sample_data.group_id[: batch_node_num]

            group_id += [i for i in y_gid.cpu().numpy()]
            nodes += [i for i in y_id.cpu().numpy()]
            probs += [i for i in y_pre.squeeze(dim=-1).cpu().detach().numpy()]
        
        return group_id, nodes, probs
    
def filter(model_predict, predict_threshold):
    return model_predict.loc[model_predict['probs'] > predict_threshold]
    
if __name__ == '__main__':
    config_file_path = sys.argv[1]

    date_time = time.strftime('%Y%m%d%H%M', time.localtime())
    
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
    pyg_graph_load_path = all_config['DATASET_PATH']['PYG_GRAPH_SAVE2FILE_PATH'].format(DATA_TYPE='inference')

    logging.info('\n\nloading data...')
    homo_graph_data = torch.load(f'{pyg_graph_load_path}/PYG_HETERO_GRAPH_FOR_TRAIN.pth').to_homogeneous()
    
    train_mask = homo_graph_data.train_mask
    
    fan_outs = all_config['NETWORK_CONFIG']['FAN_OUTS']
    
    model_path = all_config['DATASET_PATH']['MODEL_SAVE2FILE_PATH']
    model_threshold = all_config['DATASET_PATH']['MODEL_THRESHOLD']
    state_dict = torch.load(model_path)
    
    model = Graph_Model(
        input_channels=-1, hidden_channels=all_config['NETWORK_CONFIG']['HIDEN_SIZE'], 
        out_channels=1, num_layers=all_config['NETWORK_CONFIG']['N_LAYERS'], model_type=MODEL_TYPE
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
    
    group_id, nodes, probs = inference(train_dataloader, model, device)
    df = pd.DataFrame({'group_id': group_id, 'node_id': nodes, 'probs': probs})

    origin_size = df.shape[0]
    df = filter(df, model_threshold)
    filtered_size = df.shape[0]

    df.to_csv(f'model_result_{date_time}.csv')
    print(f'Inference Finished! \norigin size: {origin_size}, filtered size: {filtered_size}\nSave to model_result_{date_time}.csv')

    # run_command(f'''hdfs dfs -put 'model_result_{inference_date}.csv' {all_config['DATASET_PATH']['PREDICT_SAVE2FILE_PATH']}''')