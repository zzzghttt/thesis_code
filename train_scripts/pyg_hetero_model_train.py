# -*- coding: utf-8 -*-
import sys
import torch
import time
import yaml
import wandb
import logging
from utils.metrics import calculate_auc, calculate_pr_auc, calculate_ks, compute_recall_at_accuracy, compute_accuracy_at_recall
from utils.utility import EarlyStopper
from modules.hetero_models import Graph_Model
from torch_geometric.loader import NeighborLoader


logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(filename)s: %(levelname)s: %(message)s'
)

def evaluate(val_dataloader, model, device):
    model.eval()

    eval_labels, probs = [], []
    with torch.no_grad():
        for sample_data in val_dataloader:
            sample_data.to(device)
            batch_node_num = sample_data['uid'].batch_size

            y_pre = model(x_dict=sample_data.x_dict, edge_index_dict=sample_data.edge_index_dict)[: batch_node_num]
            y_true = sample_data['uid'].y[: batch_node_num]

            eval_labels += [i for i in y_true.cpu().numpy()]
            probs += [i for i in y_pre.squeeze(dim=-1).cpu().detach().numpy()]
    
        # metrics evaluate 
        auc = calculate_auc(eval_labels, probs)
        pr_auc = calculate_pr_auc(eval_labels, probs)
        ks = calculate_ks(eval_labels, probs)
        rec_95pre, rec_95pre_thr = compute_recall_at_accuracy(eval_labels, probs, 0.95)
        pre_95rec, pre_95rec_thr = compute_accuracy_at_recall(eval_labels, probs, 0.95)
        
        return auc, pr_auc, ks, rec_95pre, rec_95pre_thr, pre_95rec, pre_95rec_thr

def train(config, device, train_dataloader, val_dataloader, model):
    # loss fuction and optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config['TRAIN_CONF']['LEARNING_RATE'], weight_decay=config['TRAIN_CONF']['WEIGHT_DECAY']
        )

    # setup earlystopper to save best validation model
    if config['TRAIN_CONF']['EARLY_STOP'] > 0:
        stopper = EarlyStopper(
            patience=config['TRAIN_CONF']['EARLY_STOP'], start_time=time.strftime('%Y%m%d%H%M', time.localtime()), 
            model_save_dir=config['DATASET_PATH']['MODEL_SAVE_PATH'].format(COUNTRY=country), model_save_name=MODEL_NAME
            )
    
    for epoch in range(config['TRAIN_CONF']['TRAIN_EPOCHS']):
        torch.cuda.empty_cache()
        logging.info(f'Train on epoch {epoch:>4d}:')
        model.train()
        
        i_idx = 0
        all_loss = 0.0
        for sample_data in train_dataloader:
            sample_data.to(device)
            batch_node_num = sample_data['uid'].batch_size
            
            y_pre = model(x_dict=sample_data.x_dict, edge_index_dict=sample_data.edge_index_dict)[: batch_node_num]
            y_true = sample_data['uid'].y[: batch_node_num]
            
            loss = torch.nn.BCELoss(reduction='mean')(y_pre.squeeze(dim=-1), y_true)
            all_loss += loss.item() * batch_node_num

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            i_idx += 1
            if i_idx % 30 == 0:
                logging.info('proc: {}/{}, batch_loss: {};'.format(i_idx * batch_node_num, len(train_id), loss.item()))

        all_loss_mean = all_loss / len(train_id)     
    
        wandb.log(
            {'train/loss/epoch': all_loss_mean}
            )
        
        logging.info(
            {'train/loss/epoch': all_loss_mean}
            )
            
        
        if epoch % config['TRAIN_CONF']['EVAL_INTERNAL'] == 0:  
            torch.cuda.empty_cache()      
            auc, pr_auc, ks, rec_95pre, rec_95pre_thr, pre_95rec, pre_95rec_thr = evaluate(val_dataloader, model, device)

            # log evaluation results
            wandb.log(
                {
                'eval/ROC_AUC': auc,
                'eval/PR_AUC': pr_auc,
                'eval/KS': ks,
                'eval/R@0.95P': rec_95pre,
                'eval/R@0.95P/threhold': rec_95pre_thr,
                'eval/P@0.95R': pre_95rec,
                'eval/P@0.95R/threhold': pre_95rec_thr
                }
            )
            
            logging.info(
                'epoch:{} - ROC_AUC: {}; PR_AUC: {}; KS: {}; R@0.95P: {}; threhold: {}; P@0.95R: {}; threhold: {};' \
                    .format(epoch, auc, pr_auc, ks, rec_95pre, rec_95pre_thr, pre_95rec, pre_95rec_thr)
                )

            if config['TRAIN_CONF']['EARLY_STOP'] > 0:
                if stopper.step(ks, epoch, model):
                    break
    
    logging.info(
        '\nBest Epoch {}, Val {:.4f}'.format(stopper.best_ep, stopper.best_score)
    )
    
    if config['TRAIN_CONF']['EARLY_STOP']:
        stopper.load_checkpoint(model)
        auc, pr_auc, ks, rec_95pre, rec_95pre_thr, pre_95rec, pre_95rec_thr = evaluate(val_dataloader, model, device)
        
        logging.info(
            '\nBest Epoch {} - ROC_AUC: {}; PR_AUC: {}; KS: {}; R@0.95P: {}; threhold: {}; P@0.95R: {}; threhold: {};' \
                .format(stopper.best_ep, auc, pr_auc, ks, rec_95pre, rec_95pre_thr, pre_95rec, pre_95rec_thr)
        )
        
        wandb.log(
            {
            'eval/ROC_AUC': auc,
            'eval/PR_AUC': pr_auc,
            'eval/KS': ks,
            'eval/R@0.95P': rec_95pre,
            'eval/R@0.95P/threhold': rec_95pre_thr,
            'eval/P@0.95R': pre_95rec,
            'eval/P@0.95R/threhold': pre_95rec_thr
            }
        )
        
    return
    
    
if __name__ == '__main__':
    train_date = sys.argv[1]
    val_date = sys.argv[2]
    country = sys.argv[3]
    config_file_path = sys.argv[4]
    
    logging.info(f'train_date is {train_date}, val_date is {val_date}, config file path is {config_file_path}')

    # logging configs
    with open(config_file_path, 'r') as file:
        all_config = yaml.safe_load(file)
    
    logging.info(
        '\n{}\n\nDATASET_CONFIG: \n{}\n{}'.format(
            '**' * 95, yaml.dump(all_config, sort_keys=False, indent=4), '**' * 95
            )
        )

    # model save name
    MODEL_NAME = f'''{country}_{all_config['TRAIN_CONF']['CONV_TYPE']}_{all_config['TRAIN_CONF']['BATCH_SIZE']}_{all_config['TRAIN_CONF']['LEARNING_RATE']}_{all_config['TRAIN_CONF']['WEIGHT_DECAY']}_{all_config['NETWORK_CONFIG']['N_HEADS']}_{all_config['NETWORK_CONFIG']['N_LAYERS']}_{all_config['NETWORK_CONFIG']['HIDEN_SIZE']}_{all_config['NETWORK_CONFIG']['FAN_OUTS']}'''
    
    # init wandb
    wandb_project_name = all_config['TRAIN_CONF']['WANDB_PROJECT_NAME']
    wandb_name = f'''{country}_Onboarding_Model_{all_config['TRAIN_CONF']['CONV_TYPE']}_{val_date}_{train_date}_{int(time.time())}'''

    wandb.init(project=wandb_project_name, name=wandb_name, id=wandb_name, config=all_config)
    
    # device env
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load pyg graph
    train_pyg_graph_load_path = all_config['DATASET_PATH']['PYG_GRAPH_SAVE2FILE_PATH'].format(DATE=train_date, COUNTRY=country)
    val_pyg_graph_load_path = all_config['DATASET_PATH']['PYG_GRAPH_SAVE2FILE_PATH'].format(DATE=val_date, COUNTRY=country)

    logging.info('\n\nloading data...')
    hetero_graph_train_data = torch.load(f'{train_pyg_graph_load_path}/PYG_HETERO_GRAPH_FOR_TRAIN.pth')
    hetero_graph_val_data = torch.load(f'{val_pyg_graph_load_path}/PYG_HETERO_GRAPH_FOR_TRAIN.pth')
    
    train_id = hetero_graph_train_data['uid'].train_idx
    val_id = hetero_graph_val_data['uid'].train_idx
    
    graph_metadata = list(hetero_graph_train_data.metadata())
    fan_outs = all_config['NETWORK_CONFIG']['FAN_OUTS']
    
    model = Graph_Model(
            input_channels=-1, out_channels=1, 
            hidden_channels=all_config['NETWORK_CONFIG']['HIDEN_SIZE'], num_heads=all_config['NETWORK_CONFIG']['N_HEADS'], 
            num_layers=all_config['NETWORK_CONFIG']['N_LAYERS'], meta_data=graph_metadata
        ).to(device)
    
    train_dataloader = NeighborLoader(
        hetero_graph_train_data,
        num_neighbors={key: fan_outs for key in hetero_graph_train_data.edge_types},
        batch_size=all_config['TRAIN_CONF']['BATCH_SIZE'],
        input_nodes=('uid', train_id),
        directed=False
    )

    val_dataloader = NeighborLoader(
        hetero_graph_val_data,
        num_neighbors={key: fan_outs for key in hetero_graph_val_data.edge_types},
        batch_size=all_config['TRAIN_CONF']['BATCH_SIZE'],
        input_nodes=('uid', val_id),
        directed=False
    )
    
    train(all_config, device, train_dataloader, val_dataloader, model)