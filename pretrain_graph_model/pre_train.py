import sys
import yaml
import time 
import wandb
from prompt_graph.pretrain import GraphCL, GraphMAE, GraphclClusterLoss
from prompt_graph.utils import seed_everything


if __name__ == '__main__':
    DATE = sys.argv[1]
    CONFIG_FILE_PATH = sys.argv[2]

    with open(CONFIG_FILE_PATH, 'r') as file:
        ALL_CONFIG = yaml.safe_load(file)

    TRAIN_CONF = ALL_CONFIG['TRAIN_CONF']

    dataset_path = ALL_CONFIG['DATASET_PATH']['PYG_GRAPH_SAVE2FILE_PATH'].format(DATE=DATE)
    model_save2file_path = ALL_CONFIG['DATASET_PATH']['MODEL_SAVE2FILE_PATH']

    device = TRAIN_CONF['DEVICE']
    seed = TRAIN_CONF['SEED']
    epochs = TRAIN_CONF['TRAIN_EPOCHS']
    batch_size = TRAIN_CONF['BATCH_SIZE']
    model_type = TRAIN_CONF['MODEL_TYPE']
    gnn_type = TRAIN_CONF['CONV_TYPE']
    num_layer = TRAIN_CONF['NUM_LAYER']
    num_heads = TRAIN_CONF['NUM_HEADS']
    mask_rate = TRAIN_CONF['MASK_RATE']
    drop_edge_rate = TRAIN_CONF['DROP_EDGE_RATE']
    replace_rate = TRAIN_CONF['REPLACE_RATE']
    hid_dim = TRAIN_CONF['HIDDEN_DIM']
    out_dim = TRAIN_CONF['OUT_DIM']
    lr = TRAIN_CONF['LEARNING_RATE']
    decay = TRAIN_CONF['WEIGHT_DECAY']

    seed_everything(seed)

    wandb_project_name = "Graph_Pretrain"
    wandb_name = "{}_{}_{}_{}_{}_{}_{}".format(wandb_project_name, model_type, gnn_type, batch_size, hid_dim, out_dim, int(time.time()))
    wandb.init(project=wandb_project_name, name=wandb_name, id=wandb_name, config=ALL_CONFIG)

    if model_type == 'GraphMAE':
        pt = GraphMAE(
            dataset_name=dataset_path, gnn_type=gnn_type, hid_dim=hid_dim, out_dim=out_dim, gln=num_layer, num_epoch=epochs, 
            num_heads=num_heads, device=device, mask_rate=mask_rate, drop_edge_rate=drop_edge_rate, replace_rate=replace_rate,
            )

        pt.pretrain(batch_size=batch_size, lr=lr, decay=decay, epochs=epochs, model_save_dir=model_save2file_path)

    if model_type == 'GraphCL':
        aug1= TRAIN_CONF['AUG1']
        aug2= TRAIN_CONF['AUG2']
        aug_ratio = TRAIN_CONF['AUG_RATIO']

        pt = GraphCL(
            dataset_name=dataset_path, gnn_type=gnn_type, hid_dim=hid_dim, out_dim=out_dim,
            gln=num_layer, num_epoch=epochs, num_heads=num_heads, device=device
            )
            
        pt.pretrain(
            batch_size=batch_size, aug1=aug1, aug2=aug2, aug_ratio=aug_ratio, 
            lr=lr, decay=decay, epochs=epochs, model_save_dir=model_save2file_path
            )

    if model_type == 'GraphclClusterLoss':
        aug1= TRAIN_CONF['AUG1']
        aug2= TRAIN_CONF['AUG2']
        aug_ratio = TRAIN_CONF['AUG_RATIO']

        pt = GraphclClusterLoss(
            dataset_name=dataset_path, gnn_type=gnn_type, hid_dim=hid_dim, out_dim=out_dim,
            gln=num_layer, num_epoch=epochs, num_heads=num_heads, device=device
            )
            
        pt.pretrain(
            batch_size=batch_size, aug1=aug1, aug2=aug2, aug_ratio=aug_ratio, 
            lr=lr, decay=decay, epochs=epochs, model_save_dir=model_save2file_path
            )