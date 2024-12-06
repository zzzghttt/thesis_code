import os
import sys
import yaml
from prompt_graph.inference import GraphInference, NodeInference
from prompt_graph.utils import seed_everything


if __name__ == '__main__':
    DATE = sys.argv[1]
    CONFIG_FILE_PATH = sys.argv[2]
    TASK = sys.argv[3]

    with open(CONFIG_FILE_PATH, 'r') as file:
        ALL_CONFIG = yaml.safe_load(file)

    TRAIN_CONF = ALL_CONFIG['TRAIN_CONF']

    dataset_path = ALL_CONFIG['DATASET_PATH']['PYG_GRAPH_SAVE2FILE_PATH'].format(DATE=DATE)
    model_file_path = ALL_CONFIG['DATASET_PATH']['PRETRAIN_MODEL_SAVE2FILE_PATH']
    embedding_file_path = ALL_CONFIG['DATASET_PATH']['Embedding_SAVE2FILE_PATH']

    device = TRAIN_CONF['DEVICE']
    seed = TRAIN_CONF['SEED']
    batch_size = TRAIN_CONF['BATCH_SIZE']
    model_type = TRAIN_CONF['MODEL_TYPE']
    gnn_type = TRAIN_CONF['CONV_TYPE']
    num_layer = TRAIN_CONF['NUM_LAYER']
    num_heads = TRAIN_CONF['NUM_HEADS']
    hid_dim = TRAIN_CONF['HIDDEN_DIM']
    out_dim = TRAIN_CONF['OUT_DIM']

    seed_everything(seed)

    if TASK == 'GraphTask':
        task = GraphInference(pre_train_model_path=model_file_path, dataset_name=dataset_path, num_layer=num_layer, num_heads=num_heads, gnn_type=gnn_type, hid_dim=hid_dim, out_dim=out_dim, device=device)
        df = task.inference(batch_size=batch_size)
        df.to_csv(os.path.join(embedding_file_path, f"sub_graph_embedding_{DATE}.csv"))
    
    if TASK == 'NodeTask':
        task = NodeInference(pre_train_model_path=model_file_path, dataset_name=dataset_path, num_layer=num_layer, num_heads=num_heads, gnn_type=gnn_type, hid_dim=hid_dim, out_dim=out_dim, device=device)
        df = task.inference(batch_size=batch_size)
        df.to_csv(os.path.join(embedding_file_path, f"node_embedding_{DATE}.csv"))