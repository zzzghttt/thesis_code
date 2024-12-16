# -*- coding: utf-8 -*-
import os
import gc
import torch
import numpy as np
import sys
import logging
import yaml
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm
import pickle
from sklearn.preprocessing import LabelEncoder
from utils.utility import dataset_random_split
from torch_geometric.data import HeteroData


logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(filename)s: %(levelname)s: %(message)s'
)

CONFIG_FILE_PATH = sys.argv[1]
DATA_TYPE = sys.argv[2]

logging.info(f'config file path is {CONFIG_FILE_PATH}')

with open(CONFIG_FILE_PATH, 'r') as file:
    ALL_CONFIG = yaml.safe_load(file)

logging.info(
    '\n{}\n\nDATASET_CONFIG: \n{}\n{}'.format(
        '**' * 80, yaml.dump(ALL_CONFIG['TRAIN_CONF'], sort_keys=False, indent=4), '**' * 80
        )
    )

EDGE_INDEX_SAVE2FILE_PATH = ALL_CONFIG['DATASET_PATH']['EDGE_INDEX_SAVE2FILE_PATH'].format(DATA_TYPE=DATA_TYPE)
NODE_FEATURE_SAVE2FILE_PATH = ALL_CONFIG['DATASET_PATH']['NODE_FEATURE_SAVE2FILE_PATH'].format(DATA_TYPE=DATA_TYPE)
FEATURE_SAVE2FILE_PATH = ALL_CONFIG['DATASET_PATH']['FEATURE_SAVE2FILE_PATH']

TRAIN_CONF = ALL_CONFIG['TRAIN_CONF']

PYG_GRAPH_SAVE2FILE_PATH = ALL_CONFIG['DATASET_PATH']['PYG_GRAPH_SAVE2FILE_PATH'].format(DATA_TYPE=DATA_TYPE)   # PYG Graph 存储路径
os.makedirs(PYG_GRAPH_SAVE2FILE_PATH, exist_ok=True)

# 构建异构图
# load graph data
hetero_graph_data = HeteroData()

for edge_type in os.listdir(EDGE_INDEX_SAVE2FILE_PATH):
    tmp_files_path = os.path.join(EDGE_INDEX_SAVE2FILE_PATH, edge_type)
    if not os.path.exists(tmp_files_path):
        logging.info(f'{tmp_files_path} is empty...')
        df = pd.DataFrame(columns=['src_id', 'dst_id', 'weight'])
    else:
        logging.info(f'LOADING {edge_type} from {tmp_files_path}...')
        df = pq.read_table(tmp_files_path, columns=['src_id', 'dst_id', 'weight', 'group_id'], memory_map=True)
    
    src = df['src_id'].to_numpy()
    dst = df['dst_id'].to_numpy()
    edges = torch.tensor([np.concatenate((src, dst)), np.concatenate((dst, src))], dtype=torch.long)
    
    edge_name = edge_type
    hetero_graph_data[('node', edge_name, 'node')].edge_index = edges
    
    del edges, df; gc.collect()
    
# node feature process
logging.info('add node features...')
node_feature = pq.read_table(NODE_FEATURE_SAVE2FILE_PATH, columns=['id', 'label', 'feature_map_string', 'feature_map_double', 'group_id'], memory_map=True).to_pandas()
node_feature.sort_values(by='id', ascending=True, inplace=True) 
node_feature.reset_index(drop=True, inplace=True)
logging.info(
    'check node_feature id order minval:{}; maxval:{}; count distinct:{}'.format(
        node_feature.iloc[0, 0], node_feature.iloc[-1, 0], node_feature['id'].shape
        )
    )

group_id = node_feature['group_id']

label = node_feature['label']

# sparse feature process
# 转换成 DataFrame 处理
logging.info('sparse feature process...')
cat_arr = node_feature['feature_map_string'].values
df = pd.DataFrame([list(x) for x in cat_arr])

tmp_uid_cat_feats = np.zeros(df.shape, dtype='int32')
for i in tqdm(range(df.shape[1])):
    # 获取每个类别的频率
    vc = df.iloc[:, i].value_counts(normalize=True)
    
    with open(f'{FEATURE_SAVE2FILE_PATH}/{i}.pickle', 'rb') as f:
        le = pickle.load(f)
    
    map_label_set = set(le.classes_)
    df.iloc[:, i] = df.iloc[:, i].where(df.iloc[:, i].isin(map_label_set), 'other')
    
    # 使用 LabelEncoder 将类别转换为0-k的数字
    df.iloc[:, i] = le.transform(df.iloc[:, i])
    # 保存到新的 numpy array
    tmp_uid_cat_feats[:, i] = df.iloc[:, i].values

logging.info( f'cat feature process finished')

# dense feature process
## get dense feature mean value 
dense_node_feature = np.stack(node_feature['feature_map_double'].values)
del node_feature; gc.collect()

dense_node_feature_mean = np.nanmean(dense_node_feature, axis=0) 
logging.info( f'MEAN UID FEATURES: {dense_node_feature_mean}')

## nan value process
# fillna with mean feat
# nan_elements = np.isnan(dense_node_feature)
# for column_index, mean_value in enumerate(tqdm(dense_node_feature_mean)):
#     dense_node_feature[nan_elements[:, column_index], column_index] = mean_value
   
# fillna with 0
tmp_uid_dense_feats = np.nan_to_num(dense_node_feature, 0)

if TRAIN_CONF['DENSE_FEATURE_NORMALIZE']:
    uid_dense_feats_mean = np.load(f'{FEATURE_SAVE2FILE_PATH}/uid_dense_feats_mean.npy')
    uid_dense_feats_std = np.load(f'{FEATURE_SAVE2FILE_PATH}/uid_dense_feats_std.npy')
    tmp_uid_dense_feats = (tmp_uid_dense_feats - uid_dense_feats_mean) / uid_dense_feats_std
    tmp_uid_dense_feats = np.nan_to_num(tmp_uid_dense_feats, 0)

logging.info( f'dense feature process finished')

dense_feature = torch.tensor(tmp_uid_dense_feats, dtype=torch.float)
cat_feature = torch.tensor(tmp_uid_cat_feats, dtype=torch.int)
feature = torch.cat((dense_feature, cat_feature), dim=1)
label = torch.tensor(label, dtype=torch.float)
group_id = torch.tensor(group_id, dtype=torch.int)
train_mask, val_mask, train_idx, val_idx = dataset_random_split(label, train_size=TRAIN_CONF['TRAIN_SIZE'], val_size=TRAIN_CONF['VAL_SIZE'])

hetero_graph_data['node'].x = feature
hetero_graph_data['node'].y = label
hetero_graph_data['node'].group_id = group_id
hetero_graph_data['node'].train_idx = train_idx
hetero_graph_data['node'].val_idx = val_idx
hetero_graph_data['node'].train_mask = train_mask
hetero_graph_data['node'].val_mask = val_mask

logging.info('saving graph...')
torch.save(hetero_graph_data, f'{PYG_GRAPH_SAVE2FILE_PATH}/PYG_HETERO_GRAPH_FOR_TRAIN.pth')