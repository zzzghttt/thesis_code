# -*- coding: utf-8 -*-
import os
import gc
import dgl
import torch
import numpy as np
import sys
import logging
import yaml
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from utils.utility import dataset_random_split


logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(filename)s: %(levelname)s: %(message)s'
)

DATE = sys.argv[1] # format 'yyyyMMdd'
COUNTRY = sys.argv[2]
CONFIG_FILE_PATH = sys.argv[3]
logging.info(f'DATE is {DATE}, config file path is {CONFIG_FILE_PATH}')


with open(CONFIG_FILE_PATH, 'r') as file:
    ALL_CONFIG = yaml.safe_load(file)

logging.info(
    '\n{}\n\nDATASET_CONFIG: \n{}\n{}'.format(
        '**' * 80, yaml.dump(ALL_CONFIG['TRAIN_CONF'], sort_keys=False, indent=4), '**' * 80
        )
    )

NODE_INDEX_MAP_SAVE2FILE_PATH = ALL_CONFIG['DATASET_PATH']['NODE_INDEX_MAP_SAVE2FILE_PATH'].format(DATE=DATE, COUNTRY=COUNTRY)
EDGE_INDEX_SAVE2FILE_PATH = ALL_CONFIG['DATASET_PATH']['EDGE_INDEX_SAVE2FILE_PATH'].format(DATE=DATE)
NODE_FEATURE_SAVE2FILE_PATH = ALL_CONFIG['DATASET_PATH']['NODE_FEATURE_SAVE2FILE_PATH'].format(DATE=DATE, COUNTRY=COUNTRY)

TRAIN_CONF = ALL_CONFIG['TRAIN_CONF']

DGL_GRAPH_SAVE2FILE_PATH = ALL_CONFIG['DATASET_PATH']['DGL_GRAPH_SAVE2FILE_PATH'].format(DATE=DATE, COUNTRY=COUNTRY) # DGL Graph 存储路径
os.makedirs(DGL_GRAPH_SAVE2FILE_PATH, exist_ok=True)

# 获得节点数量
uid2idx = pq.read_table(NODE_INDEX_MAP_SAVE2FILE_PATH, columns=['idx', 'seller_id', 'label'], memory_map=True).to_pandas()
uid2idx.sort_values(by=['idx'], ascending=True, inplace=True)
MAX_IDX = uid2idx['idx'].max() + 1
logging.info(f'MAX_IDX {MAX_IDX}')

# 获取节点 label
label = uid2idx['label']

# 构建异构图
# load graph data
DATA_DICT_ALL = {}
TMP_EDGE_WEIGHTS = {}

for edge_type in os.listdir(EDGE_INDEX_SAVE2FILE_PATH):
    tmp_files_path = os.path.join(EDGE_INDEX_SAVE2FILE_PATH, edge_type, f'operation_country={COUNTRY}')
    logging.info(f'LOADING {edge_type} from {tmp_files_path}...')
    df = pq.read_table(tmp_files_path, columns=['src_id', 'dst_id', 'weight'], memory_map=True)

    edges = torch.tensor(np.concatenate((df['src_id'].to_numpy(), df['dst_id'].to_numpy())), dtype=torch.long), torch.tensor(np.concatenate((df['dst_id'].to_numpy(), df['src_id'].to_numpy())), dtype=torch.long)
    weights = torch.tensor(np.concatenate((df['weight'].to_numpy(), df['weight'].to_numpy())), dtype=torch.float)
    
    edge_name = edge_type
    DATA_DICT_ALL[('uid', edge_name, 'uid')] = edges
    TMP_EDGE_WEIGHTS[edge_name] = weights
    
    del edges, weights, df; gc.collect()
    
# build graph
g = dgl.heterograph(DATA_DICT_ALL, num_nodes_dict={'uid': MAX_IDX})

# dgl 特性，存成csr格式更加节约内存，并且不会影响后续使用
# g = g.formats('csr')
logging.info(f'check graph: {g}')

# add edge weight
for kk, vv in TMP_EDGE_WEIGHTS.items():
    g.edges[kk].data['weight'] = vv

del DATA_DICT_ALL, TMP_EDGE_WEIGHTS; gc.collect()
logging.info(f'GRAPH: {g}')

if TRAIN_CONF['FEATURE_WITH_MMAP']:
    logging.info('saving graph...')
    dgl.save_graphs(f'{DGL_GRAPH_SAVE2FILE_PATH}/GRAPH_FOR_TRAIN.bin', [g])
    del g; gc.collect()
    
# node feature process
logging.info('add node features...')
node_feature = pq.read_table(NODE_FEATURE_SAVE2FILE_PATH, columns=['id', 'feature_map_string', 'feature_map_double'], memory_map=True).to_pandas()
node_feature.sort_values(by='id', ascending=True, inplace=True) 
node_feature.reset_index(drop=True, inplace=True)
logging.info(
    'check node_feature id order minval:{}; maxval:{}; count distinct:{}'.format(
        node_feature.iloc[0, 0], node_feature.iloc[-1, 0], node_feature['id'].shape
        )
    )

# sparse feature process
# 转换成 DataFrame 处理
logging.info('sparse feature process...')
cat_arr = node_feature['feature_map_string'].values
df = pd.DataFrame([list(x) for x in cat_arr])

tmp_uid_cat_feats = np.zeros(df.shape, dtype='int32')
uid_sparse_feats_num = []
for i in tqdm(range(df.shape[1])):
    le = LabelEncoder()

    # 获取每个类别的频率
    vc = df.iloc[:, i].value_counts(normalize=True)
    if TRAIN_CONF['LABEL_TOPN'] > 0 and TRAIN_CONF['LABEL_TOPN'] < 1:
        _n = int(len(vc) * TRAIN_CONF['LABEL_TOPN'])
        if _n==len(vc):
            logging.warning(f'LABEL TOPN not effect at feature idx: {i}')
        top_labels = vc.index[:_n]
        df.iloc[:, i] = df.iloc[:, i].where(df.iloc[:, i].isin(top_labels), 'other')
        
    # 使用 LabelEncoder 将类别转换为0-k的数字
    df.iloc[:, i] = le.fit_transform(df.iloc[:, i])
    # 保存到新的 numpy array
    tmp_uid_cat_feats[:, i] = df.iloc[:, i].values
    uid_sparse_feats_num.append(len(le.classes_))

np.save(f'{DGL_GRAPH_SAVE2FILE_PATH}/uid_sparse_feats.npy', tmp_uid_cat_feats)
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
    tmp_uid_dense_feats = (tmp_uid_dense_feats - tmp_uid_dense_feats.mean(axis=0)) / tmp_uid_dense_feats.std(axis=0)  
    tmp_uid_dense_feats = np.nan_to_num(tmp_uid_dense_feats, 0)

np.save(f'{DGL_GRAPH_SAVE2FILE_PATH}/uid_dense_feats.npy', tmp_uid_dense_feats)
logging.info( f'dense feature process finished')

if TRAIN_CONF['FEATURE_WITH_MMAP']:
    fp = np.memmap(f'{DGL_GRAPH_SAVE2FILE_PATH}/uid_dense_feats.dat', dtype=np.float32, mode='w+', shape=tmp_uid_dense_feats.shape)
    fp[:] = tmp_uid_dense_feats[:]
    del fp
    fp = np.memmap(f'{DGL_GRAPH_SAVE2FILE_PATH}/uid_sparse_feats.dat', dtype=np.float32, mode='w+', shape=tmp_uid_cat_feats.shape)
    fp[:] = tmp_uid_cat_feats[:]
    del fp
    logging.info('successfully save as mmap array')
else:
    if TRAIN_CONF['GRAPH_TYPE'] == 'Homo':
        g_homo = dgl.to_homogeneous(g)
        dgl.save_graphs(f'{DGL_GRAPH_SAVE2FILE_PATH}/HOMO_GRAPH_FOR_TRAIN.bin', g_homo)

    dense_feature = torch.tensor(tmp_uid_dense_feats, dtype=torch.float)
    cat_feature = torch.tensor(tmp_uid_cat_feats, dtype=torch.int)
    feature = torch.cat((dense_feature, cat_feature), dim=1)
    label = torch.tensor(label, dtype=torch.long)
    train_mask, val_mask = dataset_random_split(label, train_size=TRAIN_CONF['TRAIN_SIZE'], val_size=TRAIN_CONF['VAL_SIZE'])
    
    logging.info('saving graph...')
    dgl.save_graphs(f'{DGL_GRAPH_SAVE2FILE_PATH}/HETERO_GRAPH_FOR_TRAIN.bin', g)
    torch.save({'train_mask': train_mask, 'val_mask': val_mask}, f'{DGL_GRAPH_SAVE2FILE_PATH}/mask_tensor.pth')
    torch.save(feature, f'{DGL_GRAPH_SAVE2FILE_PATH}/feature.pth')
    torch.save(label, f'{DGL_GRAPH_SAVE2FILE_PATH}/label.pth')
       
logging.info('finished all')