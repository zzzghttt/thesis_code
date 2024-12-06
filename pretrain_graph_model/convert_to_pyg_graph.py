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
import pickle 
import multiprocessing
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected


logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(filename)s: %(levelname)s: %(message)s'
)

def get_subgraph_pyg_data(edge_idx, seller_group_id):
    group_id_list = edge_idx['group_id'].unique()

    for group_id in group_id_list:
        sub_graph_df = edge_idx[edge_idx.group_id == group_id]
        
        src_id = sub_graph_df['src_id'].to_numpy()
        dst_id = sub_graph_df['dst_id'].to_numpy()
        edge_index = torch.tensor([src_id, dst_id], dtype=torch.long)
        edge_index = to_undirected(edge_index)
        
        sub_graph_feature_df = seller_group_id[seller_group_id.group_id == group_id]
        dense_feature = torch.tensor(np.array(sub_graph_feature_df['dense_feature'].to_list()), dtype=torch.float)
        cat_feature = torch.tensor(np.array(sub_graph_feature_df['cat_feature'].to_list()), dtype=torch.int)
        feature = torch.cat((dense_feature, cat_feature), dim=1)
        
        sub_graph_data = Data(x=feature, edge_index=edge_index)
        sub_graph_data.seller_id = sub_graph_feature_df['seller_id'].values
        
        torch.save(sub_graph_data, f'{PYG_GRAPH_SAVE2FILE_PATH}/pyg_subgraph_{group_id}.pth')

def multiprocess_run(group_id_list, edge_idx, seller_group_id, num_pros=128):
    split_num = int(len(group_id_list) / num_pros) + 1

    ps = []
    for i in range(num_pros):
        batch_group_id = group_id_list[i * split_num: (i + 1) * split_num]
        if len(batch_group_id) > 0:
            batch_edge_idx = edge_idx[(edge_idx['group_id'] >= batch_group_id[0]) & (edge_idx['group_id'] <= batch_group_id[-1])]
            batch_seller_group_id = seller_group_id[(seller_group_id['group_id'] >= batch_group_id[0]) & (seller_group_id['group_id'] <= batch_group_id[-1])]
            p = multiprocessing.Process(target=get_subgraph_pyg_data, args=(batch_edge_idx, batch_seller_group_id))
            ps.append(p)

    logging.info(f'multi-process subgraph process started...')
    for p in ps:
        p.start()
    for p in ps:
        p.join()
    logging.info(f'multi-process subgraph process finished...')


if __name__ == '__main__':
    DATE = sys.argv[1] # format 'yyyyMMdd'
    CONFIG_FILE_PATH = sys.argv[2]
    logging.info(f'DATE is {DATE}, config file path is {CONFIG_FILE_PATH}')

    with open(CONFIG_FILE_PATH, 'r') as file:
        ALL_CONFIG = yaml.safe_load(file)

    logging.info(
        '\n{}\n\TRAIN_CONF: \n{}\n{}'.format(
            '**' * 80, yaml.dump(ALL_CONFIG['TRAIN_CONF'], sort_keys=False, indent=4), '**' * 80
            )
        )

    EDGE_INDEX_SAVE2FILE_PATH = ALL_CONFIG['DATASET_PATH']['EDGE_INDEX_SAVE2FILE_PATH'].format(DATE=DATE)
    NODE_FEATURE_SAVE2FILE_PATH = ALL_CONFIG['DATASET_PATH']['NODE_FEATURE_SAVE2FILE_PATH'].format(DATE=DATE)
    FEATURE_SAVE2FILE_PATH = ALL_CONFIG['DATASET_PATH']['FEATURE_SAVE2FILE_PATH']
    os.makedirs(FEATURE_SAVE2FILE_PATH, exist_ok=True)

    TRAIN_CONF = ALL_CONFIG['TRAIN_CONF']

    PYG_GRAPH_SAVE2FILE_PATH = ALL_CONFIG['DATASET_PATH']['PYG_GRAPH_SAVE2FILE_PATH'].format(DATE=DATE) # PYG Graph 存储路径
    os.makedirs(PYG_GRAPH_SAVE2FILE_PATH, exist_ok=True)

    # 节点特征处理
    logging.info('add node features...')
    node_feature = pq.read_table(
        NODE_FEATURE_SAVE2FILE_PATH, columns=['id', 'feature_map_string', 'feature_map_double', 'group_id', 'seller_id'], memory_map=True
        ).to_pandas()

    node_feature.sort_values(by=['group_id', 'id'], ascending=True, inplace=True) 
    seller_group_id = node_feature[['group_id', 'id', 'seller_id']]

    sparse_node_feature = pd.DataFrame([list(x) for x in node_feature['feature_map_string'].values])
    dense_node_feature = np.stack(node_feature['feature_map_double'].values)

    del node_feature; gc.collect()

    # sparse feature process
    logging.info('sparse feature process...')
    tmp_uid_cat_feats = np.zeros(sparse_node_feature.shape, dtype='int32')
    for i in tqdm(range(sparse_node_feature.shape[1])):
        le = LabelEncoder()

        # 获取每个类别的频率
        vc = sparse_node_feature.iloc[:, i].value_counts(normalize=True)
        if TRAIN_CONF['LABEL_TOPN'] > 0 and TRAIN_CONF['LABEL_TOPN'] < 1:
            _n = int(len(vc) * TRAIN_CONF['LABEL_TOPN'])
            if _n==len(vc):
                logging.warning(f'LABEL TOPN not effect at feature idx: {i}')
            top_labels = vc.index[:_n]
            sparse_node_feature.iloc[:, i] = sparse_node_feature.iloc[:, i].where(sparse_node_feature.iloc[:, i].isin(top_labels), 'other')
            
        # 使用 LabelEncoder 将类别转换为0-k的数字
        sparse_node_feature.iloc[:, i] = le.fit_transform(sparse_node_feature.iloc[:, i])
        with open(f'{FEATURE_SAVE2FILE_PATH}/{i}.pickle', 'wb') as f:
            pickle.dump(le, f)
            
        # 保存到新的 numpy array
        tmp_uid_cat_feats[:, i] = sparse_node_feature.iloc[:, i].values
        
    del sparse_node_feature; gc.collect()
    # np.save(f'{PYG_GRAPH_SAVE2FILE_PATH}/uid_sparse_feats.npy', tmp_uid_cat_feats)
    logging.info( f'cat feature process finished')

    tmp_uid_cat_feats_mean = tmp_uid_cat_feats.mean(axis=0)
    tmp_uid_cat_feats_std = tmp_uid_cat_feats.std(axis=0)

    np.save(f'{FEATURE_SAVE2FILE_PATH}/uid_cat_feats_mean.npy', tmp_uid_cat_feats_mean)
    np.save(f'{FEATURE_SAVE2FILE_PATH}/uid_cat_feats_std.npy', tmp_uid_cat_feats_std)

    if TRAIN_CONF['CAT_FEATURE_NORMALIZE']:
        tmp_uid_cat_feats = (tmp_uid_cat_feats - tmp_uid_cat_feats_mean) / tmp_uid_cat_feats_std 
        tmp_uid_cat_feats = np.nan_to_num(tmp_uid_cat_feats, 0)

    seller_group_id['cat_feature'] = pd.Series([row for row in tmp_uid_cat_feats])
    del tmp_uid_cat_feats; gc.collect()

    # dense feature process
    # fillna with 0
    tmp_uid_dense_feats = np.nan_to_num(dense_node_feature, 0)
    del dense_node_feature; gc.collect()

    tmp_uid_dense_feats_mean = tmp_uid_dense_feats.mean(axis=0)
    tmp_uid_dense_feats_std = tmp_uid_dense_feats.std(axis=0)
    np.save(f'{FEATURE_SAVE2FILE_PATH}/uid_dense_feats_mean.npy', tmp_uid_dense_feats_mean)
    np.save(f'{FEATURE_SAVE2FILE_PATH}/uid_dense_feats_std.npy', tmp_uid_dense_feats_std)

    if TRAIN_CONF['DENSE_FEATURE_NORMALIZE']:
        tmp_uid_dense_feats = (tmp_uid_dense_feats - tmp_uid_dense_feats_mean) / tmp_uid_dense_feats_std 
        tmp_uid_dense_feats = np.nan_to_num(tmp_uid_dense_feats, 0)

    # np.save(f'{PYG_GRAPH_SAVE2FILE_PATH}/uid_dense_feats.npy', tmp_uid_dense_feats)
    logging.info( f'dense feature process finished')
    seller_group_id['dense_feature'] = pd.Series([row for row in tmp_uid_dense_feats])
    del tmp_uid_dense_feats; gc.collect()

    # 获得边
    edge_idx = pq.read_table(
        EDGE_INDEX_SAVE2FILE_PATH, columns=['src', 'dst', 'src_id', 'dst_id', 'group_id'], memory_map=True
        ).to_pandas()
    edge_idx.sort_values(by=['group_id'], ascending=True, inplace=True)

    # 获取子图的最大节点的 group_id
    group_id_list = np.sort(edge_idx['group_id'].unique())

    multiprocess_run(group_id_list, edge_idx, seller_group_id)