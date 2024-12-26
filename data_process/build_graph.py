import torch
import sys
import os
import numpy as np
import pandas as pd
from util import *
from table_config import *
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# Graph Data (Use Class Node Only)

def build_graph_data(database = 'train', api_key_path=None):
    # use class to class edge
    class_processed_data = pull_data(
        '''
        select * from class_processed_data
        ''',
        database
    )

    c2c_sp_df = pull_data(
        '''
        select * from edge_class_to_class
        ''',
        database
    )
    # print(class_processed_data.toPandas())
    raw_edge_index = torch.from_numpy(c2c_sp_df.select('src_id', 'dst_id').toPandas().values).T

    if database == 'inference':
        target_class_node = class_processed_data.where('isTest == 0').distinct()
        target_class_node_list = target_class_node.toPandas()['index'].tolist()
        class_edge_index, class_node_map, group_ids, labels = remap_node_edge(raw_edge_index, target_class_node_list)
    else:
        # Test Node
        target_class_node = class_processed_data.where('isTest == 1').distinct()
        target_class_node_list = target_class_node.toPandas()['index'].tolist()

        # Source Class Node
        test_list, class_list = find_class_by_test(class_processed_data.toPandas(), target_class_node_list)

        # remap node edge by both test_lsit and class list
        class_edge_index, class_node_map, group_ids, labels = remap_node_edge(raw_edge_index, class_list, test_list)
    
    node_feature_idx = list(class_node_map.values())

    node_feature_string = class_processed_data.select(
        'moduleName',
    ).toPandas().iloc[node_feature_idx]

    # Use embeddings for identifiers and source code
    if api_key_path:
        all_node_features = []
        embedding_map = {}
        
        # 获取 embedding
        for idx in tqdm(node_feature_idx):
            package_name = class_processed_data.select('packageName').toPandas().iloc[idx].values[0]
            class_name = class_processed_data.select('className').toPandas().iloc[idx].values[0]
            class_declaration_code = class_processed_data.select('classDeclarationCode').toPandas().iloc[idx].values[0]

            
            package_name_embedding = get_embedding(package_name, api_key_path) if package_name not in embedding_map else embedding_map[package_name]
            class_name_embedding = get_embedding(class_name, api_key_path) if class_name not in embedding_map else embedding_map[class_name]
            class_declaration_code_embedding = get_embedding(class_declaration_code, api_key_path) if class_declaration_code not in embedding_map else embedding_map[class_declaration_code]

            # 如果嵌入成功，将其与数值特征合并
            if package_name_embedding and class_name_embedding and class_declaration_code_embedding:
                embedding_map[package_name] = package_name_embedding
                embedding_map[class_name] = class_name_embedding
                embedding_map[class_declaration_code] = class_declaration_code_embedding
                
                # 获取该节点的数值特征
                numerical_features = class_processed_data.select(
                    'line_count', 
                    'import_count', 
                    'super_class_count',
                    'implement_count',
                    'method_count',
                    'constructor_count',
                    'constructor_dep_count',
                    'sub_class_count'
                ).toPandas().iloc[idx].values
                
                # 将 embeddings 合并到数值特征中（假设每个嵌入是 1536 维）
                embeddings = np.concatenate([
                    package_name_embedding,
                    class_name_embedding,
                    class_declaration_code_embedding
                ], axis=0)
                
                # 将合并后的特征添加到列表中
                combined_features = np.concatenate([numerical_features, embeddings])
                all_node_features.append(combined_features)
            else:
                print(f"Failed to get embeddings for node at index {idx}")

        node_feature_double = pd.DataFrame(all_node_features)
        print(f"Combined feature vector size: {node_feature_double.shape}")
    else:
        node_feature_double = class_processed_data.select(
            'line_count', 
            'import_count', 
            'super_class_count',
            'implement_count',
            'method_count',
            'constructor_count',
            'constructor_dep_count',
            'sub_class_count'
        ).toPandas().iloc[node_feature_idx]

        print(f"Node feature double vector size: {node_feature_double.shape}")

    # node feature
    columns_1 = ['id', 'label', 'feature_map_string', 'feature_map_double', 'group_id']
    node_feature = pd.DataFrame(columns=columns_1)
    node_feature['id'] = list(class_node_map.keys())
    node_feature['label'] = labels
    node_feature['feature_map_string'] = node_feature_string.values.tolist()
    node_feature['feature_map_double'] = node_feature_double.values.tolist()
    node_feature['group_id'] = group_ids
    node_feature = node_feature.astype({'id': int, 'label': int, 'feature_map_string': 'string', 'feature_map_double': 'string', 'group_id': int})

    # edge_index
    columns_2 = ['src_id', 'dst_id', 'weight', 'type', 'group_id']
    edge_index = pd.DataFrame(columns=columns_2)
    edge_index['src_id'] = class_edge_index[0]
    edge_index['dst_id'] = class_edge_index[1]
    edge_index['weight'] = 1
    edge_index['weight'] = edge_index['weight'].astype(float)
    edge_index['type'] = 'class_to_class'
    edge_index['group_id'] = class_edge_index[2]
    edge_index = edge_index.astype({'src_id': int, 'dst_id': int, 'weight': float, 'type': 'string', 'group_id': int})

    return node_feature, edge_index, class_node_map


def save_data(node_feature, edge_index, class_node_map, save_path):
    # Save Data
    node_feature['feature_map_string'] = node_feature['feature_map_string'].apply(eval)
    node_feature['feature_map_double'] = node_feature['feature_map_double'].apply(eval)

    edge_index_save_path = f'{save_path}/edge_index/class_to_class'
    node_feature_save_path = f'{save_path}/node_feature'
    os.makedirs(edge_index_save_path, exist_ok=True)
    os.makedirs(node_feature_save_path, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(edge_index), f'{edge_index_save_path}/edge_index.pq', compression=None)
    pq.write_table(pa.Table.from_pandas(node_feature), f'{node_feature_save_path}/node_feature.pq', compression=None)
    with open(f'{save_path}/class_node_map.json', 'w') as f:
        json.dump(class_node_map, f)
    print('Data Saved!')

if __name__ == '__main__':
    api_key_path = "/Users/chenyi/Documents/sag/Final_Project/code/API-KEY.txt"
    save_path = '/Users/chenyi/Documents/sag/Final_Project/data/graph_data'
    database = sys.argv[1] if len(sys.argv) > 1 else 'train' # [train , test, eval, inference]

    node_feature, edge_index, class_node_map = build_graph_data(database=database, api_key_path=api_key_path)

    save_path = os.path.join(save_path, database)
    save_data(node_feature, edge_index, class_node_map, save_path)