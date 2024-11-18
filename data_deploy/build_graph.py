# %%
import torch
import sys
import os
import numpy as np
import pandas as pd
import json
import pymysql
import cryptography
from datetime import datetime, date
from pyspark.sql import SparkSession, Row
from pyspark.sql.types import StructType, StructField, StringType, BooleanType, IntegerType, ArrayType, DoubleType
from hashlib import sha256, md5
from util import *
from table_config import *
from torch_geometric.data import HeteroData
from torch_geometric.utils import to_undirected, k_hop_subgraph

# %% [markdown]
# ## SQL Statements (Create Partition Table)

# %%
sql = '''
CREATE TABLE IF NOT EXISTS edge_data
(
    src_id BIGINT,
    dst_id BIGINT,
    weight FLOAT,
    edge_type VARCHAR(255)
)
PARTITION BY LIST COLUMNS(edge_type)(
    PARTITION class_to_class VALUES IN ('class_to_class'),
    PARTITION class_to_method VALUES IN ('class_contain_method'),
    PARTITION method_to_class VALUES IN ('method_to_class'),
    PARTITION method_to_method VALUES IN ('method_to_method')
    );

INSERT INTO edge_data
SELECT src_id, dst_id, weight, type FROM edge_class_to_class
UNION ALL
SELECT src_id, dst_id, weight, type FROM edge_class_to_method
UNION ALL
SELECT src_id, dst_id, weight, type FROM edge_method_to_class
UNION ALL
SELECT src_id, dst_id, weight, type FROM edge_method_to_method;
'''

# %%
sql = '''
select * from edge_data
'''
# sp_res = pull_data(sql)

# c2c_sp_df.select('src_id', 'dst_id').toPandas()
# sp_res.select('src_id', 'dst_id').where(sp_res['edge_type']=='class_to_class').toPandas()

# %% [markdown]
# ## Graph Data (Use Class Node Only)

# %%
sp_res = pull_data(
    '''
    select * from class_processed_data
    '''
)

c2c_sp_df = pull_data(
    '''
    select * from edge_class_to_class
    '''
)

# %%
# Test Node
target_class_node = sp_res.where('isTest == 1').distinct()
target_class_node_list = target_class_node.toPandas()['index'].tolist()

# target_method_node = new_sp_df2.where('isTest == 1').distinct()
# target_method_node_list = target_method_node.toPandas()['index'].tolist()

# Source Class Node
test_list, class_list = find_class_by_test(sp_res.toPandas(), target_class_node_list)

# %%
edge_index = torch.from_numpy(c2c_sp_df.select('src_id', 'dst_id').toPandas().values).T

test_edge_index, test_node_map = remap_node_edge(test_list, edge_index)
class_edge_index, class_node_map = remap_node_edge(class_list, edge_index)

# %%
class_edge_index_t = np.array(class_edge_index).T
test_edge_index_t = np.array(test_edge_index).T
test_edge_index_ori = [[test_node_map[src], test_node_map[dst]] for src, dst in test_edge_index_t]

labels = np.zeros(len(class_node_map))

for pair in class_edge_index_t:
    src_id = pair[0]
    dst_id = pair[1]
    if [class_node_map[src_id], class_node_map[dst_id]] in test_edge_index_ori:
        labels[src_id] = 1
        labels[dst_id] = 1
labels

# %%
node_feature_string = sp_res.select(
    'packageName',
    'className'
).toPandas().iloc[list(class_node_map.values())]

node_feature_double = sp_res.select(
    'line_count', 
    'import_count', 
    'super_class_count',
    'implement_count',
    'method_count',
    'constructor_count',
    'constructor_dep_count',
    'sub_class_count'
).toPandas().iloc[list(class_node_map.values())]

# %%
# node feature
columns_1 = ['id', 'label', 'feature_map_string', 'feature_map_double']
node_feature = pd.DataFrame(columns=columns_1)
node_feature['id'] = list(class_node_map.keys())
node_feature['label'] = labels.tolist()
node_feature['feature_map_string'] = node_feature_string.values.tolist()
node_feature['feature_map_double'] = node_feature_double.values.tolist()
node_feature = node_feature.astype({'id': int, 'label': int, 'feature_map_string': 'string', 'feature_map_double': 'string'})

# %%
# edge_index
columns_2 = ['src_id', 'dst_id', 'weight', 'type']
edge_index = pd.DataFrame(columns=columns_2)
edge_index['src_id'] = class_edge_index[0]
edge_index['dst_id'] = class_edge_index[1]
edge_index['weight'] = 1
edge_index['weight'] = edge_index['weight'].astype(float)
edge_index['type'] = 'class_to_class'
edge_index = edge_index.astype({'src_id': int, 'dst_id': int, 'weight': float, 'type': 'string'})

# %%
sp_node_feature = convert_empty_list_to_NULL(spark.createDataFrame(node_feature))
sp_edge_index = convert_empty_list_to_NULL(spark.createDataFrame(edge_index))

# %%
# push_to_mysql(sp_node_feature.rdd, 'final_node_feature', node_feature_table_columns)
# push_to_mysql(sp_edge_index.rdd, 'final_edge_index', edge_index_table_columns)

# %% [markdown]
# # Save Data

# %%
import pyarrow as pa
import pyarrow.parquet as pq

# %%
node_feature['feature_map_string'] = node_feature['feature_map_string'].apply(eval)
node_feature['feature_map_double'] = node_feature['feature_map_double'].apply(eval)

# %%
save_path = '/Users/chenyi/Documents/sag/Final_Project/data/train'
edge_index_save_path = f'{save_path}/edge_index/class_to_class'
node_feature_save_path = f'{save_path}/node_feature'
os.makedirs(edge_index_save_path, exist_ok=True)
os.makedirs(node_feature_save_path, exist_ok=True)
pq.write_table(pa.Table.from_pandas(edge_index), f'{edge_index_save_path}/edge_index.pq', compression=None)
pq.write_table(pa.Table.from_pandas(node_feature), f'{node_feature_save_path}/node_feature.pq', compression=None)

# %%



