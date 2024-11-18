# %%
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

# 数据处理

# Get Raw Data
data_source_path = '/Users/chenyi/Documents/sag/Final_Project/data_deploy/chatunitest-info'
class_json_files, method_json_files = find_all_json(data_source_path)

# 存储所有DataFrame的列表
class_df_list = []
method_df_list = []

for json_file in class_json_files:
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        df = pd.DataFrame([data])
        class_df_list.append(df)

for json_file in method_json_files:
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        df = pd.DataFrame([data])
        method_df_list.append(df)

class_df = pd.concat(class_df_list, ignore_index=True)
method_df = pd.concat(method_df_list, ignore_index=True)

# 设置预定义类型
for key, dtype in expected_types.items():
    if key in class_df.columns:
        class_df[key] = class_df[key].astype(dtype)
    if key in method_df.columns:
        method_df[key] = method_df[key].astype(dtype)

# %%
class_df.drop_duplicates(subset=['fullClassName'], keep='last', inplace=True)
method_df.drop_duplicates(subset=['packageName', 'className', 'methodSignature'], keep='last', inplace=True)

# %%
# 转换spark DataFrame 并写入mysql表

sp_df1 = convert_empty_list_to_NULL(spark.createDataFrame(class_df))
# push_to_mysql(sp_df1.rdd, 'class_raw_data', class_table_columns)

sp_df2 = convert_empty_list_to_NULL(spark.createDataFrame(method_df))
# push_to_mysql(sp_df2.rdd, 'method_raw_data', method_table_columns)

# %% [markdown]
# ## Class Raw Data

# %%
sp_df1.toPandas()

# %% [markdown]
# ## Method Raw Data

# %%
sp_df2.toPandas()

# %% [markdown]
# ## Process RAW Data

# %% [markdown]
# ### 计算 Hash (不使用)

# %%
df1 = sp_df1.toPandas()
df2 = sp_df2.toPandas()

# %%
df1_hash = df1.apply(lambda col: col.apply(lambda x: process_element(x, col.name)), axis=0)
df2_hash = df2.apply(lambda col: col.apply(lambda x: process_element(x, col.name)), axis=0)

# %%
df1_hash = df1_hash.drop('index', axis=1)
df1_hash = df1_hash.reset_index()
df1_hash['import_count'] = df1_hash['imports'].map(len)
df1_hash['field_count'] = df1_hash['fields'].map(len)
df1_hash['super_class_count'] = df1_hash['superClasses'].map(len)
df1_hash['implement_count'] = df1_hash['implementedTypes'].map(len)
df1_hash['method_count'] = df1_hash['methodSigs'].map(len)
df1_hash['constructor_count'] = df1_hash['constructorSigs'].map(len)
df1_hash['constructor_dep_count'] = df1_hash['constructorDeps'].map(len)
df1_hash['sub_class_count'] = df1_hash['subClasses'].map(len)
df1_hash

# %%
df2_hash = df2_hash.reset_index()
df2_hash['parameter_count'] = df2_hash['parameters'].map(len)
df2_hash['method_dep_count'] = df2_hash['dependentMethods'].map(len)
df2_hash

# %%
# 转换spark DataFrame 并写入mysql表

new_sp_df1 = convert_empty_list_to_NULL(spark.createDataFrame(df1_hash))
# push_to_mysql(new_sp_df1.rdd, 'class_processed_data', class_processed_table_columns)

new_sp_df2 = convert_empty_list_to_NULL(spark.createDataFrame(df2_hash))
# push_to_mysql(new_sp_df2.rdd, 'method_processed_data', method_processed_table_columns)

# %% [markdown]
# # 构图

# %% [markdown]
# ## method - rely - class, method - rely - method

# %%
edge_index_method_rely_class_1 = match_dict(df2_hash['index'].tolist(), df2_hash, df1_hash, 'dependentMethods', 'fullClassName')
edge_index_method_rely_class_2 = match_list(df2_hash['index'].tolist(), df2_hash, df1_hash, 'parameters', 'fullClassName')
edge_index_method_rely_class = np.concatenate((edge_index_method_rely_class_1, edge_index_method_rely_class_2),axis=1)
edge_index_method_rely_class = np.unique(edge_index_method_rely_class, axis=1)

edge_index_method_rely_method = match_dict(df2_hash['index'].tolist(), df2_hash, df2_hash, 'dependentMethods', 'methodSignature', False)

# %%
print(edge_index_method_rely_class.shape)
print(edge_index_method_rely_method.shape)

# %% [markdown]
# ## class - rely - class, class - rely - method, class - contain - method

# %%
# edge_index_class_rely_method = match_dict(df1_hash['index'].tolist(), df1_hash, df2_hash, 'constructorDeps', 'fullClassName', False) # no use
edge_index_class_contain_method = match_dict(df1_hash['index'].tolist(), df1_hash, df2_hash, 'methodSigs', 'methodSignature') # methodSigs包含了构造函数

edge_index_class_rely_class_1 = match_dict(df1_hash['index'].tolist(), df1_hash, df1_hash, 'constructorDeps', 'fullClassName')
edge_index_class_rely_class_2 = match_list(df1_hash['index'].tolist(), df1_hash, df1_hash, 'importTypes', 'fullClassName')
edge_index_class_method_rely_class = build_class_to_class(edge_index_class_contain_method, edge_index_method_rely_class)

# %%
# Test Node
target_class_node = new_sp_df1.where('isTest == 1').distinct()
target_method_node = new_sp_df2.where('isTest == 1').distinct()

target_class_node_list = target_class_node.toPandas()['index'].tolist()
target_method_node_list = target_method_node.toPandas()['index'].tolist()

# Source Class Node - Test list pair
test_list, class_list = find_class_by_test(df1_hash, target_class_node_list)
edge_index_class_rely_class = np.concatenate((np.array([test_list, class_list]), edge_index_class_rely_class_1, edge_index_class_rely_class_2, edge_index_class_method_rely_class),axis=1)
edge_index_class_rely_class = np.unique(edge_index_class_rely_class, axis=1)

# %%
# print(edge_index_class_rely_method.shape)
print(edge_index_class_contain_method.shape)
print(edge_index_class_method_rely_class.shape)
print(edge_index_class_rely_class.shape)

# %% [markdown]
# ## Edge Data: class-class, class-method, method-class, method-method

# %%
m2m_sp_df = convert_empty_list_to_NULL(spark.createDataFrame(edge_index_append(edge_index_method_rely_method, 'method_to_method')))
m2c_sp_df = convert_empty_list_to_NULL(spark.createDataFrame(edge_index_append(edge_index_method_rely_class, 'method_to_class')))
c2m_sp_df = convert_empty_list_to_NULL(spark.createDataFrame(edge_index_append(edge_index_class_contain_method, 'class_contain_method')))
c2c_sp_df = convert_empty_list_to_NULL(spark.createDataFrame(edge_index_append(edge_index_class_rely_class, 'class_to_class')))

# push_to_mysql(m2m_sp_df.rdd, 'edge_method_to_method', edge_index_table_columns)
# push_to_mysql(m2c_sp_df.rdd, 'edge_method_to_class', edge_index_table_columns)
# push_to_mysql(c2m_sp_df.rdd, 'edge_class_to_method', edge_index_table_columns)
# push_to_mysql(c2c_sp_df.rdd, 'edge_class_to_class', edge_index_table_columns)

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
# # TODO: 对所有三方依赖库的类名构建节点


