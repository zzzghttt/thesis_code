import pymysql
import sys
import os
import torch
import json
import pandas as pd
import numpy as np
from pyspark.sql.functions import when, col, size
from pyspark.sql import SparkSession, Row
from torch_geometric.utils import to_undirected, k_hop_subgraph
from table_config import *
import openai

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

spark = SparkSession.builder.master("local[*]").appName("sparkSql").getOrCreate()

# 连接到数据库的函数
def connect_to_database(database_name='train'):
    connection = pymysql.connect(
        host='localhost',
        user='root',
        password='yueyujia520C@',
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )
    try:
        with connection.cursor() as cursor:
            # 检查数据库是否存在
            cursor.execute(f"SHOW DATABASES LIKE '{database_name}'")
            result = cursor.fetchone()
            if not result:
                # 如果数据库不存在，则创建
                cursor.execute(f"CREATE DATABASE {database_name} CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci")
                print(f"Database '{database_name}' created successfully.")
        # 切换到指定的数据库
        connection.select_db(database_name)
        # print(f"Connected to the database '{database_name}'.")
        return connection
    except Exception as e:
        print("Error occurred:", e)
        connection.close()
        raise e

# 获取数据并转换为 pandas DataFrame 的函数
def pull_data(sql, database='train'):
    connection = connect_to_database(database)
    try:
        with connection.cursor() as cursor:
            # 执行查询           
            cursor.execute(sql)
            result = cursor.fetchall()
            # 将结果转换为 pandas DataFrame
            df_pandas = pd.DataFrame(result)
            return spark.createDataFrame(df_pandas)
    finally:
        connection.close()

def create_table_if_not_exists(connection, table_name, columns):
    try:
        with connection.cursor() as cursor:
            # 构建 CREATE 语句
            column_definitions = ', '.join([f"`{col}` {type}" for col, type in columns.items()])
            create_sql = f"CREATE TABLE IF NOT EXISTS {table_name} ({column_definitions})"
            # 执行创建表操作
            cursor.execute(create_sql)
    finally:
        connection.commit()

def push_to_mysql(rdd, table_name, columns, database='train'):
    # 遍历 RDD 中的每个元素
    for partition_rdd in rdd.glom().collect():
        # 为每个分区创建数据库连接
        connection = connect_to_database(database)
        try:
            # 检查表是否存在，如果不存在则创建
            create_table_if_not_exists(connection, table_name, columns)
            
            with connection.cursor() as cursor:
                for row in partition_rdd:
                    # 构建 INSERT 语句
                    placeholders = ', '.join(['%s'] * len(row))
                    sql = f"REPLACE INTO {table_name} VALUES ({placeholders})"
                    # 执行插入操作
                    cursor.execute(sql, list(row))
            # 提交事务
            connection.commit()
        finally:
            connection.close()

def read_all_data(
        data_source_path='/Users/chenyi/Documents/sag/Final_Project/code/data_deploy/chatunitest-info'
        ):
    # 找到所有.json文件并存储在列表中
    class_json_files = [os.path.join(dirpath, file)
                for dirpath, dirnames, files in os.walk(data_source_path)
                for file in files if file.endswith('class.json')]

    method_json_files = [os.path.join(dirpath, file)
                for dirpath, dirnames, files in os.walk(data_source_path)
                for file in files if file.endswith('.json') and not file.endswith('class.json')]

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

    class_df.drop_duplicates(subset=['fullClassName'], keep='last', inplace=True)
    method_df.drop_duplicates(subset=['packageName', 'className', 'methodSignature'], keep='last', inplace=True)
    return class_df, method_df

def convert_empty_list_to_NULL(df):
    for column_name in df.columns:
        if "array" in df.schema[column_name].dataType.simpleString(): 
            df = df.withColumn(column_name, when(size(col(column_name)) == 0, None).otherwise(col(column_name)))
    return df

def is_match(d1, d2, d1_id, d2_id):
    '''
    应该在同一项目或模块下（不考虑跨模块）
    '''
    d1_module_name = d1.loc[d1['index'] == d1_id, 'moduleName'].values[0]
    d2_module_name = d2.loc[d2['index'] == d2_id, 'moduleName'].values[0]
    if d1_module_name == d2_module_name:
        return True
    else:
        return False

def match_dict(src_id_list, src_df, dst_df, src_col_name, dst_col_name, is_match_key=True):
    '''
    在 dst_df 中匹配 src_df 的字典元素（可选对key或value进行匹配）
    '''
    source_nodes = []
    target_nodes = []
    # 创建名称到 ID 的映射
    name_to_idx = {name: idx for idx, name in zip(dst_df['index'], dst_df[dst_col_name])}
    
    for src_id in src_id_list:
        row  = src_df.loc[src_df['index'] == src_id].iloc[0]
        col_data = eval(row[src_col_name])
            
        for key in col_data.keys():    
            if is_match_key:
                if key in name_to_idx and is_match(src_df, dst_df, src_id, name_to_idx[key]):
                    source_nodes.append(src_id)
                    target_nodes.append(name_to_idx[key])
            else:
                val_list = col_data[key]
                assert type(val_list) == list
                for value in val_list:
                    if value in name_to_idx and is_match(src_df, dst_df, src_id, name_to_idx[value]):
                        source_nodes.append(src_id)
                        target_nodes.append(name_to_idx[value])
    
    return np.unique(np.array([source_nodes, target_nodes]), axis=1)

def match_list(src_id_list, src_df, dst_df, src_col_name, dst_col_name):
    '''
    在 dst_df 中匹配src_df的列表元素
    '''
    source_nodes = []
    target_nodes = []
    # 创建名称到 ID 的映射
    name_to_idx = {name: idx for idx, name in zip(dst_df['index'], dst_df[dst_col_name])}
    
    # 遍历 dataframe2 中的每一行
    for src_id in src_id_list:
        row  = src_df.loc[src_df['index'] == src_id].iloc[0]
        col_data = eval(row[src_col_name])
        
        for value in col_data:
            if value in name_to_idx and is_match(src_df, dst_df, src_id, name_to_idx[value]):
                source_nodes.append(src_id)
                target_nodes.append(name_to_idx[value])
            
    return np.unique(np.array([source_nodes, target_nodes]), axis=1)

def build_class_to_class(class_to_method, method_to_class):
    # 步骤1：构建方法到类的映射
    method_to_classes = {}
    for method_id, class_id in method_to_class.T:
        if method_id not in method_to_classes:
            method_to_classes[method_id] = set()
        method_to_classes[method_id].add(class_id)
    
    # 步骤2：构建类到类的边关系
    class_to_class = []
    for class_id, method_id in class_to_method.T:
        if method_id in method_to_classes:
            # 将类ID和它包含的方法依赖的类ID添加到结果列表中
            for dependent_class_id in method_to_classes[method_id]:
                class_to_class.append([class_id, dependent_class_id])
    
    return np.unique(np.array(class_to_class).T, axis=1)

def edge_index_append(array, type):
    original_array = array.T
    # 创建一列全为1的数据
    ones_column = np.ones((original_array.shape[0], 1), dtype=original_array.dtype)
    
    # 创建一列字符串数据
    string_column = np.array([[type] * original_array.shape[0]]).T
    
    # 在原始数组上增加这两列
    new_array = np.append(original_array, ones_column, axis=1)
    df = pd.DataFrame(new_array, columns=['src_id', 'dst_id', 'weight'])
    df['type'] = string_column
    df = df.astype({'src_id': int, 'dst_id': int, 'weight': float, 'type': 'string'})
    return df

def find_class_by_test(df, test_id_list) -> list:
    class_list = []
    test_list = []
    for test_id in test_id_list:
        test_name = df.loc[test_id, 'fullClassName']
        if (test_name.endswith('Test')):
            fc_name = test_name[:-4]
        elif (test_name.endswith('Tests')):
            fc_name = test_name[:-5]
        else:
            continue
        # print(df.loc[test_id, 'fullClassName'], '\n' + fc_name, '\n', '-' * 20)
        class_id = df[df['fullClassName'] == fc_name]['index']
        if not class_id.empty:
            class_list.append(int(class_id.values[0]))
            test_list.append(test_id)
    return test_list, class_list

def remap_node_edge(edge_index, class_list, test_list=None, k_hop=2):
    # 初始化新的节点特征和边列表
    new_edge_index = []
    group_ids = []
    labels = []
    node_map = {}  # 记录旧节点到新节点的映射
    current_idx = 0
    
    # 遍历每个源节点，生成其两跳子图
    for i, src_node in enumerate(class_list):
        
        # 使用 k_hop_subgraph 提取两跳子图
        sub_src_nodes, sub_src_edge_index, _, _ = k_hop_subgraph(
            src_node, k_hop, edge_index, flow='target_to_source', relabel_nodes=False)
        
        if test_list is not None:
            test_node = test_list[i]
            sub_test_nodes, _, _, _ = k_hop_subgraph(
                test_node, k_hop, edge_index, flow='target_to_source', relabel_nodes=False)
            sub_test_nodes = sub_test_nodes.tolist()
        else:
            sub_test_nodes = []

        sub_labels = []
        # 为当前子图的节点重新分配新的索引
        local_map = {}  # 用于存储子图中重新编号的节点映射
        for node in sub_src_nodes.tolist():
            if node not in local_map:
                node_map[current_idx] = node
                local_map[node] = current_idx
                current_idx += 1

                if node == src_node or node in sub_test_nodes:
                    sub_labels.append(1)
                else:
                    sub_labels.append(0)
        
        # 重新映射边    
        remapped_edges = torch.tensor([[local_map[edge[0]], local_map[edge[1]], src_node]
                                       for edge in sub_src_edge_index.t().tolist()], dtype=torch.long).t()
        # print(src_node, '\n', sub_nodes, '\n', sub_edge_index, '\n', remapped_edges,  '\n', '-'*20)
    
        if new_edge_index == []:
            new_edge_index = remapped_edges
        else:
            new_edge_index = torch.cat([new_edge_index, remapped_edges], dim=1)
        
        labels += sub_labels
        group_ids += [src_node] * len(local_map)
    
    new_edge_index = np.unique(np.array(new_edge_index), axis=1).tolist()

    return new_edge_index, node_map, group_ids, labels


def get_embedding(text: str, key_path: str = 'openai_api_key.txt'):
    """使用 OpenAI API 获取标识符的嵌入"""
    with open(key_path, 'r') as f:
        openai.api_key = f.read().strip()
    try:
        response = openai.Embedding.create(
            model="text-embedding-3-small",
            input=text[:4096],
            dimensions=64
        )
        # 返回嵌入
        return response['data'][0]['embedding']
    except Exception as e:
        raise ValueError("Failed to get embeddings for package name, class name, or class declaration")