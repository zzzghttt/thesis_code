import numpy as np
import sys
from util import *
from table_config import *

def process_project(process_path, database):
    # 数据处理

    # Get Raw Data
    class_df, method_df = read_all_data(process_path)

    # 转换spark DataFrame 并写入mysql表
    sp_df1 = convert_empty_list_to_NULL(spark.createDataFrame(class_df))
    sp_df2 = convert_empty_list_to_NULL(spark.createDataFrame(method_df))

    # Process RAW Data

    df1 = sp_df1.toPandas()
    df2 = sp_df2.toPandas()

    processed_class_df = df1.drop('index', axis=1)
    processed_class_df = processed_class_df.reset_index()
    processed_class_df['import_count'] = processed_class_df['imports'].map(len)
    processed_class_df['field_count'] = processed_class_df['fields'].map(len)
    processed_class_df['super_class_count'] = processed_class_df['superClasses'].map(len)
    processed_class_df['implement_count'] = processed_class_df['implementedTypes'].map(len)
    processed_class_df['method_count'] = processed_class_df['methodSigs'].map(len)
    processed_class_df['constructor_count'] = processed_class_df['constructorSigs'].map(len)
    processed_class_df['constructor_dep_count'] = processed_class_df['constructorDeps'].map(len)
    processed_class_df['sub_class_count'] = processed_class_df['subClasses'].map(len)

    processed_method_df = df2.reset_index()
    processed_method_df['parameter_count'] = processed_method_df['parameters'].map(len)
    processed_method_df['method_dep_count'] = processed_method_df['dependentMethods'].map(len)

    # 转换spark DataFrame 并写入mysql表
    new_sp_df1 = convert_empty_list_to_NULL(spark.createDataFrame(processed_class_df))
    new_sp_df2 = convert_empty_list_to_NULL(spark.createDataFrame(processed_method_df))

    # 构图

    # method - rely - class, method - rely - method
    edge_index_method_rely_class_1 = match_dict(processed_method_df['index'].tolist(), processed_method_df, processed_class_df, 'dependentMethods', 'fullClassName')
    edge_index_method_rely_class_2 = match_list(processed_method_df['index'].tolist(), processed_method_df, processed_class_df, 'parameters', 'fullClassName')
    edge_index_method_rely_class = np.concatenate((edge_index_method_rely_class_1, edge_index_method_rely_class_2),axis=1)
    edge_index_method_rely_class = np.unique(edge_index_method_rely_class, axis=1)

    edge_index_method_rely_method = match_dict(processed_method_df['index'].tolist(), processed_method_df, processed_method_df, 'dependentMethods', 'methodSignature', False)

    # print(edge_index_method_rely_class.shape)
    # print(edge_index_method_rely_method.shape)

    # class - rely - class, class - rely - method, class - contain - method

    # edge_index_class_rely_method = match_dict(processed_class_df['index'].tolist(), processed_class_df, processed_method_df, 'constructorDeps', 'fullClassName', False) # no use
    edge_index_class_contain_method = match_dict(processed_class_df['index'].tolist(), processed_class_df, processed_method_df, 'methodSigs', 'methodSignature') # methodSigs包含了构造函数

    edge_index_class_rely_class_1 = match_dict(processed_class_df['index'].tolist(), processed_class_df, processed_class_df, 'constructorDeps', 'fullClassName')
    edge_index_class_rely_class_2 = match_list(processed_class_df['index'].tolist(), processed_class_df, processed_class_df, 'importTypes', 'fullClassName')
    edge_index_class_rely_class_3 = match_list(processed_class_df['index'].tolist(), processed_class_df, processed_class_df, 'fieldDeps', 'fullClassName')
    edge_index_class_method_rely_class = build_class_to_class(edge_index_class_contain_method, edge_index_method_rely_class)

    # Test Node
    target_class_node = new_sp_df1.where('isTest == 1').distinct()
    target_method_node = new_sp_df2.where('isTest == 1').distinct()

    target_class_node_list = target_class_node.toPandas()['index'].tolist()
    target_method_node_list = target_method_node.toPandas()['index'].tolist()

    # Source Class Node - Test list pair
    test_list, class_list = find_class_by_test(processed_class_df, target_class_node_list)
    edge_index_class_rely_class = np.concatenate((np.array([test_list, class_list]), edge_index_class_rely_class_1, edge_index_class_rely_class_2, edge_index_class_rely_class_3, edge_index_class_method_rely_class),axis=1)
    edge_index_class_rely_class = np.unique(edge_index_class_rely_class, axis=1)

    # print(edge_index_class_rely_method.shape)
    print('Edge index:')
    print(edge_index_class_contain_method.shape)
    print(edge_index_class_method_rely_class.shape)
    print(edge_index_class_rely_class_3.shape)
    print(edge_index_class_rely_class.shape)

    # Edge Data: class-class, class-method, method-class, method-method

    m2m_sp_df = convert_empty_list_to_NULL(spark.createDataFrame(edge_index_append(edge_index_method_rely_method, 'method_to_method')))
    m2c_sp_df = convert_empty_list_to_NULL(spark.createDataFrame(edge_index_append(edge_index_method_rely_class, 'method_to_class')))
    c2m_sp_df = convert_empty_list_to_NULL(spark.createDataFrame(edge_index_append(edge_index_class_contain_method, 'class_contain_method')))
    c2c_sp_df = convert_empty_list_to_NULL(spark.createDataFrame(edge_index_append(edge_index_class_rely_class, 'class_to_class')))


    push_to_mysql(new_sp_df1.rdd, 'class_processed_data', class_processed_table_columns, database)
    push_to_mysql(new_sp_df2.rdd, 'method_processed_data', method_processed_table_columns, database)
    push_to_mysql(sp_df1.rdd, 'class_raw_data', class_table_columns, database)
    push_to_mysql(sp_df2.rdd, 'method_raw_data', method_table_columns, database)
    push_to_mysql(m2m_sp_df.rdd, 'edge_method_to_method', edge_index_table_columns, database)
    push_to_mysql(m2c_sp_df.rdd, 'edge_method_to_class', edge_index_table_columns, database)
    push_to_mysql(c2m_sp_df.rdd, 'edge_class_to_method', edge_index_table_columns, database)
    push_to_mysql(c2c_sp_df.rdd, 'edge_class_to_class', edge_index_table_columns, database)

if __name__ == '__main__':
    process_path = '/Users/chenyi/Documents/sag/Final_Project/data/raw_data'
    database = sys.argv[1] if len(sys.argv) > 1 else 'train'
    
    process_path = os.path.join(process_path, database)
    process_project(process_path, database)