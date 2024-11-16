# -*- coding: utf-8 -*-
import os
import sys
import logging
import yaml
from utils.utility import run_command

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(filename)s: %(levelname)s: %(message)s'
)

DATE = sys.argv[1] # format 'yyyyMMdd'
COUNTRY = sys.argv[2]
CONFIG_FILE_PATH = sys.argv[3]
logging.info(f'DATE is {DATE}, config file path is {CONFIG_FILE_PATH}')

# config
with open(CONFIG_FILE_PATH, 'r') as file:
    ALL_CONFIG = yaml.safe_load(file)

logging.info(
    '\n{}\n\nDATASET_CONFIG: \n{}\n{}'.format(
        '**' * 80, yaml.dump(ALL_CONFIG['DATASET_PATH'], sort_keys=False, indent=4), '**' * 80
        )
    )

# delete previous download file：
SAVE2FILE_ROOTPATH = ALL_CONFIG['DATASET_PATH']['SAVE2FILE_ROOTPATH']

run_command(f'du -d 3 -h {SAVE2FILE_ROOTPATH}', 'du info', 'fail to run du info files')
# run_command(f'rm -rf {SAVE2FILE_ROOTPATH}', 'delete previous files', 'fail to delete previous files')

# loading node index map
NODE_INDEX_MAP_HDFS_FILEPATH = ALL_CONFIG['DATASET_PATH']['NODE_INDEX_MAP_HDFS_FILEPATH'].format(DATE=DATE, COUNTRY=COUNTRY)
NODE_INDEX_MAP_SAVE2FILE_PATH = ALL_CONFIG['DATASET_PATH']['NODE_INDEX_MAP_SAVE2FILE_PATH'].format(DATE=DATE, COUNTRY=COUNTRY)
os.makedirs(NODE_INDEX_MAP_SAVE2FILE_PATH, exist_ok=True)

logging.info('uid2idx HIVE TABLE STATUS')
run_command(f'hdfs dfs -du -h {NODE_INDEX_MAP_HDFS_FILEPATH}',)
logging.info('uid2idx DOWNLOAD...')
run_command(f'hdfs dfs -get {NODE_INDEX_MAP_HDFS_FILEPATH} {NODE_INDEX_MAP_SAVE2FILE_PATH}')
logging.info('uid2idx FINISHED...')

# loading edge index
EDGE_INDEX_HDFS_PATH = ALL_CONFIG['DATASET_PATH']['EDGE_INDEX_HDFS_PATH'].format(DATE=DATE, COUNTRY=COUNTRY)
EDGE_INDEX_SAVE2FILE_PATH = ALL_CONFIG['DATASET_PATH']['EDGE_INDEX_SAVE2FILE_PATH'].format(DATE=DATE)
ALL_EDGE_TYPES = ALL_CONFIG['DATASET_PATH']['EDGE_TYPES']
os.makedirs(EDGE_INDEX_SAVE2FILE_PATH, exist_ok=True)

logging.info('edge idx HIVE TABLE STATUS')
run_command(f'hdfs dfs -ls {EDGE_INDEX_HDFS_PATH}')
logging.info(f'edge idx DOWNLOAD {ALL_EDGE_TYPES}...')
for edge_type in ALL_EDGE_TYPES:
    logging.info(f'\tDOWNLOAD {edge_type}...')
    tmp_files_path = os.path.join(EDGE_INDEX_SAVE2FILE_PATH, edge_type)
    os.makedirs(tmp_files_path, exist_ok=True)
    run_command(f'hdfs dfs -get {EDGE_INDEX_HDFS_PATH}/edge_type={edge_type}/operation_country={COUNTRY} {tmp_files_path}')
logging.info('edge idx FINISHED...')

# loading node feature
NODE_FEATURE_HDFS_FILEPATH = ALL_CONFIG['DATASET_PATH']['NODE_FEATURE_HDFS_FILEPATH'].format(DATE=DATE, COUNTRY=COUNTRY)
NODE_FEATURE_SAVE2FILE_PATH = ALL_CONFIG['DATASET_PATH']['NODE_FEATURE_SAVE2FILE_PATH'].format(DATE=DATE, COUNTRY=COUNTRY)
os.makedirs(NODE_FEATURE_SAVE2FILE_PATH, exist_ok=True)

logging.info('uid features HIVE TABLE STATUS')
run_command(f'hdfs dfs -ls {NODE_FEATURE_HDFS_FILEPATH}')

logging.info('uid features DOWNLOAD...')
run_command(f'hdfs dfs -get {NODE_FEATURE_HDFS_FILEPATH} {NODE_FEATURE_SAVE2FILE_PATH}')
logging.info('uid features FINISHED...')