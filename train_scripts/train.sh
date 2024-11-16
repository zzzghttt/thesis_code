#!/bin/bash
set -e
set -o pipefail

COUNTRY=$1
CONFIG_FILE_PATH=$2
TRAIN_DATE=$3
VAL_DATE=$4

###############################################################################################
if [ -z "$2" ]; then
    CONFIG_FILE_PATH="./config/inference.yaml"
else
    if [[ $CONFIG_FILE_PATH == hdfs* ]]; then
        # 从HDFS拉取文件到本地
        hadoop fs -get $CONFIG_FILE_PATH "./config/inference_hdfs.yaml"
        CONFIG_FILE_PATH="./config/inference_hdfs.yaml"
        if [ $? -eq 0 ]; then
            echo "成功从HDFS拉取配置文件到本地"
        else
            echo "从HDFS拉取配置文件失败"
            exit 1
        fi
    else
        CONFIG_FILE_PATH="$2"
    fi
fi
echo "使用的配置文件为: $CONFIG_FILE_PATH"

###############################################################################################
sleep 10
python3 -u pyg_hetero_model_train.py  $TRAIN_DATE $VAL_DATE $COUNTRY $CONFIG_FILE_PATH
echo "training success"