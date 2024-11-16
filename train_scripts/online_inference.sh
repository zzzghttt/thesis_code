#!/bin/bash
set -e
set -o pipefail

COUNTRY=$1
CONFIG_FILE_PATH=$2

###############################################################################################
# 日期获取
RUN_DAY_TIMESTAMP=$((RH2_RUNDAY/1000))
FORMAT_DATE=$(date -d @$RUN_DAY_TIMESTAMP +"%Y%m%d")
echo "Formatted date: $FORMAT_DATE"

DORADO_DATE=$(date -d "$FORMAT_DATE - 1 day" +"%Y%m%d")
echo "Dorado date: $DORADO_DATE"

###############################################################################################
if [[ $CONFIG_FILE_PATH == hdfs* ]]; then
    # 从HDFS拉取文件到本地
    hadoop fs -get $CONFIG_FILE_PATH "./model"
    CONFIG_FILE_PATH="./model/local_inference.yaml"
    if [ $? -eq 0 ]; then
        echo "成功从HDFS拉取配置文件到本地"
    else
        echo "从HDFS拉取配置文件失败"
        exit 1
    fi
else
    exit 1
fi

echo "使用的配置文件为: $CONFIG_FILE_PATH"

###############################################################################################
sleep 10
python3 -u pull_dataset_from_hdfs.py  $DORADO_DATE $COUNTRY $CONFIG_FILE_PATH
echo "pull dataset success"

sleep 10
python3 -u feature_process_inference.py  $DORADO_DATE $COUNTRY $CONFIG_FILE_PATH
echo "convert to pyg graph success"

sleep 10
python3 -u inference.py  $DORADO_DATE $COUNTRY $CONFIG_FILE_PATH
echo "inference success"