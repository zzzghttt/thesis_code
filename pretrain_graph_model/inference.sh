#!/bin/bash
set -e
set -o pipefail

CONFIG_FILE_PATH=$1

###############################################################################################
# 日期获取
RUN_DAY_TIMESTAMP=$((RH2_RUNDAY/1000))
FORMAT_DATE=$(date -d @$RUN_DAY_TIMESTAMP +"%Y%m%d")
echo "Formatted date: $FORMAT_DATE"

DORADO_DATE=$(date -d "$FORMAT_DATE - 1 day" +"%Y%m%d")
echo "Dorado date: $DORADO_DATE"

python3 -u pull_dataset_from_hdfs.py  $DORADO_DATE $CONFIG_FILE_PATH
echo "pull dataset success"

sleep 10
python3 -u feature_process_inference.py  $DORADO_DATE $CONFIG_FILE_PATH
echo "convert to pyg graph success"

sleep 10
python3 -u inference.py $DORADO_DATE $CONFIG_FILE_PATH NodeTask
echo "inference success"