#!/bin/bash
set -e
set -o pipefail

TRIAN_DATE=$1
VAL_DATE=$2
COUNTRY=$3
CONFIG_FILE_PATH=$4
TEST_DATE=$5

###############################################################################################
# 日期检查
if [[ -z "$TRIAN_DATE" ]]; then
    echo "错误：未提供有效的日期参数。请提供日期参数并重新运行脚本。"
    exit 1
else
    echo "训练日期：$TRIAN_DATE"
fi

if [[ -z "$VAL_DATE" ]]; then
    echo "错误：未提供有效的日期参数。请提供日期参数并重新运行脚本。"
    exit 1
else
    echo "验证日期：$VAL_DATE"
fi

if [[ -z "$TEST_DATE" ]]; then
    echo "错误：未提供有效的日期参数。请提供日期参数并重新运行脚本。"
    exit 1
else
    echo "测试日期：$TEST_DATE"
fi

else
    if [[ $CONFIG_FILE_PATH == hdfs* ]]; then
        # 从HDFS拉取文件到本地
        hadoop fs -get $CONFIG_FILE_PATH "./config/config_hdfs.yaml"
        CONFIG_FILE_PATH="./config/config_hdfs.yaml"
        if [ $? -eq 0 ]; then
            echo "成功从HDFS拉取配置文件到本地"
        else
            echo "从HDFS拉取配置文件失败"
            exit 1
        fi
    else
        CONFIG_FILE_PATH="$4"
    fi
fi
echo "使用的配置文件为: $CONFIG_FILE_PATH"
###############################################################################################

sleep 10
CUDA_VISIBLE_DEVICES=0 python3 -u pyg_homo_model_train.py  $TRIAN_DATE $VAL_DATE $COUNTRY $CONFIG_FILE_PATH
CUDA_VISIBLE_DEVICES=0 python3 -u val_inference_homo.py  $TEST_DATE $COUNTRY $CONFIG_FILE_PATH
