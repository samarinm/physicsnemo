#!/bin/bash

# Set to fail on error
set -e

#INPUT_TYPE="eobs"
INPUT_TYPE="coarse"

CONFIG="config_training_"$INPUT_TYPE"_mch_regression.yaml"
DATA_PATH="/mydata/speed2zero/shared/DeepDown/"
STATS_PATH="/myhome/physicsnemo/corrdiff/datasets/stats/stats_"$INPUT_TYPE"_mch.json"
TMP_PATH="/mydata/speed2zero/shared/DeepDown/tmp/target_pickle/"
PERIOD="train-period"


echo "PERIOD: $PERIOD"






HYDRA_FULL_ERROR=1

/usr/bin/python /myhome/physicsnemo/corrdiff/train.py --config-name=$CONFIG ++dataset.data_path=$DATA_PATH ++dataset.stats_path=$STATS_PATH ++dataset.tmp_path=$TMP_PATH ++dataset.period=$PERIOD
