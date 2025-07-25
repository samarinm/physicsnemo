#!/bin/bash

# Set to fail on error
set -e

#INPUT_TYPE="eobs"
INPUT_TYPE="coarse"

CONFIG="config_training_"$INPUT_TYPE"_mch_regression.yaml"
DATA_PATH="/mydata/speed2zero/shared/DeepDown/"
STATS_PATH="/myhome/physicsnemo/corrdiff/datasets/stats/stats_"$INPUT_TYPE"_mch.json"
TMP_PATH="/mydata/speed2zero/shared/DeepDown/tmp/target_pickle/"
#PERIOD="train-period"

echo "PREDID: $PREDID"
echo "PERIOD: $PERIOD"

REGCKPT="/mydata/speed2zero/shared/DeepDown/physicsnemo/corrdiff/outputs/"$INPUT_TYPE"_mch_regression/checkpoints_regression/UNet.0.4000000.mdlus"
DIFFCKPT="/mydata/speed2zero/shared/DeepDown/physicsnemo/corrdiff/outputs/"$INPUT_TYPE"_mch_diffusion/checkpoints_diffusion/EDMPrecondSR.0.26000000.mdlus"
PRED="/mydata/speed2zero/shared/DeepDown/physicsnemo/corrdiff/outputs/predictions/corrdiff_pred_id-"$PREDID"_sample-10_"$PERIOD"_$(date '+%d%m%y_%H%M%S').nc"


HYDRA_FULL_ERROR=1

/usr/bin/python /myhome/physicsnemo/corrdiff/generate.py --config-name=$CONFIG ++dataset.data_path=$DATA_PATH ++dataset.stats_path=$STATS_PATH ++dataset.tmp_path=$TMP_PATH ++dataset.period=$PERIOD ++generation.io.regression_checkpoint_path=$REGCKPT ++generation.io.res_ckpt_filename=$DIFFCKPT ++generation.io.output_filename=$PRED
