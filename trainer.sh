#!/usr/bin/env bash

#model params
EMBEDDING_HIDDEN_DIM=128
NUM_HIDDEN_LAYER=1
GRU_HIDDEN_DIM=512
DROPOUT_P=0.1
ATTENTION_METHOD="dot"

#train params
BATCH_SIZE=1024
EVAL_BATCH_SIZE=512
LEARNING_RATE=3e-5
EPOCHS=5
EVAL_STEP=1
LOGGING_STEP=1000
GRAD_CLIP_NORM=1.0

#other parameters
NUM_WORKERS=8
DEVICE="cuda"
FP16=1
FP16_OPT_LEVEL="O1"
#SEED=0

#path params
SAVE_PATH="layer=${NUM_HIDDEN_LAYER}.hidden=${GRU_HIDDEN_DIM}.batch=${BATCH_SIZE}.epoch=${EPOCHS}"

#run traininer
for SEED in {1..4}
do
	TMP_PATH="${SAVE_PATH}/seed${SEED}"

	python train.py\
		--embedding_hidden_dim=${EMBEDDING_HIDDEN_DIM}\
		--num_hidden_layer=${NUM_HIDDEN_LAYER}\
		--gru_hidden_dim=${GRU_HIDDEN_DIM}\
		--dropout_p=${DROPOUT_P}\
		--attention_method=${ATTENTION_METHOD}\
		--batch_size=${BATCH_SIZE}\
		--eval_batch_size=${EVAL_BATCH_SIZE}\
		--learning_rate=${LEARNING_RATE}\
		--logging_step=${LOGGING_STEP}\
		--epochs=${EPOCHS}\
		--eval_step=${EVAL_STEP}\
		--grad_clip_norm=${GRAD_CLIP_NORM}\
		--device=${DEVICE}\
		--fp16=${FP16}\
		--fp16_opt_level=${FP16_OPT_LEVEL}\
		--seed=${SEED}\
		--save_path=${TMP_PATH}
	
	python inference.py\
		--bestmodel_path=${TMP_PATH}
done
