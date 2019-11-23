#!/usr/bin/env bash

#model params
EMBEDDING_HIDDEN_DIM=64
NUM_HIDDEN_LAYER=1
GRU_HIDDEN_DIM=512
DROPOUT_P=0.1
ATTENTION_METHOD="dot"

#train params
BATCH_SIZE=2048
EVAL_BATCH_SIZE=512
LEARNING_RATE=3e-5
EPOCHS=1
EVAL_STEP=1
LOGGING_STEP=100
GRAD_CLIP_NORM=1.0

#other parameters
NUM_WORKERS=8
DEVICE="cuda"
FP16=0
FP16_OPT_LEVEL="O1"
SEED=0

#run traininer
for fp in 0 1; do
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
		--fp16=${fp}\
		--fp16_opt_level=${FP16_OPT_LEVEL}\
		--seed=${SEED}
done
