#!/usr/bin/env bash

BIN_PATH=/home/kenzosaki/repos/llama.cpp/build/bin/

# 26B
HF_REPO="unsloth/gemma-4-26B-A4B-it-GGUF"
HF_FILE="gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"
# 31B
# HF_REPO="unsloth/gemma-4-31B-it-GGUF"
# HF_FILE="gemma-4-31B-it-Q4_K_M.gguf"

REASONING="off" # on/off - para nao poluir a saida do json
CONTEXT_SIZE=2048
TO_KEEP=2048 # n tokens do input a manter
NUM_GPU_LAYERS="auto" # gpu layers
FIT="on" # ajuste automatico de parametros para caber na memoria
HOST=127.0.0.1
PORT=8080
N_PARALLEL=12 # quantidade de reqs em paralelo (default: 1). 12 para 26B. 4 para 31B
VERBOSE=1 # 2 mostra warnings, 1 mostra apenas erros. 

echo "Launching llama server with the following parameters:"
echo "- HF_REPO: $HF_REPO"
echo "- HF_FILE: $HF_FILE"
echo "- CONTEXT_SIZE: $CONTEXT_SIZE"
echo "- TO_KEEP: $TO_KEEP"
echo "- NUM_GPU_LAYERS: $NUM_GPU_LAYERS"
echo "- FIT: $FIT"
echo "- HOST: $HOST"
echo "- PORT: $PORT"
echo "- N_PARALLEL: $N_PARALLEL"
echo "- VERBOSE: $VERBOSE"  

$BIN_PATH/llama-server \
  --hf-repo $HF_REPO \
  --hf-file $HF_FILE \
  --host $HOST \
  --port $PORT \
  --keep $TO_KEEP \
  --fit $FIT \
  --ctx-size $(( $CONTEXT_SIZE * $N_PARALLEL )) \
  -ngl $NUM_GPU_LAYERS \
  -np $N_PARALLEL \
  --reasoning $REASONING \
  -lv $VERBOSE