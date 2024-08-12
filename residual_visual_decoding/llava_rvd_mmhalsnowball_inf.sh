#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT="llava_1.5_ft_7b"
DATADIR="evaluation"
MODEL_PATH="your_model_path"
VISION_MODEL_PATH="CLIP_VIT_path"

TASK_TYPE="mmhalsnowball"
beta=2
# evaluation conversation settings
declare -a SPLIT_arr=("cleanconv_formatting" "cleanconv_question" "halluconv_formatting" "halluconv_question" "factconv_formatting" "irrconv_formatting")


OUTPUT_EXP_LABEL="${CKPT}_${beta}"
for SPLIT in "${SPLIT_arr[@]}"; do
    for IDX in $(seq 0 $((CHUNKS-1))); do
        CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m residual_visual_decoding.LLaVA.mmhalsnowball_inf \
            --model-path ${MODEL_PATH} \
            --vision-model-path ${VISION_MODEL_PATH} \
            --question-file evaluation/data/${TASK_TYPE}_test.json \
            --conversation-file evaluation/data/utterance/utterance_${TASK_TYPE}_${SPLIT}.json \
            --image-folder ${DATADIR}/data/images \
            --answers-file ${DATADIR}/generation_results/$OUTPUT_EXP_LABEL/${CHUNKS}_${IDX}_${SPLIT}.json \
            --num-chunks $CHUNKS \
            --chunk-idx $IDX \
            --temperature 1.0 \
            --rvd \
            --adb \
            --rvd-beta ${beta} \
            --top_p 0.95 \
            --conv-mode vicuna_v1
    done

    wait

    output_file=${DATADIR}/generation_results/$OUTPUT_EXP_LABEL/generated_file_utterance_${TASK_TYPE}_${SPLIT}.json

    # Clear out the output file if it exists.
    > "$output_file"

    # Loop through the indices and concatenate each file.
    for IDX in $(seq 0 $((CHUNKS-1))); do 
        cat ${DATADIR}/generation_results/$OUTPUT_EXP_LABEL/${CHUNKS}_${IDX}_${SPLIT}.json >> "$output_file"
    done
done