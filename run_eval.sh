# 2帧之间插7帧
#!/bin/bash
# run_parallel_eval.sh

METHODS=("linear" "quadratic" "si_dis_flow" "bi_dis_flow")
OUTPUT_DIR="./Result/InterpolationMetrics7"
# 定义你想使用的 GPU 列表，比如 (0 1) 代表使用第0块和第1块显卡
GPU_IDS=(7) 
mkdir -p $OUTPUT_DIR

echo "开始并行评估，共 ${#METHODS[@]} 个任务..."

for i in "${!METHODS[@]}"
do
    METHOD=${METHODS[$i]}
    # 自动循环分配 GPU: 任务0->GPU0, 任务1->GPU1, 任务2->GPU0...
    GPU_ID=${GPU_IDS[$((i % ${#GPU_IDS[@]}))]}
    
    echo "正在启动: $METHOD (使用 GPU: $GPU_ID)"
    
    # 使用 CUDA_VISIBLE_DEVICES 限制该进程只能看到指定的 GPU
    CUDA_VISIBLE_DEVICES=$GPU_ID python -u EvaluateMetric.py \
        --method "$METHOD" \
        --output "$OUTPUT_DIR" \
        --device "cuda" > "${OUTPUT_DIR}/${METHOD}_run.log" 2>&1 &
done

wait
echo "全部评估任务完成！"