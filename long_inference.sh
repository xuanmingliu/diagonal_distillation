# export CUDA_VISIBLE_DEVICES=6,7

# # 定义ckpt文件夹路径
# CKPT_DIR="/home/liujinxiu/new_self-forcing/ckpt"
# BASE_OUTPUT_DIR="/home/liujinxiu/new_self-forcing/test_1"

# # 遍历ckpt文件夹中的所有.pt文件
# for checkpoint_file in "$CKPT_DIR"/*.pt; do
#     if [ -f "$checkpoint_file" ]; then
#         # 提取文件名（不包含路径和扩展名）
#         checkpoint_name=$(basename "$checkpoint_file" .pt)
        
#         # 为每个权重文件创建独立的输出文件夹
#         output_folder="$BASE_OUTPUT_DIR/$checkpoint_name"
        
#         echo "Processing checkpoint: $checkpoint_name"
#         echo "Output folder: $output_folder"
        
#         # 运行推理
#         python long_inference.py \
#             --config_path /home/liujinxiu/new_self-forcing/configs/self_forcing_dmd.yaml \
#             --output_folder "$output_folder" \
#             --checkpoint_path "$checkpoint_file" \
#             --data_path /home/liujinxiu/new_self-forcing/single_prompt.txt \
#             --full_budget False \
#             --use_ema
        
#         echo "Finished processing: $checkpoint_name"
#         echo "----------------------------------------"
#     fi
# done

# echo "All checkpoints processed!" 


        # /home/liujinxiu/new_self-forcing/20250717_1049_0500.pt \
        # /home/liujinxiu/new_self-forcing/20250717_0343_1500.pt
        # --checkpoint_path checkpoints/self_forcing_dmd.pt \
        # --checkpoint_path /mnt/bn/jinxiuliu-hl/Long_Video_Gen/causal_distill/Self-Forcing/checkpoints/ode_init.pt
        # --checkpoint_path /mnt/bn/jinxiuliu-hl/Long_Video_Gen/causal_distill/Self-Forcing/logs/self_forcing_dmd/checkpoint_model_000600/model.pt \

        # # 确保HDFS目录存在
        # 上传文件并重命名
        # hdfs dfs -put /mnt/bn/jinxiuliu-hl/Long_Video_Gen/causal_distill/Self-Forcing/logs/self_forcing_dmd/20250623_000310/checkpoint_model_001850/model.pt \
        #       hdfs://haruna/dp/mloops/datasets/autoaigc/.stage/liujinxiu/Self-Forcing/checkpoints/20250623_000310_checkpoint_model_001850_model.pt
        # 
    # --checkpoint_path /mnt/bn/jinxiuliu-hl/Long_Video_Gen/causal_distill/Self-Forcing/logs/self_forcing_dmd/20250623_000310/checkpoint_model_002250/model.pt \
    # /mnt/bn/jinxiuliu-hl/Long_Video_Gen/causal_distill/Self-Forcing/logs/self_forcing_dmd/20250623_000310/checkpoint_model_002250/model.pt

export CUDA_VISIBLE_DEVICES=5
python long_inference.py \
    --config_path /home/liujinxiu/new_self-forcing/configs/self_forcing_dmd.yaml \
    --output_folder /home/liujinxiu/new_self-forcing/test \
    --checkpoint_path /home/liujinxiu/new_self-forcing/20250928_1733_2000.pt \
    --data_path prompts/vidprom_filtered_extended.txt \
    --full_budget False \
    --use_ema

#我们的方法看，长视频生成