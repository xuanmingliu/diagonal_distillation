# torchrun --nnodes=8 --nproc_per_node=8 --rdzv_id=5235 \
#   --rdzv_backend=c10d \
#   --rdzv_endpoint $MASTER_ADDR \
#   train.py \
#   --config_path configs/self_forcing_dmd.yaml \
#   --logdir logs/self_forcing_dmd \
#   --disable-wandb

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

torchrun --nproc_per_node=8 --master_port 29501 \
    train.py \
    --config_path configs/self_forcing_dmd.yaml \
    --logdir logs/self_forcing_dmd \
    --disable-wandb
    --block_type "奇数"
    --block_type "偶数"



# nohup bash training.sh > training_$(date +%Y%m%d_%H%M%S).log 2>&1 &
# pkill -f "train.py --config_path configs/self_forcing_dmd.yaml" || kill -9 $(pgrep -f "train.py --config_path configs/self_forcing_dmd.yaml")

# hdfs dfs -put /mnt/bn/jinxiuliu-hl/Long_Video_Gen/causal_distill/Self-Forcing/logs/self_forcing_dmd/20250623_000310/checkpoint_model_001800/model.pt hdfs://haruna/dp/mloops/datasets/autoaigc/.stage/liujinxiu/Self-Forcing/checkpoints/20250623_000310_checkpoint_model_001800_model.pt
