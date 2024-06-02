CUDA_VISIBLE_DEVICES=2 python main_linprobe.py \
    --batch_size 256 \
    --epochs 100 \
    --model deit_tiny \
    --cls_token \
    --blr 0.1 \
    --weight_decay 0.0 \
    --dist_eval \
    --data_path './data' \
    --output_dir './output_dir/bmae/linear' \
    --log_dir './output_dir/bmae/linear' \
    --finetune './output_dir/bmae/checkpoint-199.pth'
