#TODO blr
rt=0.9
CUDA_VISIBLE_DEVICES=1 python main_finetune.py \
    --batch_size 256 \
    --epochs 100 \
    --model deit_tiny \
    --input_size 32 \
    --blr 5e-4 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --dist_eval \
    --data_path './data' \
    --output_dir "./output_dir/mae_mk$rt/finetune" \
    --log_dir "./output_dir/mae_mk$rt/finetune" \
    --finetune "./output_dir/mae_mk$rt/checkpoint-199.pth"