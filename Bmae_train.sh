#note comment out "--enable_ema   --ema_alpha    --ema_warmup_epochs" to disable ema 
CUDA_VISIBLE_DEVICES=3 python 'main_pretrain.py' \
    --batch_size 256 \
    --model deit_tiny \
    --norm_pix_loss \
    --input_size 32 \
    --mask_ratio 0.6 \
    --epochs 200 \
    --blr 5e-4 --weight_decay 0.05 \
    --data_path './data' \
    --output_dir "./output_dir/bmae" \
    --log_dir "./output_dir/bmae" \
    --enable_bootstrap \
    --bmae_k 4 \
    --enable_ema \
    --ema_alpha=0.99 \
    --ema_warmup_epochs=1
