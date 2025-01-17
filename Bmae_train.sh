# note: use "--enable_ema   --ema_alpha    --ema_warmup_epochs" to enable MAE_ema
# once ema is enabled , bmae_K is useless
python 'main_pretrain.py' \
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
    --bmae_k 8 \
    # --enable_ema \
    # --ema_alpha=0.9999 \
    # --ema_warmup_epochs=40

