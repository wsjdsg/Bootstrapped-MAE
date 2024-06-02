#model vit_base_patch16 -> deit_tiny_small_decoder
#we need to specfy the following parameters
#batch_size, epochs(200) ,model(set to DeiT) 
#input_size adapt to 32(cifar-10),data_path
#mask_ratio (TODO)
#blr 1.5e-4 (TODO)
#batch_size (TODO)
CUDA_VISIBLE_DEVICES=3 python 'main_pretrain.py' \
    --batch_size 256 \
    --model deit_tiny \
    --norm_pix_loss \
    --input_size 32 \
    --mask_ratio 0.5 \
    --epochs 200 \
    --blr 5e-4 --weight_decay 0.05 \
    --data_path './data' \
    --output_dir "./output_dir/mae" \
    --log_dir "./output_dir/mae"
