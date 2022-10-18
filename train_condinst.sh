CUDA_VISIBLE_DEVICES=0  \
OMP_NUM_THREADS=1   \
python tools/train_net.py   \
    --config-file configs/ycbv_real/R50_rgb.yaml   \
    --num-gpus 1   \
    OUTPUT_DIR /ailab_mat/personal/rho_heeseon/InstanceSegmentation/tmp
