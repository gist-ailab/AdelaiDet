CUDA_VISIBLE_DEVICES=3,4  \
OMP_NUM_THREADS=1   \
python tools/train_net.py   \
    --config-file configs/aihub/condinst.yaml   \
    --num-gpus 2   \
    OUTPUT_DIR /ailab_mat/personal/rho_heeseon/InstanceSegmentation/ours_2x/condinst/
