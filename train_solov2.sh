CUDA_VISIBLE_DEVICES=6,7  \
OMP_NUM_THREADS=2   \
/home/gyuri/anaconda3/envs/rhs_u/bin/python tools/train_net.py   \
    --config-file configs/aihub/Base-SOLOv2.yaml   \
    --num-gpus 2    \
    --resume    \
    OUTPUT_DIR /ailab_mat/personal/rho_heeseon/InstanceSegmentation/ours_2x/solov2
    
