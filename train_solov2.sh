CUDA_VISIBLE_DEVICES=5,7  \
OMP_NUM_THREADS=2   \
/home/gyuri/anaconda3/envs/rhs_u/bin/python tools/train_net.py   \
    --config-file configs/SOLOv2/Base-SOLOv2.yaml   \
    --num-gpus 2    \
    --resume    \
    OUTPUT_DIR /ailab_mat/personal/rho_heeseon/InstanceSegmentation/tmp
    
