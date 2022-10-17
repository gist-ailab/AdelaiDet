CUDA_VISIBLE_DEVICES=0  \
OMP_NUM_THREADS=1   \
/home/gyuri/anaconda3/envs/rhs_u/bin/python tools/train_net.py   \
    --config-file configs/CondInst/Base-CondInst.yaml   \
    --num-gpus 1   \
    OUTPUT_DIR /ailab_mat/personal/rho_heeseon/InstanceSegmentation/tmp
