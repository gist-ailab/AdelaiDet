CUDA_VISIBLE_DEVICES=2,3,4,6  \
OMP_NUM_THREADS=1   \
python tools/train_net.py   \
    --config-file configs/BlendMask/R_101_3x.yaml   \
    --num-gpus 4    \
    OUTPUT_DIR /home/gyuri/NAS_AIlab_dataset/personal/rho_heeseon/InstanceSegmentation/mrcnn_base_gyuri