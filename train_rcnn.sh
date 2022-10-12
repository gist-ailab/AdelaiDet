CUDA_VISIBLE_DEVICES=0,1,2,3  \
OMP_NUM_THREADS=1   \
python tools/train_net.py   \
    --config-file configs/RCNN/R_101_3x.yaml   \
    --num-gpus 4    \
    OUTPUT_DIR /home/gyuri/NAS_AIlab_dataset/personal/rho_heeseon/InstanceSegmentation/rcnn_base