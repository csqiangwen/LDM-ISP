CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 python3 main.py \
--train \
--base configs/LLIE/v2-finetune_text_T_512.yaml \
--gpus 0, \
--name LLIE_UNet \
--scale_lr False