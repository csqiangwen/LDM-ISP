CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 python3 scripts/test_custom.py \
--config configs/LLIE/v2-finetune_text_T_512.yaml \
--ckpt ./pretrained_models/LDM_ISP_UNet_epoch=000100.ckpt \
--init_img test_raw_images \
--outdir results_raw_images \
--ddpm_steps 200 \
--n_samples 1 \
--amplify_ratio 250