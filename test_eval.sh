CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 python3 scripts/eval.py \
--config configs/LLIE/v2-finetune_text_T_512.yaml \
--ckpt ./pretrained_models/LDM_ISP_UNet_epoch=000100.ckpt \
--init_img None \
--outdir test_SID_Sony_129 \
--ddpm_steps 200 \
--n_samples 1