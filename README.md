# LDM-ISP: Enhancing Neural ISP for Low Light with Latent Diffusion Models
[Paper](https://arxiv.org/abs/2312.01027) | [Project Page](https://csqiangwen.github.io/projects/ldm-isp/)

This is the official PyTorch implementation of ''LDM-ISP: Enhancing Neural ISP for Low Light with Latent Diffusion Models''.

## ToDo
- [ ] Release training code
- [x] Release testing code
- [x] Release pretrained models
- [x] Release test results 

## Preparation
```bash
conda env create -f environment.yml -n ldm_isp
```

## Evaluation
We released our test results with their corresponding GTs. You may directly compare them with your results during your experiments.
- Test results: [SID-Sony](https://hkustconnect-my.sharepoint.com/:f:/g/personal/qwenab_connect_ust_hk/Er1By5So2HhAp86LtjYV8ooBvrB2TAx9BGoMGlWReTxFxg?e=1uuEej), [ELD-Sony](https://hkustconnect-my.sharepoint.com/:f:/g/personal/qwenab_connect_ust_hk/EnIaXWkZEuxKkmdg5dzRNL0BqU1tPZSKPpfYJMkxgx_u8w?e=I6RBRl), [LRD](https://hkustconnect-my.sharepoint.com/:f:/g/personal/qwenab_connect_ust_hk/Ep-Ibxe_UOtCkD47YhDSVn4BMyA8I_WdkPGTOLJuWcIFfw?e=rarEkQ).

## Testing (your own data)
- Download the [pretrained models]([https://drive.google.com/drive/folders/1c3JYdv64U-OmOyksNK6n51sNwBgy-iQC?usp=sharing](https://hkustconnect-my.sharepoint.com/:f:/g/personal/qwenab_connect_ust_hk/EvlceEym2fBAj1EmvKr6DXQB2thH4rd3OekF3HoGLwcuEw?e=kznnON)), and put it in ```pretrained_models/```;
- (The released pretrained models are re-implementations, so the evaluation scores are slightly better than those reported in the published paper.)
- Put your own RAW files (Bayer Pattern) into ''test_raw_images'' and the sRGB results will be in ''results_raw_images''.
- To test:
```
$ bash test_custom.sh
```

## Acknowledgements
- This code is based on previous excellent work [StableSR](https://github.com/IceClear/StableSR).

## Citation
If you find this repository useful for your research, please cite the following work.
```
@article{wen2023ldm,
  title={LDM-ISP: Enhancing Neural ISP for Low Light with Latent Diffusion Models},
  author={Wen, Qiang and Xing, Yazhou and Rao, Zhefan and Chen, Qifeng},
  journal={arXiv preprint arXiv:2312.01027},
  year={2023}
}

```
<p align='center'>
<img src='Logo/HKUST_VIL.png' width=500>
</p>
