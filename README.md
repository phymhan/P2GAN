# P2GAN: Dual Projection Generative Adversarial Networks for Conditional Image Generation

[[pdf]](https://openaccess.thecvf.com/content/ICCV2021/papers/Han_Dual_Projection_Generative_Adversarial_Networks_for_Conditional_Image_Generation_ICCV_2021_paper.pdf) [[supp]](https://openaccess.thecvf.com/content/ICCV2021/supplemental/Han_Dual_Projection_Generative_ICCV_2021_supplemental.pdf) [[arXiv]](https://arxiv.org/abs/2108.09016)  [[slides]](https://www.dropbox.com/s/h4kl683snx5ptlq/P2GAN_slides.pdf?dl=0) [[poster]](https://www.dropbox.com/s/9xg8vgsz0jg1xke/P2GAN_poster.pdf?dl=0)

![Discriminator models for conditional GANs.](/assets/discriminator.png)

The code consists of the following forks of BigGAN-PyTorch and PyTorch-StudioGAN:
- [phymhan/BigGAN-PyTorch](https://github.com/phymhan/BigGAN-PyTorch)
- [phymhan/PyTorch-StudioGAN](https://github.com/phymhan/PyTorch-StudioGAN)


# 1-D Mixture-of-Gaussian Experiment (Based on TAC-GAN Code)
Coming soon...

# VGGFace2 Experiments (BigGAN Codebase)
Experiments for VGGFace2, CIFAR100, and ImageNet at 64-by-64 resolution are based on the [BigGAN-PyTorch](https://github.com/ajbrock/BigGAN-PyTorch) codebase.
## Prepare VGGFace2 subsets
Download VGGFace2 dataset from [official website](https://github.com/ox-vgg/vgg_face2). Make subsets that contain 200, 500, 2000 identities. The lists of corresponding identities used in the experiments are provided in [`id_v200.txt`](https://drive.google.com/file/d/1YzG0LAdDRygGcuHrehSrv3jswsdZt-B8/view?usp=sharing), [`id_v500.txt`](https://drive.google.com/file/d/1ZNa9t1GQQtsTOUI4zQnQeZn3HLkBvcyU/view?usp=sharing), and [`id_v2000.txt`](https://drive.google.com/file/d/1n9ELLBNGi09VpiZVOt4sn09Nx7lNxNOA/view?usp=sharing), respectively.

To create a HDF5 dataset, run
```
python make_hdf5.py --dataset V2000 --dataset_hdf5 VGGFace2000_ --num_workers 8
```

## Finetune an Inception V3 model for evaluation on VGGFace2
```
python train_inception.py \
--dataset V2000_hdf5 \
--experiment_name finetune_v2000_adam \
--optimizer adam \
--tensorboard \
--shuffle --batch_size 256 --parallel \
--num_epochs 50 \
--seed 0
```
To load the model used in our experiments, please download the checkpoint as provided [here](https://drive.google.com/file/d/1_vuJqkhRjIrgthyZCBxZiLJ5yuXlBpUB/view?usp=sharing), and save it as `weights/newinc_v2000/model_itr_20000.pth`.

## Prepare Inception Moments
Prepare inception moments for calculating FID:
```
python calculate_inception_moments.py --dataset V200_hdf5 --custom_inception_model_path weights/newinc_v2000/model_itr_20000.pth --inception_moments_path data/v200_inc_itr20000.npz
```
Prepare inception moments for calculating Intra-FID:
```
python calculate_intra_inception_moments.py --dataset V200_hdf5 --custom_inception_model_path weights/newinc_v2000/model_itr_20000.pth --intra_inception_moments_path data/v200_intra_inc_itr20000
```

## Train P2GAN and baseline models
The current implementation requires training with multiple GPUs (for VGGFace2 experiments, we observe a significant performance drop when running on a single GPU). The following commands are tested on two GPUs.

Train a **P2GAN**:
```
CUDA_VISIBLE_DEVICES=0,1 python train.py \
--dataset V200_hdf5 \
--custom_inception_model_path weights/newinc_v2000/model_itr_20000.pth --custom_num_classes 2000 \
--inception_moments_path data/v200_inc_itr20000.npz \
--experiment_name V200_p2 --seed 2018 \
--f_div_loss revkl \
--loss_type hybrid --no_projection --AC --TAC \
--AC_weight 1.0 \
--model BigGAN_hybrid --which_train_fn hybrid \
--tensorboard \
--parallel --shuffle --num_workers 16 --batch_size 256  \
--num_G_accumulations 1 --num_D_accumulations 1 \
--G_ch 32 --D_ch 32 \
--num_D_steps 1 --G_lr 1e-4 --D_lr 4e-4 --D_B2 0.999 --G_B2 0.999 \
--G_attn 32 --D_attn 32 \
--G_nl inplace_relu --D_nl inplace_relu \
--SN_eps 1e-6 --BN_eps 1e-5 --adam_eps 1e-6 \
--G_ortho 0.0 \
--G_shared \
--G_init ortho --D_init ortho \
--hier --dim_z 120 --shared_dim 128 \
--G_eval_mode \
--ema --use_ema --ema_start 20000 --num_epochs 100 \
--test_every 2000 --save_every 2000 --num_best_copies 2 --num_save_copies 2 \
--use_multiepoch_sampler --sv_log_interval -1 \
--which_best FID \
--save_test_iteration --no_intra_fid 
```

Train a **P2GAN-w** (P2GAN-ap):
```
CUDA_VISIBLE_DEVICES=0,1 python train.py \
--dataset V200_hdf5 \
--custom_inception_model_path weights/newinc_v2000/model_itr_20000.pth --custom_num_classes 2000 \
--inception_moments_path data/v200_inc_itr20000.npz \
--experiment_name V200_p2ap --seed 2018 \
--f_div_loss revkl \
--detach_weight_linear \
--add_weight_penalty \
--use_hybrid --adaptive_loss sigmoid --adaptive_loss_detach \
--loss_type hybrid --no_projection --AC --TAC \
--AC_weight 1.0 \
--model BigGAN_hybrid --which_train_fn amortised \
--tensorboard \
--parallel --shuffle --num_workers 16 --batch_size 256  \
--num_G_accumulations 1 --num_D_accumulations 1 \
--G_ch 32 --D_ch 32 \
--num_D_steps 1 --G_lr 1e-4 --D_lr 4e-4 --D_B2 0.999 --G_B2 0.999 \
--G_attn 32 --D_attn 32 \
--G_nl inplace_relu --D_nl inplace_relu \
--SN_eps 1e-6 --BN_eps 1e-5 --adam_eps 1e-6 \
--G_ortho 0.0 \
--G_shared \
--G_init ortho --D_init ortho \
--hier --dim_z 120 --shared_dim 128 \
--G_eval_mode \
--ema --use_ema --ema_start 20000 --num_epochs 100 \
--test_every 2000 --save_every 2000 --num_best_copies 2 --num_save_copies 2 \
--use_multiepoch_sampler --sv_log_interval -1 \
--which_best FID \
--save_test_iteration --no_intra_fid 
```

Train a **f-cGAN**:
```
CUDA_VISIBLE_DEVICES=0,1 python train.py \
--dataset V200_hdf5 \
--custom_inception_model_path weights/newinc_v2000/model_itr_20000.pth --custom_num_classes 2000 \
--inception_moments_path data/v200_inc_itr20000.npz \
--experiment_name V200_fc --seed 2018 \
--f_div_loss revkl \
--loss_type TAC --no_projection --AC --TAC \
--AC_weight 1.0 \
--model BigGAN_hybrid --which_train_fn hybrid \
--tensorboard \
--parallel --shuffle --num_workers 16 --batch_size 256 \
--num_G_accumulations 1 --num_D_accumulations 1 \
--G_ch 32 --D_ch 32 \
--num_D_steps 1 --G_lr 1e-4 --D_lr 4e-4 --D_B2 0.999 --G_B2 0.999 \
--G_attn 32 --D_attn 32 \
--G_nl inplace_relu --D_nl inplace_relu \
--SN_eps 1e-6 --BN_eps 1e-5 --adam_eps 1e-6 \
--G_ortho 0.0 \
--G_shared \
--G_init ortho --D_init ortho \
--hier --dim_z 120 --shared_dim 128 \
--G_eval_mode \
--ema --use_ema --ema_start 20000 --num_epochs 100 \
--test_every 2000 --save_every 2000 --num_best_copies 2 --num_save_copies 2 \
--use_multiepoch_sampler --sv_log_interval -1 \
--which_best FID \
--save_test_iteration --no_intra_fid 
```

Train a **Proj-GAN**:
```
CUDA_VISIBLE_DEVICES=0,1 python train.py \
--dataset V200_hdf5 \
--custom_inception_model_path weights/newinc_v2000/model_itr_20000.pth --custom_num_classes 2000 \
--inception_moments_path data/v200_inc_itr20000.npz \
--experiment_name V200_proj --seed 2018 \
--f_div_loss revkl \
--loss_type Projection \
--AC_weight 1.0 \
--model BigGAN --which_train_fn GAN \
--tensorboard \
--parallel --shuffle --num_workers 16 --batch_size 256 \
--num_G_accumulations 1 --num_D_accumulations 1 \
--G_ch 32 --D_ch 32 \
--num_D_steps 1 --G_lr 1e-4 --D_lr 4e-4 --D_B2 0.999 --G_B2 0.999 \
--G_attn 32 --D_attn 32 \
--G_nl inplace_relu --D_nl inplace_relu \
--SN_eps 1e-6 --BN_eps 1e-5 --adam_eps 1e-6 \
--G_ortho 0.0 \
--G_shared \
--G_init ortho --D_init ortho \
--hier --dim_z 120 --shared_dim 128 \
--G_eval_mode \
--ema --use_ema --ema_start 20000 --num_epochs 100 \
--test_every 2000 --save_every 2000 --num_best_copies 2 --num_save_copies 2 \
--use_multiepoch_sampler --sv_log_interval -1 \
--which_best FID \
--save_test_iteration --no_intra_fid 
```

Train a **TAC-GAN**:
```
CUDA_VISIBLE_DEVICES=0,1 python train.py \
--dataset V200_hdf5 \
--custom_inception_model_path weights/newinc_v2000/model_itr_20000.pth --custom_num_classes 2000 \
--inception_moments_path data/v200_inc_itr20000.npz \
--experiment_name V200_tac --seed 2018 \
--f_div_loss revkl \
--loss_type TAC --no_projection --AC --TAC \
--train_AC_on_fake \
--AC_weight 1.0 \
--model BigGAN_hybrid --which_train_fn hybrid \
--tensorboard \
--parallel --shuffle --num_workers 16 --batch_size 256 \
--num_G_accumulations 1 --num_D_accumulations 1 \
--G_ch 32 --D_ch 32 \
--num_D_steps 1 --G_lr 1e-4 --D_lr 4e-4 --D_B2 0.999 --G_B2 0.999 \
--G_attn 32 --D_attn 32 \
--G_nl inplace_relu --D_nl inplace_relu \
--SN_eps 1e-6 --BN_eps 1e-5 --adam_eps 1e-6 \
--G_ortho 0.0 \
--G_shared \
--G_init ortho --D_init ortho \
--hier --dim_z 120 --shared_dim 128 \
--G_eval_mode \
--ema --use_ema --ema_start 20000 --num_epochs 100 \
--test_every 2000 --save_every 2000 --num_best_copies 2 --num_save_copies 2 \
--use_multiepoch_sampler --sv_log_interval -1 \
--which_best FID \
--save_test_iteration --no_intra_fid 
```

# ImageNet 64x64 Resolution Experiments (BigGAN Codebase)
Train a **P2GAN**:
```
CUDA_VISIBLE_DEVICES=0,1 python train.py \
--dataset I64 \
--experiment_name I64_p2 --seed 0 \
--f_div_loss revkl \
--loss_type hybrid --no_projection --AC --TAC \
--AC_weight 1.0 \
--model BigGAN_hybrid --which_train_fn hybrid \
--tensorboard \
--parallel --shuffle --num_workers 16 --batch_size 256 \
--num_G_accumulations 8 --num_D_accumulations 8 \
--G_ch 32 --D_ch 32 \
--num_D_steps 1 --G_lr 1e-4 --D_lr 4e-4 --D_B2 0.999 --G_B2 0.999 \
--G_attn 32 --D_attn 32 \
--G_nl inplace_relu --D_nl inplace_relu \
--SN_eps 1e-6 --BN_eps 1e-5 --adam_eps 1e-6 \
--G_ortho 0.0 \
--G_shared \
--G_init ortho --D_init ortho \
--hier --dim_z 120 --shared_dim 128 \
--G_eval_mode \
--ema --use_ema --ema_start 20000 --num_epochs 200 \
--test_every 2000 --save_every 2000 --num_best_copies 2 --num_save_copies 2 \
--use_multiepoch_sampler --sv_log_interval -1 \
--no_intra_fid \
--which_best FID --save_test_iteration 
```

Train a **P2GAN-w** (P2GAN-ap):
```
CUDA_VISIBLE_DEVICES=0,1 python train.py \
--dataset I64 \
--experiment_name I64_p2ap --seed 0 \
--add_weight_penalty \
--detach_weight_linear \
--use_hybrid --adaptive_loss sigmoid --adaptive_loss_detach \
--loss_type hybrid --no_projection --AC --TAC \
--AC_weight 1.0 \
--model BigGAN_hybrid --which_train_fn amortised \
--tensorboard \
--parallel --shuffle --num_workers 16 --batch_size 256 \
--num_G_accumulations 8 --num_D_accumulations 8 \
--G_ch 32 --D_ch 32 \
--num_D_steps 1 --G_lr 1e-4 --D_lr 4e-4 --D_B2 0.999 --G_B2 0.999 \
--G_attn 32 --D_attn 32 \
--G_nl inplace_relu --D_nl inplace_relu \
--SN_eps 1e-6 --BN_eps 1e-5 --adam_eps 1e-6 \
--G_ortho 0.0 \
--G_shared \
--G_init ortho --D_init ortho \
--hier --dim_z 120 --shared_dim 128 \
--G_eval_mode \
--ema --use_ema --ema_start 20000 --num_epochs 200 \
--test_every 2000 --save_every 2000 --num_best_copies 2 --num_save_copies 2 \
--use_multiepoch_sampler --sv_log_interval -1 \
--no_intra_fid \
--which_best FID --save_test_iteration 
```

# ImageNet 128x128 Resolution Experiments (StudioGAN Codebase)
Experiments for ImageNet at 128-by-128 resolution and CIFAR10 are based on the [StudioGAN](https://github.com/POSTECH-CVLab/PyTorch-StudioGAN) codebase. The code is tested on 4 A100 GPUs.

To train a **P2GAN** model:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/main.py -t -e -sync_bn -c src/configs/P2GAN/I128_p2.json --eval_type "valid"
```

To train a **P2GAN-ap** model:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/main.py -t -e -sync_bn -c src/configs/P2GAN/I128_p2ap.json --eval_type "valid"
```

To train a **P2GAN-ap-alt** model:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/main.py -t -e -sync_bn -c src/configs/P2GAN/I128_p2ap_exp.json --eval_type "valid"
```


# Pretrained Weights and Training Logs

<!-- **CIFAR10**:
|     | Weight | Log | FID |
| --- | :---: | :---: | :---: |
| Proj-GAN    | - | - | \*- |
| P2GAN       | [gema]() | [log]() | - |
| P2GAN-w     | [gema]() | [log]() | - | -->


**ImageNet 128**:
|     | Weight | Log | FID |
| --- | :---: | :---: | :---: |
| Proj-GAN     | [gema](https://drive.google.com/file/d/1kAo-hW_MogPxgunuCcFgaGoQcuaVDoid/view?usp=sharing) | [log](https://drive.google.com/file/d/1OdulkVBfAuEuCuZG-sEA7WQ7fPpVq-MJ/view?usp=sharing) | 23.07 |
| P2GAN        | [gema](https://drive.google.com/file/d/1GDbeT-2mjj_uYHcgxEEN4sb0VVxkeM-L/view?usp=sharing) | [log](https://drive.google.com/file/d/1tFDP_q05QTxAX8T2E8vUHMidgpbDkgv5/view?usp=sharing) | 16.98 |
| P2GAN-ap     | [gema](https://drive.google.com/file/d/1v_p8VBUpuN0r8l5WN96l5Ua1BVnaeco3/view?usp=sharing) | [log](https://drive.google.com/file/d/1WmUzYjkcfDfFbt9hpo6-0CNvQZpN3XZJ/view?usp=sharing) | 19.20 |
| P2GAN-ap-alt | [gema](https://drive.google.com/file/d/1yJ0EDMnvn9KxBXIybrbU3Pf11G6wpaWz/view?usp=sharing) | [log](https://drive.google.com/file/d/1P1bVcEFijOhFQ6z3xWkwlLqQN-EGFdN4/view?usp=sharing) | 16.53 |


**VGGFace2-200**:
|     | Weight | Log | FID |
| --- | :---: | :---: | :---: |
| Proj-GAN | [gema](https://drive.google.com/file/d/1mg8RLMJUThJJMbfAdo2okR4S_w2PrwyQ/view?usp=sharing) | [log](https://drive.google.com/file/d/1nBv6azKs1_bELjpccDSfyI4K9WJi8UWD/view?usp=sharing) | 61.43 |
| TAC-GAN  | [gema](https://drive.google.com/file/d/1sbZh-0vg3E0xB05ds234mJyxCzLnq0XG/view?usp=sharing) | [log](https://drive.google.com/file/d/1dh0VMCKLUb3PrIbwtv5s0VvGKOdhkhwj/view?usp=sharing) | 96.06 |
| f-cGAN | [gema](https://drive.google.com/file/d/1F8wRrFWCLZjc3F7R3xglFLSjc3dLoCgf/view?usp=sharing) | [log](https://drive.google.com/file/d/1u4r-AiT6gDShElU4IxFCXHm0pxeQTYtD/view?usp=sharing) | 29.54 |
| P2GAN    | [gema](https://drive.google.com/file/d/1nJg4Mxk0pArjrO7u-4fzbdmYJfM3thLw/view?usp=sharing) | [log](https://drive.google.com/file/d/1vhfWo2U7sCiQvWF3k7Kf6F62wRI1qhok/view?usp=sharing) | 20.70 |
| P2GAN-w  | [gema](https://drive.google.com/file/d/1xdgh_GmjUxqpVxJFJKjTwwnmmkHTDz29/view?usp=sharing) | [log](https://drive.google.com/file/d/1RqLI382ZOQ487n5h4LWf40CarIBzRqU8/view?usp=sharing) | 15.70 |


**VGGFace2-500**:
|     | Weight | Log | FID |
| --- | :---: | :---: | :---: |
| Proj-GAN | [gema](https://drive.google.com/file/d/1ntqKMT_PxTIoPd9txKAxJG4q_R4WJ-NQ/view?usp=sharing) | [log](https://drive.google.com/file/d/1p_S6hXHBwhiIY99T1P114IlbHtasqlzc/view?usp=sharing) | 23.57 |
| TAC-GAN  | [gema](https://drive.google.com/file/d/1xluFV2EdkQNrxxPwotfpyYCPtB9TMk3f/view?usp=sharing) | [log](https://drive.google.com/file/d/1Z14YmEfHz4-KczgDjfiOY5QxxD__I0bI/view?usp=sharing) | 19.30 |
| f-cGAN | [gema](https://drive.google.com/file/d/14ZHqtdYknO2SoQqprZnvbBsLAjLTx9lS/view?usp=sharing) | [log](https://drive.google.com/file/d/1igaJRuVKJd0cfDykp_5nnU174cOgIq90/view?usp=sharing) | 16.74 |
| P2GAN    | [gema](https://drive.google.com/file/d/1h1yqghj1VSA3pXG5c8PSoqZ1l4SexROT/view?usp=sharing) | [log](https://drive.google.com/file/d/1HRFVOkCIkW8XuLVS_F3reR1VaHBDKtAm/view?usp=sharing) | 12.09 |
| P2GAN-w  | [gema](https://drive.google.com/file/d/1xm_HMDHCFjeBPfueL-8oYz5EjqW0p-Fc/view?usp=sharing) | [log](https://drive.google.com/file/d/1HSnf88xoLgTAKnRHEY-C4nSuDq0dJ_sJ/view?usp=sharing) | 12.73 |


# Citation
P2GAN implementation is heavily based on [BigGAN](https://github.com/ajbrock/BigGAN-PyTorch), [StudioGAN](https://github.com/POSTECH-CVLab/PyTorch-StudioGAN), and [TAC-GAN](https://github.com/batmanlab/twin-auxiliary-classifiers-gan). If you use this code, please cite

```
@InProceedings{Han_2021_ICCV,
    author    = {Han, Ligong and Min, Martin Renqiang and Stathopoulos, Anastasis and Tian, Yu and Gao, Ruijiang and Kadav, Asim and Metaxas, Dimitris N.},
    title     = {Dual Projection Generative Adversarial Networks for Conditional Image Generation},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {14438-14447}
}
```
