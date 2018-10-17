DATASET=mnist
NENC=100
NZ=100
NQUANT=2
RDCOEFF=1
MODE=p2p
NAME=dcgan_comp_${DATASET}_nenc_${NENC}_nz_${NZ}_nquant_${NQUANT}_rdcoeff_${RDCOEFF}_mode_${MODE}
python dcgan_comp.py --dataset $DATASET --dataroot ./mnist --cuda --ncenc $NENC --nz $NZ --nquant $NQUANT --rdcoeff $RDCOEFF --mode $MODE --outf $NAME



## Mnist experiment from paper (slightly different architecture, 64x64, slightly different schedule)
python wae_comp.py --outf ./mnist_out --dataset mnist --dataroot ./data/mnist --cuda --decay_steps 30 50 --decay_gamma 0.4 --check_every 25

python wae_comp.py --outf ./celeba_out_trick_crop --dataset celeba --dataroot ./data/celeba_waecrop/ --cuda --nz 64 --sigmasqz 2.0 --lbd 1.0 --avbtrick --lr_d 0.001 --lr_eg 0.0003 --niter 55 --decay_steps 30 50 --decay_gamma 0.4 --check_every 25

# Compress with pretrained generator
python wae_comp.py --outf ./celeba_out_trick_crop_comp_nz2 --comp --dataset celeba --dataroot ./data/celeba_waecrop/ --netG celeba_out_trick_crop/netG_epoch_54.pth --ncenc 2 --cuda --nz 64 --sigmasqz 2.0 --lbd 1.0 --avbtrick --lr_d 0.001 --lr_eg 0.0003 --niter 55 --decay_steps 30 50 --decay_gamma 0.4 --check_every 25 --workers 4 --nresenc 2 --freezedec


# Vanilla compressive autoencoder
python wae_comp.py --outf ./celeba_test2 --dataset celeba --dataroot ./data/celeba_waecrop/ --cuda --nz 64 --lbd 0.0 --avbtrick --lr_d 0.001 --lr_eg 0.0003 --niter 55 --decay_steps 30 50 --decay_gamma 0.4 --check_every 25 --nresenc 2 --nresdec 2 --detenc


# cityscapes
python wae_comp.py --outf ./cityscapes_lbd0.25_nz512 --dataset cityscapes --dataroot ./data/cityscapes/ --cuda --nz 512 --sigmasqz 2.0 --lbd 0.25 --avbtrick --lr_d 0.001 --lr_eg 0.001 --niter 150 --decay_steps 75 125 --decay_gamma 0.25 --check_every 25 --imageSize 128 --nresdec 4 --workers 4

# cityscapes plain autoencoder
python wae_comp.py --outf ./cityscapes_autoenc_nz2048_nores --dataset cityscapes --dataroot ./data/cityscapes/ --cuda --nz 2048 --lbd 0.0 --avbtrick --lr_d 0.001 --lr_eg 0.0003 --niter 55 --decay_steps 30 50 --decay_gamma 0.4 --check_every 25 --detenc --workers 4


### VAE CELEBA 19.04.2018
### commit c561d99e631a3d6f0d9653f56b7bb2d7e0eea0c1
python wae_comp.py --dataset celeba --dataroot ./data/celeba_waecrop/ --cuda --nz 64 --sigmasqz 2.0 --avbtrick --lr_d 0.001 --lr_eg 0.0003 --niter 55 --decay_steps 30 50 --decay_gamma 0.4 --check_every 25 --workers 4 \
# testing set
python wae_comp.py --dataset celeba --dataroot ./data/celeba_waecrop/resized_celebA_train/ --testroot ./data/celeba_waecrop/resized_celebA_test/ --cuda --nz 64 --sigmasqz 2.0 --avbtrick --lr_d 0.001 --lr_eg 0.0003 --niter 55 --decay_steps 30 50 --decay_gamma 0.4 --check_every 25 --workers 4 \


## plain autoencoder
--outf celeba_plain_autoencoder --lbd 0.0 --detenc

## plain autoencoder, quantized, ncenc=4
--outf celeba_plain_autoencoder_quant_ncenc4 --lbd 0.0 --detenc --comp --ncenc 4 --nresenc 2

## plain autoencoder, quantized, ncenc=8
--outf celeba_plain_autoencoder_quant_ncenc8 --lbd 0.0 --detenc --comp --ncenc 8 --nresenc 2

## plain WAE
--outf celeba_plain_wae --lbd 1.0

## compress into generator, nenc=4
--outf celeba_waedec_quant_ncenc4 --lbd 1.0 --comp --ncenc 4 --nresenc 2 --netG ./celeba_plain_wae/netG_epoch_54.pth --freezedec

## compress into generator, nenc=8
--outf celeba_waedec_quant_ncenc8 --lbd 1.0 --comp --ncenc 8 --nresenc 2 --netG ./celeba_plain_wae/netG_epoch_54.pth --freezedec

## compress into generator, nenc=4, no gan penalty, deterministic encoder
--outf celeba_waedec_quant_ncenc4_lambda0_detenc --lbd 0.0 --comp --ncenc 4 --nresenc 2 --netG ./celeba_plain_wae/netG_epoch_54.pth --freezedec --detenc

## compress into generator, nenc=4, lambda=2
--outf celeba_waedec_quant_ncenc4_lambda2 --lbd 2.0 --comp --ncenc 4 --nresenc 2 --netG ./celeba_plain_wae/netG_epoch_54.pth --freezedec

## plain WAE deterministic encoder
--outf celeba_plain_wae_detenc --lbd 1.0 --detenc

## plain autoencoder, quantized, ncenc=2
--outf celeba_plain_autoencoder_quant_ncenc2 --lbd 0.0 --detenc --comp --ncenc 2 --nresenc 2

## compress into generator, nenc=2, deterministic encoder for WAE
--outf results/celeba_waedec_quant_ncenc2_detwae --lbd 1.0 --comp --ncenc 2 --nresenc 2 --netG ./results/celeba_plain_wae_detenc/netG_epoch_54.pth --freezedec

## compress into generator, nenc=4, no gan penalty, deterministic encoder
--outf celeba_waedec_quant_ncenc4_lambda0_detenc_detwae --lbd 0.0 --comp --ncenc 2 --nresenc 2 --netG ./results/celeba_plain_wae_detenc/netG_epoch_54.pth --freezedec --detenc



## compress into WGAN
python wae_comp.py --dataset lsun --dataroot ./data/lsun --cuda --nz 100 --sigmasqz 2.0 --avbtrick --lr_d 0.001 --lr_eg 0.0003 --niter 55 --decay_steps 30 50 --decay_gamma 0.4 --check_every 25 --workers 4 --outf lsun_wgandec_test --lbd 0.0 --comp --ncenc 4 --nresenc 2 --netG ../WassersteinGAN/wgan_lsun_MTG/netG_epoch_0.pth --freezedec --detenc --batchSize 64

## Train wgan
python main.py --dataset celeba --dataroot ../wae-pytorch/data/celeba_waecrop/resized_celebA_train/ --workers 4 --cuda --experiment wgan_celeba_MTG --niter 100

python main.py --dataset lsun --dataroot ../wae-pytorch/data/lsun/ --workers 8 --cuda --experiment wgan_lsun_MTG2 --niter 100 --nz 256

source activate pytroch-master-env
