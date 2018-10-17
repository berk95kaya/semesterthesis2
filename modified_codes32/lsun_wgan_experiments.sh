BASE_COMMAND=python\ wae_comp.py\ --dataset\ lsun\ --dataroot\ ./data/lsun/\ --cuda\ --nz\ 100\ --sigmasqz\ 1.0\ --avbtrick\ --lr_d\ 0.001\ --lr_eg\ 0.0003\ --niter\ 15\ --decay_steps\ 10\ 13\ --decay_gamma\ 0.4\ --check_every\ 5\ --workers\ 16\ --ngpu\ 2
#
## compress into generator, nenc=4
$BASE_COMMAND --outf ./results/lsun_wgandec_quant_ncenc4 --lbd 1.0 --comp --ncenc 4 --nresenc 2 --netG ../WassersteinGAN/wgan_lsun_MTG/netG_epoch_23.pth --freezedec
#
## compress into generator, nenc=4, no gan penalty, deterministic encoder
$BASE_COMMAND --outf ./results/lsun_wgandec_quant_ncenc4_lambda0_detenc --lbd 0.0 --comp --ncenc 4 --nresenc 2 --netG ../WassersteinGAN/wgan_lsun_MTG/netG_epoch_23.pth --freezedec --detenc
#
## compress into generator, nenc=4, 6 resunits
$BASE_COMMAND --outf ./results/lsun_wgandec_quant_ncenc4_6res --lbd 1.0 --comp --ncenc 4 --nresenc 6 --netG ../WassersteinGAN/wgan_lsun_MTG/netG_epoch_23.pth --freezedec
#
## compress into generator, nenc=8
$BASE_COMMAND --outf ./results/lsun_wgandec_quant_ncenc8 --lbd 1.0 --comp --ncenc 8 --nresenc 2 --netG ../WassersteinGAN/wgan_lsun_MTG/netG_epoch_23.pth --freezedec

$BASE_COMMAND --outf ./results/lsun_wgandec_quant_ncenc8_6res --lbd 1.0 --comp --ncenc 8 --nresenc 6 --netG ../WassersteinGAN/wgan_lsun_MTG/netG_epoch_23.pth --freezedec

$BASE_COMMAND --outf ./results/lsun_wgandec_lambda0_detenc --lbd 0.0 --nresenc 2 --netG ../WassersteinGAN/wgan_lsun_MTG/netG_epoch_23.pth --freezedec --detenc

$BASE_COMMAND --outf ./results/lsun_wgandec_quant_ncenc8_2res_lbd025 --lbd 0.25 --comp --ncenc 8 --nresenc 2 --netG ../WassersteinGAN/wgan_lsun_MTG/netG_epoch_23.pth --freezedec



### plain autoencoder on lsun
## 256 channels
BASE_COMMAND=python\ wae_comp.py\ --dataset\ lsun\ --dataroot\ ./data/lsun/\ --cuda\ --nz\ 256\ --sigmasqz\ 1.0\ --avbtrick\ --lr_d\ 0.001\ --lr_eg\ 0.0003\ --niter\ 15\ --decay_steps\ 10\ 13\ --decay_gamma\ 0.4\ --check_every\ 5\ --workers\ 4\ --ngpu\ 1
##
$BASE_COMMAND --outf ./results/lsun_plain_autoencoder_nz256 --lbd 0.0 --detenc
##
$BASE_COMMAND --outf ./results/lsun_wgandec_quant_ncenc16_nz256_res4 --lbd 1.0 --comp --ncenc 16 --nresenc 4 --netG ../WassersteinGAN/wgan_lsun_MTG2/netG_epoch_24.pth --freezedec
##
$BASE_COMMAND --outf ./results/lsun_wgandec_nz256_res4_lbd0 --lbd 0.0 --nresenc 4 --netG ../WassersteinGAN/wgan_lsun_MTG2/netG_epoch_24.pth --freezedec --detenc
##
$BASE_COMMAND --outf ./results/lsun_wgandec_quant_ncenc16_nz256_res4_ndl10 --lbd 1.0 --comp --ncenc 16 --nresenc 4 --netG ../WassersteinGAN/wgan_lsun_MTG2/netG_epoch_24.pth --freezedec --ndl 10
##
$BASE_COMMAND --outf ./results/lsun_wgandec_quant_ncenc16_nz256_res4_detenc --lbd 1.0 --comp --ncenc 16 --nresenc 4 --netG ../WassersteinGAN/wgan_lsun_MTG2/netG_epoch_24.pth --freezedec --detenc
