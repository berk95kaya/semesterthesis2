BASE_COMMAND=python\ wae_comp.py\ --dataset\ celeba\ --dataroot\ ./data/celeba_waecrop/resized_celebA_train/\ --testroot\ ./data/celeba_waecrop/resized_celebA_test/\ --cuda\ --nz\ 128\ --sigmasqz\ 1.0\ --avbtrick\ --lr_d\ 0.001\ --lr_eg\ 0.0003\ --niter\ 55\ --decay_steps\ 30\ 50\ --decay_gamma\ 0.4\ --check_every\ 5\ --workers\ 4
#
## compress into generator, nenc=4
$BASE_COMMAND --outf ./results/celeba_wgandec_quant_ncenc4 --lbd 1.0 --comp --ncenc 4 --nresenc 2 --netG ../WassersteinGAN/wgan_celeba_MTG/netG_epoch_99.pth --freezedec
#
## compress into generator, nenc=4, no gan penalty, deterministic encoder
$BASE_COMMAND --outf ./results/celeba_wgandec_quant_ncenc4_lambda0_detenc --lbd 0.0 --comp --ncenc 4 --nresenc 2 --netG ../WassersteinGAN/wgan_celeba_MTG/netG_epoch_99.pth --freezedec --detenc
#
## compress into generator, nenc=8
$BASE_COMMAND --outf ./results/celeba_wgandec_quant_ncenc8 --lbd 1.0 --comp --ncenc 8 --nresenc 2 --netG ../WassersteinGAN/wgan_celeba_MTG/netG_epoch_99.pth --freezedec
#
## compress into generator, nenc=4, no gan penalty, deterministic encoder
$BASE_COMMAND --outf ./results/celeba_wgandec_lambda0_detenc --lbd 0.0 --nresenc 2 --netG ../WassersteinGAN/wgan_celeba_MTG/netG_epoch_99.pth --freezedec --detenc

$BASE_COMMAND --outf ./results/celeba_test --lbd 1.0 --comp --ncenc 4 --nresenc 2 --netG ../improved-wgan-pytroch/celeba_dcgan/generator.pt --freezedec
