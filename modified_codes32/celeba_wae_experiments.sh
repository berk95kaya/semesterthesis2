BASE_COMMAND=python\ wae_comp.py\ --dataset\ celeba\ --dataroot\ ./data/celeba_waecrop/resized_celebA_train/\ --testroot\ ./data/celeba_waecrop/resized_celebA_test/\ --cuda\ --nz\ 64\ --sigmasqz\ 2.0\ --avbtrick\ --lr_d\ 0.001\ --lr_eg\ 0.0003\ --niter\ 55\ --decay_steps\ 30\ 50\ --decay_gamma\ 0.4\ --check_every\ 25\ --workers\ 4
#
## plain WAE
$BASE_COMMAND --outf ./results/celeba_plain_wae --lbd 1.0
#
## compress into generator, nenc=4
$BASE_COMMAND --outf ./results/celeba_waedec_quant_ncenc4 --lbd 1.0 --comp --ncenc 4 --nresenc 2 --netG ./results/celeba_plain_wae/netG_epoch_54.pth --freezedec
#
## compress into generator, nenc=4, no gan penalty, deterministic encoder
$BASE_COMMAND --outf ./results/celeba_waedec_quant_ncenc4_lambda0_detenc --lbd 0.0 --comp --ncenc 4 --nresenc 2 --netG ./results/celeba_plain_wae/netG_epoch_54.pth --freezedec --detenc
#
## compress into generator, no gan penalty, deterministic encoder
$BASE_COMMAND --outf ./results/celeba_waedec_lambda0_detenc --lbd 0.0 --netG ./results/celeba_plain_wae/netG_epoch_54.pth --freezedec --detenc
#
## compress into generator, nenc=8
$BASE_COMMAND --outf ./results/celeba_waedec_quant_ncenc8 --lbd 1.0 --comp --ncenc 8 --nresenc 2 --netG ./results/celeba_plain_wae/netG_epoch_54.pth --freezedec
#
## compress into generator, nenc=8, no gan penalty, deterministic encoder
$BASE_COMMAND --outf ./results/celeba_waedec_quant_ncenc8_lambda0_detenc --lbd 0.0 --comp --ncenc 8 --nresenc 2 --netG ./results/celeba_plain_wae/netG_epoch_54.pth --freezedec --detenc
#
## plain autoencoder, quantized, ncenc=4
$BASE_COMMAND --outf ./results/celeba_plain_autoencoder_quant_ncenc4 --lbd 0.0 --detenc --comp --ncenc 4 --nresenc 2
## plain autoencoder, quantized, ncenc=8
$BASE_COMMAND --outf ./results/celeba_plain_autoencoder_quant_ncenc8 --lbd 0.0 --detenc --comp --ncenc 8 --nresenc 2
