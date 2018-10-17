# WAE
python wae_comp.py --dataset celeba --dataroot ./data/celeba_waecrop/resized_celebA_train/ --testroot ./data/celeba_waecrop/resized_celebA_test/ --cuda --nz 128 --sigmasqz 1.0 --lr_eg 0.001 --niter 55 --decay_steps 30 50 --decay_gamma 0.4 --check_every 100 --workers 8 --recloss --mmd --bnz --outf ./results2/celeba_wae --lbd 100 --batchSize 256 --detenc --useenc --test_every 20 --addsamples 10000 --manualSeed 321


# celeba encode into wgandec
python wae_comp.py --dataset celeba --dataroot ./data/celeba_waecrop/resized_celebA_train/ --testroot ./data/celeba_waecrop/resized_celebA_test/ --cuda --nz 128 --sigmasqz 1.0 --lr_eg 0.001 --niter 55 --decay_steps 30 50 --decay_gamma 0.4 --check_every 100 --workers 8 --recloss --mmd --bnz --outf ./results2/celeba_wae_test --lbd 300 --batchSize 256 --useenc --comp --nresenc 2 --freezedec --ncenc 8 --netG ./results2/celeba_wae_nz128/netG_epoch_54.pth --test_every 1 --addsamples 10000 --manualSeed 321


# WGANAE
python wae_comp.py --dataset celeba --dataroot ./data/celeba_waecrop/resized_celebA_train/ --testroot ./data/celeba_waecrop/resized_celebA_test/ --cuda --nz 256 --sigmasqz 1.0 --lr_eg 0.0003 --niter 165 --decay_steps 100 140 --decay_gamma 0.4 --check_every 100 --workers 8 --recloss --mmd --bnz --outf ./results2/celeba_wganae --lbd 100 --batchSize 256 --detenc --useenc --test_every 20 --addsamples 10000 --manualSeed 321 --wganloss --useencdist --lbd_di 0.000025

python wae_comp.py --dataset celeba --dataroot ./data/celeba_waecrop/resized_celebA_train/ --testroot ./data/celeba_waecrop/resized_celebA_test/ --cuda --nz 128 --sigmasqz 1.0 --lr_eg 0.001 --niter 165 --decay_steps 100 140 --decay_gamma 0.4 --check_every 100 --workers 8 --recloss --mmd --bnz --outf ./results2/celeba_wganae_nz128 --lbd 100 --batchSize 256 --detenc --useenc --test_every 20 --addsamples 10000 --manualSeed 321 --wganloss --useencdist --lbd_di 0.000025


python wae_comp.py --dataset celeba --dataroot ./data/celeba_waecrop/resized_celebA_train/ --testroot ./data/celeba_waecrop/resized_celebA_test/ --cuda --nz 128 --sigmasqz 1.0 --lr_eg 0.0001 --lr_di 0.0001 --beta1 0.5 --beta2 0.9 --niter 165 --decay_steps 100 140 --decay_gamma 0.4 --check_every 100 --workers 6 --recloss --mmd --bnz --outf ./results2/celeba_wganae_int_nz128_wganparam --lbd 100 --batchSize 256 --detenc --useenc --test_every 20 --addsamples 10000 --manualSeed 321 --wganloss --useencdist --lbd_di 0.000025




# celeba encode into wganaedec
python wae_comp.py --dataset celeba --dataroot ./data/celeba_waecrop/resized_celebA_train/ --testroot ./data/celeba_waecrop/resized_celebA_test/ --cuda --nz 256 --sigmasqz 1.0 --lr_eg 0.001 --niter 55 --decay_steps 30 50 --decay_gamma 0.4 --check_every 100 --workers 8 --recloss --mmd --bnz --outf ./results2/celeba_wganaedec_ncenc8_lbd300 --lbd 300 --batchSize 256 --useenc --comp --nresenc 2 --freezedec --ncenc 8 --netG ./results2/celeba_wganae/netG_epoch_164.pth --test_every 10 --addsamples 10000 --manualSeed 321



# celeba HQ lbd_gp 250 wd 0.001 nz 256
python wae_comp.py --dataset celebahq --dataroot ./data/celebaHQ/celebA-HQ-png-128/ --testroot ./data/celebaHQ/celebA-HQ-png-128-test/ --cuda --nz 256 --sigmasqz 1.0 --lr_eg 0.0001 --lr_di 0.0001 --beta2 0.5 --beta2 0.9 --niter 500 --check_every 50 --workers 8 --wganloss --outf ./results2/celebahq_wgan-gp_lbdgp250 --batchSize 64 --test_every 10 --addsamples 10000 --manualSeed 321 --niter 500 --lbd_gp 250 --heinit --imageSize 128 --batchSize 60 --wd_di 0.001




#lsun encode into wgandec
python wae_comp.py --dataset lsun --dataroot ./data/lsun/ --testroot ./data/lsun/ --cuda --nz 256 --sigmasqz 1.0 --lr_eg 0.001 --niter 50 --check_every 100 --workers 8 --recloss --mmd --bnz --lbd 2000 --batchSize 256 --freezedec --useenc --test_every 100 --manualSeed 321 --comp --nresenc 4 --netG ../improved-wgan-pytorch/results_dcgan_lsun/generator.pt --lsun_custom_split --decay_steps 7 10 --decay_gamma 0.4 --outf ./results/lsun_wgangpdec_nz256_lbd2000_bs256 --ncenc 8

#lsun WGANAE
python wae_comp.py --dataset lsun --dataroot ./data/lsun/ --testroot ./data/lsun/ --cuda --nz 512 --sigmasqz 1.0 --lr_eg 0.0003 --niter 22 --decay_steps 15 20 --decay_gamma 0.4 --check_every 5 --workers 8 --recloss --mmd --bnz --outf ./results2/lsun_wganae_nz512 --lbd 300 --batchSize 256 --detenc --useenc --test_every 20 --addsamples 10000 --manualSeed 321 --wganloss --useencdist --lbd_di 0.0001 --lsun_custom_split


# celeba encode into wganaedec
python wae_comp.py --dataset lsun --dataroot ./data/lsun/ --testroot ./data/lsun/ --cuda --nz 256 --sigmasqz 1.0 --lr_eg 0.001 --niter 6 --decay_steps 3 5 --decay_gamma 0.4 --check_every 100 --workers 8 --recloss --mmd --bnz --outf ./results2/lsun_wganaedec_ncenc8_lbd600 --lbd 600 --batchSize 256 --useenc --comp --nresenc 2 --freezedec --ncenc 8 --netG ./results2/lsun_wganae/netG_epoch_10.pth --test_every 50 --addsamples 10000 --manualSeed 321 --lsun_custom_split
