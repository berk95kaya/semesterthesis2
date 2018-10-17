### CELEBA

### TRAIN WAE
python wae_comp.py --dataset celeba --dataroot ./data/celeba_waecrop/resized_celebA_train/ --testroot ./data/celeba_waecrop/resized_celebA_test/ --cuda --nz 128 --sigmasqz 1.0 --lr_eg 0.0003 --niter 55 --decay_steps 30 50 --decay_gamma 0.4 --check_every 100 --workers 8 --recloss --mmd --bnz --outf ./results/celeba_wae_detenc_mmd_bnz --lbd 100 --batchSize 256 --detenc --useenc --test_every 10 --addsamples 10000 --manualSeed 321

### TRAIN WGAN
python wae_comp.py --dataset celeba --dataroot ./data/celeba_waecrop/resized_celebA_train/ --testroot ./data/celeba_waecrop/resized_celebA_test/ --cuda --nz 128 --sigmasqz 1.0 --lr_eg 0.0001 --lr_di 0.0001 --beta2 0.9 --niter 55 --check_every 100 --workers 8 --wganloss --outf ./results/celeba_wgan-gp --batchSize 64 --test_every 10 --addsamples 10000 --manualSeed 321 --niter 125

### TRAIN WGANAE
python wae_comp.py --dataset celeba --dataroot ./data/celeba_waecrop/resized_celebA_train/ --testroot ./data/celeba_waecrop/resized_celebA_test/ --cuda --nz 128 --sigmasqz 1.0 --lr_di 0.001 --lr_eg 0.0001 --beta2 0.9 --niter 55 --decay_steps 30 50 --decay_gamma 0.4 --check_every 10 --workers 8 --recloss --wganloss --mmd --bnz --outf ./results/celeba_wganae_mmd_bnz_decay --lbd 100 --lbd_di 0.001 --batchSize 256 --detenc --useenc --test_every 10 --addsamples 10000 --manualSeed 321

python wae_comp.py --dataset celeba --dataroot ./data/celeba_waecrop/resized_celebA_train/ --testroot ./data/celeba_waecrop/resized_celebA_test/ --cuda --nz 128 --sigmasqz 1.0 --lr_di 0.001 --lr_eg 0.0001 --beta2 0.9 --niter 55 --decay_steps 30 50 --decay_gamma 0.4 --check_every 10 --workers 8 --recloss --wganloss --mmd --bnz --outf ./results/celeba_wganae_mmd_bnz_decay_lbd_di000025 --lbd 100 --lbd_di 0.00025 --batchSize 256 --detenc --useenc --test_every 10 --addsamples 10000 --manualSeed 321


### WAE experiments
BASE_COMMAND=python\ wae_comp.py\ --dataset\ celeba\ --dataroot\ ./data/celeba_waecrop/resized_celebA_train/\ --testroot\ ./data/celeba_waecrop/resized_celebA_test/\ --cuda\ --nz\ 128\ --sigmasqz\ 1.0\ --lr_eg\ 0.001\ --niter\ 50\ --check_every\ 100\ --workers\ 4\ --recloss\ --mmd\ --bnz\ --lbd\ 500\ --batchSize\ 256\ --freezedec\ --useenc\ --test_every\ 100\ --manualSeed\ 321\ --comp\ --nresenc\ 4\ --netG\ ./results/celeba_wae_detenc_mmd_bnz/netG_epoch_54.pth

$BASE_COMMAND --outf ./results/celeba_waedec_mmd_bnz_ncenc128_lbd500 --ncenc 128 &&\
$BASE_COMMAND --outf ./results/celeba_waedec_mmd_bnz_ncenc32_lbd500 --ncenc 32 &&\
$BASE_COMMAND --outf ./results/celeba_waedec_detenc_mmd_bnz_ncenc8_lbd500 --ncenc 8 &&\
$BASE_COMMAND --outf ./results/celeba_waedec_detenc_mmd_bnz_ncenc4_lbd500 --ncenc 4 &&\
$BASE_COMMAND --outf ./results/celeba_waedec_detenc_mmd_bnz_ncenc2_lbd500 --ncenc 2 &&\
$BASE_COMMAND --outf ./results/celeba_waedec_detenc_mmd_bnz_ncenc0_lbd500 --ncenc 1 --nquant 1


### WGAN-AE experiments
BASE_COMMAND=python\ wae_comp.py\ --dataset\ celeba\ --dataroot\ ./data/celeba_waecrop/resized_celebA_train/\ --testroot\ ./data/celeba_waecrop/resized_celebA_test/\ --cuda\ --nz\ 128\ --sigmasqz\ 1.0\ --lr_eg\ 0.001\ --niter\ 50\ --check_every\ 100\ --workers\ 4\ --recloss\ --mmd\ --bnz\ --lbd\ 400\ --batchSize\ 256\ --freezedec\ --useenc\ --test_every\ 100\ --manualSeed\ 321\ --comp\ --nresenc\ 4\ --netG\ ./results/celeba_wganae_mmd_bnz_decay/netG_epoch_54.pth

$BASE_COMMAND --outf ./results/celeba_wganaedec_mmd_bnz_ncenc128_lbd500 --ncenc 128 &&\
$BASE_COMMAND --outf ./results/celeba_wganaedec_mmd_bnz_ncenc32_lbd500 --ncenc 32 &&\
$BASE_COMMAND --outf ./results/celeba_wganaedec_detenc_mmd_bnz_ncenc8_lbd500 --ncenc 8 &&\
$BASE_COMMAND --outf ./results/celeba_wganaedec_detenc_mmd_bnz_ncenc4_lbd500 --ncenc 4 &&\
$BASE_COMMAND --outf ./results/celeba_wganaedec_detenc_mmd_bnz_ncenc2_lbd500 --ncenc 2 &&\
$BASE_COMMAND --outf ./results/celeba_wganaedec_detenc_mmd_bnz_ncenc0_lbd500 --ncenc 1 --nquant 1

# with noisy
BASE_COMMAND=python\ wae_comp.py\ --dataset\ celeba\ --dataroot\ ./data/celeba_waecrop/resized_celebA_train/\ --testroot\ ./data/celeba_waecrop/resized_celebA_test/\ --cuda\ --nz\ 128\ --sigmasqz\ 1.0\ --lr_eg\ 0.001\ --niter\ 50\ --check_every\ 100\ --workers\ 4\ --recloss\ --mmd\ --bnz\ --lbd\ 400\ --batchSize\ 256\ --freezedec\ --useenc\ --test_every\ 100\ --manualSeed\ 321\ --comp\ --nresenc\ 4\ --netG\ ./results/celeba_wganae_mmd_bnz_decay/netG_epoch_54.pth

$BASE_COMMAND --outf ./results/celeba_wganaedec_mmd_bnz_ncenc128_lbd500 --ncenc 128 &&\
$BASE_COMMAND --outf ./results/celeba_wganaedec_mmd_bnz_ncenc32_lbd500 --ncenc 32 &&\
$BASE_COMMAND --outf ./results/celeba_wganaedec_detenc_mmd_bnz_ncenc8_lbd500 --ncenc 8 &&\
$BASE_COMMAND --outf ./results/celeba_wganaedec_detenc_mmd_bnz_ncenc4_lbd500 --ncenc 4 &&\
$BASE_COMMAND --outf ./results/celeba_wganaedec_detenc_mmd_bnz_ncenc2_lbd500 --ncenc 2 &&\
$BASE_COMMAND --outf ./results/celeba_wganaedec_detenc_mmd_bnz_ncenc0_lbd500 --ncenc 1 --nquant 1


### WGAN experiments
BASE_COMMAND=python\ wae_comp.py\ --dataset\ celeba\ --dataroot\ ./data/celeba_waecrop/resized_celebA_train/\ --testroot\ ./data/celeba_waecrop/resized_celebA_test/\ --cuda\ --nz\ 128\ --sigmasqz\ 1.0\ --lr_eg\ 0.001\ --niter\ 50\ --check_every\ 100\ --workers\ 4\ --recloss\ --mmd\ --bnz\ --lbd\ 400\ --batchSize\ 256\ --freezedec\ --useenc\ --test_every\ 100\ --manualSeed\ 321\ --comp\ --nresenc\ 4\ --netG\ ./results/celeba_wgan-gp/netG_epoch_124.pth

$BASE_COMMAND --outf ./results/celeba_wgandec_ncenc128 --ncenc 128 &&\
$BASE_COMMAND --outf ./results/celeba_wganaedec_ncenc32 --ncenc 32 &&\
$BASE_COMMAND --outf ./results/celeba_wganaedec_detenc_ncenc8 --ncenc 8 &&\
$BASE_COMMAND --outf ./results/celeba_wganaedec_detenc_ncenc4 --ncenc 4 &&\
$BASE_COMMAND --outf ./results/celeba_wganaedec_detenc_ncenc2 --ncenc 2 &&\
$BASE_COMMAND --outf ./results/celeba_wganaedec_detenc_ncenc0 --ncenc 1 --nquant 1






### LSUN

### WAE
python wae_comp.py --dataset lsun --dataroot ./data/lsun/ --testroot ./data/lsun/ --cuda --nz 512 --sigmasqz 1.0 --lr_eg 0.0003 --niter 25 --decay_steps 20 25 --decay_gamma 0.4 --check_every 100 --workers 4 --recloss --mmd --bnz --outf ./results/lsun_wae_detenc_mmd_bnz --lbd 100 --batchSize 256 --detenc --useenc --test_every 10 --addsamples 10000 --manualSeed 321

### WGANAE
python wae_comp.py --dataset lsun --dataroot ./data/lsun/ --testroot ./data/lsun/ --cuda --nz 256 --sigmasqz 1.0 --lr_di 0.001 --lr_eg 0.0001 --niter 50 --check_every 10 --workers 8 --recloss --wganloss --mmd --bnz --outf ./results/lsun_wganae_mmd_bnz --lbd 100 --lbd_di 0.0001 --batchSize 256 --detenc --useenc --test_every 10 --addsamples 10000 --manualSeed 321


### WGANAE nz lbd up
python wae_comp.py --dataset lsun --dataroot ./data/lsun/ --testroot ./data/lsun/ --cuda --nz 512 --sigmasqz 1.0 --lr_di 0.001 --lr_eg 0.0001 --niter 25 --check_every 10 --workers 8 --recloss --wganloss --mmd --bnz --outf ./results/lsun_wganae_mmd_bnz_nz512_lbd500 --lbd 500 --lbd_di 0.0001 --batchSize 256 --detenc --useenc --test_every 10 --addsamples 10000 --manualSeed 321 --lsun_custom_split

### WGAN
python wae_comp.py --dataset lsun --dataroot ./data/lsun/ --testroot ./data/lsun/ --cuda --nz 512 --sigmasqz 1.0 --lr_eg 0.0001 --lr_di 0.0001 --beta2 0.9 --niter 55 --check_every 100 --workers 8 --wganloss --outf ./results/lsun_wgan-gp_nz512 --batchSize 64 --test_every 10 --addsamples 10000 --manualSeed 321 --lsun_custom_split
