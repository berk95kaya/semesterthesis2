
### MNIST
## Train WAE
python3 wae_comp.py --dataset mnist --dataroot ./data/ --testroot ./data/ --cuda --nz 128 --sigmasqz 1.0 --lr_eg 0.001 --niter 55 --decay_steps 30 50 --decay_gamma 0.4 --check_every 100 --workers 8 --recloss --mmd --bnz --outf ./results/mnist_wae_nz128 --lbd 100 --batchSize 256 --detenc --useenc --test_every 20 --addsamples 10000 --manualSeed 321

## WGAN-GP
python3 wae_comp.py --dataset mnist --dataroot ./data/ --testroot ./data/ --cuda --nz 128 --sigmasqz 1.0 --lr_eg 0.0001 --lr_di 0.0001 --beta1 0.5 --beta2 0.9 --niter 165 --check_every 100 --workers 6 --outf ./results/mnist_wgangp --batchSize 64 --test_every 100 --addsamples 10000 --manualSeed 321 --wganloss

## WGANAE
python3 wae_comp.py --dataset mnist --dataroot ./data/ --testroot ./data/ --cuda --nz 128 --sigmasqz 1.0 --lr_eg 0.0003 --niter 165 --decay_steps 100 140 --decay_gamma 0.4 --check_every 100 --workers 6 --recloss --mmd --bnz --outf ./results/mnist_wganae_nz128_int --lbd 100 --batchSize 256 --detenc --useenc --test_every 20 --addsamples 10000 --manualSeed 321 --wganloss --useencdist --lbd_di 0.000025 --intencprior

# encode
BASE_COMMAND=python3\ wae_comp.py\ --dataset\ mnist\ --dataroot\ ./data/\ --testroot\ ./data/\ --cuda\ --nz\ 128\ --sigmasqz\ 1.0\ --lr_eg\ 0.001\ --niter\ 55\ --decay_steps\ 30\ 50\ --decay_gamma\ 0.4\ --check_every\ 100\ --workers\ 6\ --recloss\ --mmd\ --bnz\ --batchSize\ 256\ --useenc\ --comp\ --freezedec\ --test_every\ 100\ --addsamples\ 10000\ --manualSeed\ 321

## ... into WAE
NRES_WAE=2
NETG_WAE=./results/mnist_wae_nz128/netG_epoch_54.pth
$BASE_COMMAND --outf ./results/mnist_waedec_ncenc0_lbd900 --lbd 900 --ncenc 1 --nquant 1 --nresenc $NRES_WAE --netG $NETG_WAE
$BASE_COMMAND --outf ./results/mnist_waedec_ncenc2_lbd600 --lbd 600 --ncenc 2 --nresenc $NRES_WAE --netG $NETG_WAE
$BASE_COMMAND --outf ./results/mnist_waedec_ncenc8_lbd300 --lbd 300 --ncenc 8 --nresenc $NRES_WAE --netG $NETG_WAE
$BASE_COMMAND --outf ./results/mnist_waedec_ncenc32_lbd300 --lbd 300 --ncenc 32 --nresenc $NRES_WAE --netG $NETG_WAE
$BASE_COMMAND --outf ./results/mnist_waedec_ncenc128_lbd300 --lbd 300 --ncenc 128 --nresenc $NRES_WAE --netG $NETG_WAE


$BASE_COMMAND --outf ./results/mnist_waedec_ncenc32_lbd150 --lbd 150 --ncenc 32 --nresenc $NRES_WAE --netG $NETG_WAE && \
$BASE_COMMAND --outf ./results/mnist_waedec_ncenc128_lbd100 --lbd 100 --ncenc 128 --nresenc $NRES_WAE --netG $NETG_WAE && \
$BASE_COMMAND --outf ./results/mnist_waedec_ncenc0_lbd2400 --lbd 2400 --ncenc 1 --nquant 1 --nresenc $NRES_WAE --netG $NETG_WAE


## ... into WGANAE
NRES_WGANAE=2
NETG_WGANAE=./results/mnist_wganae_nz128_int/netG_epoch_164.pth
$BASE_COMMAND --outf ./results/mnist_wganaedec_int_ncenc0_lbd900 --lbd 900 --ncenc 1 --nquant 1 --nresenc $NRES_WGANAE --netG $NETG_WGANAE
$BASE_COMMAND --outf ./results/mnist_wganaedec_int_ncenc2_lbd600 --lbd 600 --ncenc 2 --nresenc $NRES_WGANAE --netG $NETG_WGANAE
$BASE_COMMAND --outf ./results/mnist_wganaedec_int_ncenc8_lbd300 --lbd 300 --ncenc 8 --nresenc $NRES_WGANAE --netG $NETG_WGANAE
$BASE_COMMAND --outf ./results/mnist_wganaedec_int_ncenc32_lbd300 --lbd 300 --ncenc 32 --nresenc $NRES_WGANAE --netG $NETG_WGANAE
$BASE_COMMAND --outf ./results/mnist_wganaedec_int_ncenc128_lbd300 --lbd 300 --ncenc 128 --nresenc $NRES_WGANAE --netG $NETG_WGANAE

$BASE_COMMAND --outf ./results/mnist_wganaedec_int_ncenc32_lbd150 --lbd 150 --ncenc 32 --nresenc $NRES_WGANAE --netG $NETG_WGANAE && \
$BASE_COMMAND --outf ./results/mnist_wganaedec_int_ncenc0_lbd2400 --lbd 2400 --ncenc 1 --nquant 1 --nresenc $NRES_WGANAE --netG $NETG_WGANAE

$BASE_COMMAND --outf ./results/mnist_wganaedec_int_ncenc128_lbd100 --lbd 100 --ncenc 128 --nresenc $NRES_WGANAE --netG $NETG_WGANAE


## ... into WGAN-GP
NRES_WGANGP=2
NETG_WGANGP=./results/mnist_wgangp_sep/generator.pt
$BASE_COMMAND --outf ./results/mnist_wgangpdec_ncenc128_lbd100 --lbd 100 --ncenc 128 --nresenc $NRES_WGANGP --netG $NETG_WGANGP && \
$BASE_COMMAND --outf ./results/mnist_wgangpdec_ncenc32_lbd150 --lbd 150 --ncenc 32 --nresenc $NRES_WGANGP --netG $NETG_WGANGP && \
$BASE_COMMAND --outf ./results/mnist_wgangpdec_ncenc8_lbd300 --lbd 300 --ncenc 8 --nresenc $NRES_WGANGP --netG $NETG_WGANGP && \
$BASE_COMMAND --outf ./results/mnist_wgangpdec_ncenc2_lbd600 --lbd 600 --ncenc 2 --nresenc $NRES_WGANGP --netG $NETG_WGANGP && \
$BASE_COMMAND --outf ./results/mnist_wgangpdec_ncenc0_lbd2400 --lbd 2400 --ncenc 1 --nquant 1 --nresenc $NRES_WGANGP --netG $NETG_WGANGP




#### NEW LBD VALUES FOR LEARNING ENCODER

# encode
BASE_COMMAND=python3\ wae_comp.py\ --dataset\ mnist\ --dataroot\ ./data/\ --testroot\ ./data/\ --cuda\ --nz\ 128\ --sigmasqz\ 1.0\ --lr_eg\ 0.001\ --niter\ 55\ --decay_steps\ 30\ 50\ --decay_gamma\ 0.4\ --check_every\ 100\ --workers\ 6\ --recloss\ --mmd\ --bnz\ --batchSize\ 256\ --useenc\ --comp\ --freezedec\ --test_every\ 100\ --addsamples\ 10000\ --manualSeed\ 321\ --nresenc\ 2

## ... into WAE
NETG=./results/mnist_wae_nz128/netG_epoch_54.pth
DEC=waedec

$BASE_COMMAND --outf ./results/mnist_${DEC}_ncenc0_lbd2420 --lbd 2420 --ncenc 1 --nquant 1 --netG $NETG && \
$BASE_COMMAND --outf ./results/mnist_${DEC}_ncenc2_lbd739 --lbd 739 --ncenc 2 --netG $NETG && \
$BASE_COMMAND --outf ./results/mnist_${DEC}_ncenc8_lbd349 --lbd 349 --ncenc 8 --netG $NETG && \
$BASE_COMMAND --outf ./results/mnist_${DEC}_ncenc32_lbd201 --lbd 201 --ncenc 32 --netG $NETG && \
$BASE_COMMAND --outf ./results/mnist_${DEC}_ncenc128_lbd150 --lbd 150 --ncenc 128 --netG $NETG


## ... into WGANAE
NETG=./results/mnist_wganae_nz128_int/netG_epoch_164.pth
DEC=wganaedec


## ... into WGAN-GP
NETG=./results/mnist_wgangp_sep/generator.pt
DEC=wgangpdec


#######################################################################





## Compressive autoencoder baseline
BASE_COMMAND=python3\ wae_comp.py\ --dataset\ mnist\ --dataroot\ ./data/\ --testroot\ ./data/\ --cuda\ --nz\ 128\ --sigmasqz\ 1.0\ --lr_eg\ 0.001\ --niter\ 55\ --decay_steps\ 30\ 50\ --decay_gamma\ 0.4\ --check_every\ 100\ --workers\ 6\ --recloss\ --lbd\ 0.0\ --batchSize\ 256\ --useenc\ --comp\ --test_every\ 100\ --addsamples\ 10000\ --manualSeed\ 321

## ... into WAE
NRES_CAE=2
$BASE_COMMAND --outf ./results/mnist_cae_ncenc128 --ncenc 128 --nresenc $NRES_CAE && \
$BASE_COMMAND --outf ./results/mnist_cae_ncenc32 --ncenc 32 --nresenc $NRES_CAE && \
$BASE_COMMAND --outf ./results/mnist_cae_ncenc8 --ncenc 8 --nresenc $NRES_CAE && \
$BASE_COMMAND --outf ./results/mnist_cae_ncenc2 --ncenc 2 --nresenc $NRES_CAE && \
$BASE_COMMAND --outf ./results/mnist_cae_ncenc0 --ncenc 1 --nquant 1 --nresenc $NRES_CAE



## ECCV baseline
BASE_COMMAND=python3\ wae_comp.py\ --dataset\ mnist\ --dataroot\ ./data/\ --testroot\ ./data/\ --cuda\ --nz\ 128\ --sigmasqz\ 1.0\ --lr_eg\ 0.0003\ --niter\ 165\ --decay_steps\ 100\ 140\ --decay_gamma\ 0.4\ --check_every\ 100\ --workers\ 6\ --recloss\ --lbd\ 0.0\ --batchSize\ 256\ --nresenc\ 2\ --comp\ --useenc\ --test_every\ 20\ --addsamples\ 10000\ --manualSeed\ 321\ --wganloss\ --useencdist\ --upencwgan
$BASE_COMMAND --outf ./results/mnist_eccv_ncenc0 --ncenc 1 --nquant 1 --lbd_di 0.000605
$BASE_COMMAND --outf ./results/mnist_eccv_ncenc2 --ncenc 2 --lbd_di 0.000123
$BASE_COMMAND --outf ./results/mnist_eccv_ncenc8 --ncenc 8 --lbd_di 0.000087
$BASE_COMMAND --outf ./results/mnist_eccv_ncenc32 --ncenc 32 --lbd_di 0.000036
$BASE_COMMAND --outf ./results/mnist_eccv_ncenc128 --ncenc 128 --lbd_di 0.000025












