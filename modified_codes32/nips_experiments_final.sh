### CELEB A
## Train WAE
python wae_comp.py --dataset celeba --dataroot ./data/celeba_waecrop/resized_celebA_train/ --testroot ./data/celeba_waecrop/resized_celebA_test/ --cuda --nz 128 --sigmasqz 1.0 --lr_eg 0.001 --niter 55 --decay_steps 30 50 --decay_gamma 0.4 --check_every 100 --workers 8 --recloss --mmd --bnz --outf ./results2/celeba_wae_nz128 --lbd 100 --batchSize 256 --detenc --useenc --test_every 20 --addsamples 10000 --manualSeed 321

## WGAN-GP
python wae_comp.py --dataset celeba --dataroot ./data/celeba_waecrop/resized_celebA_train/ --testroot ./data/celeba_waecrop/resized_celebA_test/ --cuda --nz 128 --sigmasqz 1.0 --lr_eg 0.0001 --lr_di 0.0001 --beta1 0.5 --beta2 0.9 --niter 165 --check_every 100 --workers 6 --outf ./results2/celeba_wgangp --batchSize 64 --test_every 100 --addsamples 10000 --manualSeed 321 --wganloss

## WGANAE
python wae_comp.py --dataset celeba --dataroot ./data/celeba_waecrop/resized_celebA_train/ --testroot ./data/celeba_waecrop/resized_celebA_test/ --cuda --nz 128 --sigmasqz 1.0 --lr_eg 0.0003 --niter 165 --decay_steps 100 140 --decay_gamma 0.4 --check_every 100 --workers 6 --recloss --mmd --bnz --outf ./results2/celeba_wganae_nz128_int --lbd 100 --batchSize 256 --detenc --useenc --test_every 20 --addsamples 10000 --manualSeed 321 --wganloss --useencdist --lbd_di 0.000025 --intencprior

# encode
BASE_COMMAND=python\ wae_comp.py\ --dataset\ celeba\ --dataroot\ ./data/celeba_waecrop/resized_celebA_train/\ --testroot\ ./data/celeba_waecrop/resized_celebA_test/\ --cuda\ --nz\ 128\ --sigmasqz\ 1.0\ --lr_eg\ 0.001\ --niter\ 55\ --decay_steps\ 30\ 50\ --decay_gamma\ 0.4\ --check_every\ 100\ --workers\ 6\ --recloss\ --mmd\ --bnz\ --batchSize\ 256\ --useenc\ --comp\ --freezedec\ --test_every\ 100\ --addsamples\ 10000\ --manualSeed\ 321

## ... into WAE
NRES_WAE=2
NETG_WAE=./results2/celeba_wae_nz128/netG_epoch_54.pth
$BASE_COMMAND --outf ./results2/celeba_waedec_ncenc0_lbd900 --lbd 900 --ncenc 1 --nquant 1 --nresenc $NRES_WAE --netG $NETG_WAE
$BASE_COMMAND --outf ./results2/celeba_waedec_ncenc2_lbd600 --lbd 600 --ncenc 2 --nresenc $NRES_WAE --netG $NETG_WAE
$BASE_COMMAND --outf ./results2/celeba_waedec_ncenc8_lbd300 --lbd 300 --ncenc 8 --nresenc $NRES_WAE --netG $NETG_WAE
$BASE_COMMAND --outf ./results2/celeba_waedec_ncenc32_lbd300 --lbd 300 --ncenc 32 --nresenc $NRES_WAE --netG $NETG_WAE
$BASE_COMMAND --outf ./results2/celeba_waedec_ncenc128_lbd300 --lbd 300 --ncenc 128 --nresenc $NRES_WAE --netG $NETG_WAE


$BASE_COMMAND --outf ./results2/celeba_waedec_ncenc32_lbd150 --lbd 150 --ncenc 32 --nresenc $NRES_WAE --netG $NETG_WAE && \
$BASE_COMMAND --outf ./results2/celeba_waedec_ncenc128_lbd100 --lbd 100 --ncenc 128 --nresenc $NRES_WAE --netG $NETG_WAE && \
$BASE_COMMAND --outf ./results2/celeba_waedec_ncenc0_lbd2400 --lbd 2400 --ncenc 1 --nquant 1 --nresenc $NRES_WAE --netG $NETG_WAE


## ... into WGANAE
NRES_WGANAE=2
NETG_WGANAE=./results2/celeba_wganae_nz128_int/netG_epoch_164.pth
$BASE_COMMAND --outf ./results2/celeba_wganaedec_int_ncenc0_lbd900 --lbd 900 --ncenc 1 --nquant 1 --nresenc $NRES_WGANAE --netG $NETG_WGANAE
$BASE_COMMAND --outf ./results2/celeba_wganaedec_int_ncenc2_lbd600 --lbd 600 --ncenc 2 --nresenc $NRES_WGANAE --netG $NETG_WGANAE
$BASE_COMMAND --outf ./results2/celeba_wganaedec_int_ncenc8_lbd300 --lbd 300 --ncenc 8 --nresenc $NRES_WGANAE --netG $NETG_WGANAE
$BASE_COMMAND --outf ./results2/celeba_wganaedec_int_ncenc32_lbd300 --lbd 300 --ncenc 32 --nresenc $NRES_WGANAE --netG $NETG_WGANAE
$BASE_COMMAND --outf ./results2/celeba_wganaedec_int_ncenc128_lbd300 --lbd 300 --ncenc 128 --nresenc $NRES_WGANAE --netG $NETG_WGANAE

$BASE_COMMAND --outf ./results2/celeba_wganaedec_int_ncenc32_lbd150 --lbd 150 --ncenc 32 --nresenc $NRES_WGANAE --netG $NETG_WGANAE && \
$BASE_COMMAND --outf ./results2/celeba_wganaedec_int_ncenc0_lbd2400 --lbd 2400 --ncenc 1 --nquant 1 --nresenc $NRES_WGANAE --netG $NETG_WGANAE

$BASE_COMMAND --outf ./results2/celeba_wganaedec_int_ncenc128_lbd100 --lbd 100 --ncenc 128 --nresenc $NRES_WGANAE --netG $NETG_WGANAE


## ... into WGAN-GP
NRES_WGANGP=2
NETG_WGANGP=./results2/celeba_wgangp_sep/generator.pt
$BASE_COMMAND --outf ./results2/celeba_wgangpdec_ncenc128_lbd100 --lbd 100 --ncenc 128 --nresenc $NRES_WGANGP --netG $NETG_WGANGP && \
$BASE_COMMAND --outf ./results2/celeba_wgangpdec_ncenc32_lbd150 --lbd 150 --ncenc 32 --nresenc $NRES_WGANGP --netG $NETG_WGANGP && \
$BASE_COMMAND --outf ./results2/celeba_wgangpdec_ncenc8_lbd300 --lbd 300 --ncenc 8 --nresenc $NRES_WGANGP --netG $NETG_WGANGP && \
$BASE_COMMAND --outf ./results2/celeba_wgangpdec_ncenc2_lbd600 --lbd 600 --ncenc 2 --nresenc $NRES_WGANGP --netG $NETG_WGANGP && \
$BASE_COMMAND --outf ./results2/celeba_wgangpdec_ncenc0_lbd2400 --lbd 2400 --ncenc 1 --nquant 1 --nresenc $NRES_WGANGP --netG $NETG_WGANGP




#### NEW LBD VALUES FOR LEARNING ENCODER

# encode
BASE_COMMAND=python\ wae_comp.py\ --dataset\ celeba\ --dataroot\ ./data/celeba_waecrop/resized_celebA_train/\ --testroot\ ./data/celeba_waecrop/resized_celebA_test/\ --cuda\ --nz\ 128\ --sigmasqz\ 1.0\ --lr_eg\ 0.001\ --niter\ 55\ --decay_steps\ 30\ 50\ --decay_gamma\ 0.4\ --check_every\ 100\ --workers\ 6\ --recloss\ --mmd\ --bnz\ --batchSize\ 256\ --useenc\ --comp\ --freezedec\ --test_every\ 100\ --addsamples\ 10000\ --manualSeed\ 321\ --nresenc\ 2

## ... into WAE
NETG=./results2/celeba_wae_nz128/netG_epoch_54.pth
DEC=waedec

$BASE_COMMAND --outf ./results2/celeba_${DEC}_ncenc0_lbd2420 --lbd 2420 --ncenc 1 --nquant 1 --netG $NETG && \
$BASE_COMMAND --outf ./results2/celeba_${DEC}_ncenc2_lbd739 --lbd 739 --ncenc 2 --netG $NETG && \
$BASE_COMMAND --outf ./results2/celeba_${DEC}_ncenc8_lbd349 --lbd 349 --ncenc 8 --netG $NETG && \
$BASE_COMMAND --outf ./results2/celeba_${DEC}_ncenc32_lbd201 --lbd 201 --ncenc 32 --netG $NETG && \
$BASE_COMMAND --outf ./results2/celeba_${DEC}_ncenc128_lbd150 --lbd 150 --ncenc 128 --netG $NETG


## ... into WGANAE
NETG=./results2/celeba_wganae_nz128_int/netG_epoch_164.pth
DEC=wganaedec


## ... into WGAN-GP
NETG=./results2/celeba_wgangp_sep/generator.pt
DEC=wgangpdec


######





## Compressive autoencoder baseline
BASE_COMMAND=python\ wae_comp.py\ --dataset\ celeba\ --dataroot\ ./data/celeba_waecrop/resized_celebA_train/\ --testroot\ ./data/celeba_waecrop/resized_celebA_test/\ --cuda\ --nz\ 128\ --sigmasqz\ 1.0\ --lr_eg\ 0.001\ --niter\ 55\ --decay_steps\ 30\ 50\ --decay_gamma\ 0.4\ --check_every\ 100\ --workers\ 6\ --recloss\ --lbd\ 0.0\ --batchSize\ 256\ --useenc\ --comp\ --test_every\ 100\ --addsamples\ 10000\ --manualSeed\ 321

## ... into WAE
NRES_CAE=2
$BASE_COMMAND --outf ./results2/celeba_cae_ncenc128 --ncenc 128 --nresenc $NRES_CAE && \
$BASE_COMMAND --outf ./results2/celeba_cae_ncenc32 --ncenc 32 --nresenc $NRES_CAE && \
$BASE_COMMAND --outf ./results2/celeba_cae_ncenc8 --ncenc 8 --nresenc $NRES_CAE && \
$BASE_COMMAND --outf ./results2/celeba_cae_ncenc2 --ncenc 2 --nresenc $NRES_CAE && \
$BASE_COMMAND --outf ./results2/celeba_cae_ncenc0 --ncenc 1 --nquant 1 --nresenc $NRES_CAE



## ECCV baseline
BASE_COMMAND=python\ wae_comp.py\ --dataset\ celeba\ --dataroot\ ./data/celeba_waecrop/resized_celebA_train/\ --testroot\ ./data/celeba_waecrop/resized_celebA_test/\ --cuda\ --nz\ 128\ --sigmasqz\ 1.0\ --lr_eg\ 0.0003\ --niter\ 165\ --decay_steps\ 100\ 140\ --decay_gamma\ 0.4\ --check_every\ 100\ --workers\ 6\ --recloss\ --lbd\ 0.0\ --batchSize\ 256\ --nresenc\ 2\ --comp\ --useenc\ --test_every\ 20\ --addsamples\ 10000\ --manualSeed\ 321\ --wganloss\ --useencdist\ --upencwgan
$BASE_COMMAND --outf ./results2/celeba_eccv_ncenc0 --ncenc 1 --nquant 1 --lbd_di 0.000605
$BASE_COMMAND --outf ./results2/celeba_eccv_ncenc2 --ncenc 2 --lbd_di 0.000123
$BASE_COMMAND --outf ./results2/celeba_eccv_ncenc8 --ncenc 8 --lbd_di 0.000087
$BASE_COMMAND --outf ./results2/celeba_eccv_ncenc32 --ncenc 32 --lbd_di 0.000036
$BASE_COMMAND --outf ./results2/celeba_eccv_ncenc128 --ncenc 128 --lbd_di 0.000025



# CELEBA HQ wganae

python wae_comp.py --dataset celebahq --dataroot ./data/celebaHQ/celebA-HQ-png-128/ --testroot ./data/celebaHQ/celebA-HQ-png-128-test/ --cuda --nz 512 --sigmasqz 1.0 --lr_eg 0.0003 --niter 1650 --decay_steps 1000 1400 --decay_gamma 0.4 --check_every 200 --workers 6 --recloss --mmd --bnz --outf ./results3/celebahq_wganae_nz512_int --lbd 100 --batchSize 256 --detenc --useenc --test_every 20 --addsamples 10000 --manualSeed 321 --wganloss --useencdist --lbd_di 0.000025 --intencprior --ngpus 4 --imageSize 128










### LSUN
## Train WGANAE
python wae_comp.py --dataset lsun --dataroot ./data/lsun/ --testroot ./data/lsun/ --cuda --nz 512 --sigmasqz 1.0 --lr_eg 0.0001 --lr_di 0.0001 --beta1 0.5 --beta2 0.9 --niter 22 --decay_steps 15 20 --decay_gamma 0.4 --check_every 5 --workers 8 --recloss --mmd --bnz --outf ./results2/lsun_wganae_int_nz512 --lbd 300 --batchSize 256 --detenc --useenc --test_every 20 --addsamples 10000 --manualSeed 321 --wganloss --useencdist --lbd_di 0.0001 --lsun_custom_split


## WGAN-GP
python wae_comp.py --dataset lsun --dataroot ./data/lsun/ --testroot ./data/lsun/ --cuda --nz 512 --sigmasqz 1.0 --lr_eg 0.0001 --lr_di 0.0001 --beta1 0.5 --beta2 0.9 --niter 22 --check_every 5 --workers 6 --outf ./results2/lsun_wgangp_nz512 --batchSize 64 --test_every 100 --addsamples 10000 --manualSeed 321 --wganloss --lsun_custom_split


## WAE
python wae_comp.py --dataset lsun --dataroot ./data/lsun/ --testroot ./data/lsun/ --cuda --nz 512 --sigmasqz 1.0 --lr_eg 0.001 --niter 6 --decay_steps 3 4 --decay_gamma 0.4 --check_every 100 --workers 6 --recloss --mmd --bnz --outf ./results2/lsun_wae_nz512 --lbd 300 --batchSize 256 --detenc --useenc --test_every 20 --addsamples 10000 --manualSeed 321 --lsun_custom_split


### Encode
BASE_COMMAND=python\ wae_comp.py\ --dataset\ celeba\ --dataroot\ ./data/celeba_waecrop/resized_celebA_train/\ --testroot\ ./data/celeba_waecrop/resized_celebA_test/\ --cuda\ --nz\ 128\ --sigmasqz\ 1.0\ --lr_eg\ 0.001\ --niter\ 55\ --decay_steps\ 30\ 50\ --decay_gamma\ 0.4\ --check_every\ 100\ --workers\ 6\ --recloss\ --mmd\ --bnz\ --batchSize\ 256\ --useenc\ --comp\ --freezedec\ --test_every\ 100\ --addsamples\ 10000\ --manualSeed\ 321


### Encode 2
BASE_COMMAND=python wae_comp.py --dataset lsun --dataroot ./data/lsun/ --testroot ./data/lsun/ --cuda --nz 512 --sigmasqz 1.0 --lr_eg 0.001 --niter 6 --decay_steps 3 4 --decay_gamma 0.4 --check_every 100 --workers 6 --recloss --mmd --bnz --batchSize 256 --useenc --comp --freezedec --test_every 100 --addsamples 10000 --manualSeed 321 --lsun_custom_split --ncenc 0 --nquant 1 --nresenc 4 --outf ./results2/lsun_wganaedec_int_ncenc0_lbd2500 --lbd 2500 --netG ./results2/lsun_wganae_int_nz512/netG_epoch_21.pth



### Encode into ...
BASE_COMMAND=python\ wae_comp.py\ --dataset\ lsun\ --dataroot\ ./data/lsun/\ --testroot\ ./data/lsun/\ --cuda\ --nz\ 512\ --sigmasqz\ 1.0\ --lr_eg\ 0.001\ --niter\ 7\ --decay_steps\ 4\ 5\ --decay_gamma\ 0.4\ --check_every\ 100\ --workers\ 6\ --recloss\ --mmd\ --bnz\ --batchSize\ 256\ --useenc\ --comp\ --freezedec\ --test_every\ 100\ --addsamples\ 10000\ --manualSeed\ 321\ --lsun_custom_split\ --nresenc\ 4
## WGAN-GP
NETG=./results2/lsun_wgangp_nz512/netG_epoch_21.pth
DEC=wgangpdec
##
$BASE_COMMAND --netG $NETG --outf ./results2/lsun_${DEC}_ncenc0_lbd8376 --ncenc 1 --nquant 1 --lbd 8376 && \
$BASE_COMMAND --netG $NETG --outf ./results2/lsun_${DEC}__ncenc2_lbd3528 --ncenc 2 --lbd 3528 && \
$BASE_COMMAND --netG $NETG --outf ./results2/lsun_${DEC}_ncenc8_lbd1907 --ncenc 8 --lbd 1907 && \
$BASE_COMMAND --netG $NETG --outf ./results2/lsun_${DEC}_ncenc32_lbd923 --ncenc 32 --lbd 923 && \
$BASE_COMMAND --netG $NETG --outf ./results2/lsun_${DEC}_ncenc128_lbd516 --ncenc 128 --lbd 516 && \
$BASE_COMMAND --netG $NETG --outf ./results2/lsun_${DEC}_ncenc256_lbd400 --ncenc 256 --lbd 400 && \


## WGANAE
NETG=./results2/lsun_wganae_int_nz512/netG_epoch_21.pth
DEC=wganaedec
##
$BASE_COMMAND --netG $NETG --outf ./results2/lsun_${DEC}_ncenc0_lbd8376 --ncenc 1 --nquant 1 --lbd 8376 && \
$BASE_COMMAND --netG $NETG --outf ./results2/lsun_${DEC}__ncenc2_lbd3528 --ncenc 2 --lbd 3528 && \
$BASE_COMMAND --netG $NETG --outf ./results2/lsun_${DEC}_ncenc8_lbd1907 --ncenc 8 --lbd 1907 && \
$BASE_COMMAND --netG $NETG --outf ./results2/lsun_${DEC}_ncenc32_lbd923 --ncenc 32 --lbd 923 && \
$BASE_COMMAND --netG $NETG --outf ./results2/lsun_${DEC}_ncenc128_lbd516 --ncenc 128 --lbd 516 && \
$BASE_COMMAND --netG $NETG --outf ./results2/lsun_${DEC}_ncenc256_lbd400 --ncenc 256 --lbd 400 && \

## WAE
NETG=./results2/lsun_wae_nz512/netG_epoch_5.pth
DEC=waedec
##
$BASE_COMMAND --netG $NETG --outf ./results2/lsun_${DEC}_ncenc0_lbd16752 --ncenc 1 --nquant 1 --lbd 16752 && \
$BASE_COMMAND --netG $NETG --outf ./results2/lsun_${DEC}__ncenc2_lbd7056 --ncenc 2 --lbd 7056 && \
$BASE_COMMAND --netG $NETG --outf ./results2/lsun_${DEC}_ncenc8_lbd3815 --ncenc 8 --lbd 3815 && \
$BASE_COMMAND --netG $NETG --outf ./results2/lsun_${DEC}_ncenc32_lbd1846 --ncenc 32 --lbd 1846 && \
$BASE_COMMAND --netG $NETG --outf ./results2/lsun_${DEC}_ncenc128_lbd1032 --ncenc 128 --lbd 1032 && \
$BASE_COMMAND --netG $NETG --outf ./results2/lsun_${DEC}_ncenc256_lbd800 --ncenc 256 --lbd 800 && \


# Encode into wgangp without rate contstraint
python wae_comp.py --dataset lsun --dataroot ./data/lsun/ --testroot ./data/lsun/ --cuda --nz 512 --sigmasqz 1.0 --lr_eg 0.001 --niter 7 --decay_steps 4 5 --decay_gamma 0.4 --check_every 100 --workers 6 --recloss --lbd 0.0 --batchSize 256 --useenc --freezedec --test_every 100 --addsamples 10000 --manualSeed 321 --lsun_custom_split --nresenc 4 --netG ./results2/lsun_wgangp_nz512/netG_epoch_21.pth --outf ./results2/lsun_wgangpdec_nocomp_mseonly


## Compressive autoencoder baseline
BASE_COMMAND=python\ wae_comp.py\ --dataset\ lsun\ --dataroot\ ./data/lsun/\ --testroot\ ./data/lsun/\ --cuda\ --nz\ 512\ --sigmasqz\ 1.0\ --lr_eg\ 0.001\ --niter\ 7\ --decay_steps\ 4\ 5\ --decay_gamma\ 0.4\ --check_every\ 100\ --workers\ 6\ --recloss\ --lbd\ 0.0\ --batchSize\ 256\ --useenc\ --comp\ --test_every\ 100\ --addsamples\ 10000\ --manualSeed\ 321\ --lsun_custom_split

##
NRES_CAE=4
$BASE_COMMAND --outf ./results2/lsun_cae_ncenc256 --ncenc 256 --nresenc $NRES_CAE && \
$BASE_COMMAND --outf ./results2/lsun_cae_ncenc128 --ncenc 128 --nresenc $NRES_CAE && \
$BASE_COMMAND --outf ./results2/lsun_cae_ncenc32 --ncenc 32 --nresenc $NRES_CAE && \
$BASE_COMMAND --outf ./results2/lsun_cae_ncenc8 --ncenc 8 --nresenc $NRES_CAE && \
$BASE_COMMAND --outf ./results2/lsun_cae_ncenc2 --ncenc 2 --nresenc $NRES_CAE && \
$BASE_COMMAND --outf ./results2/lsun_cae_ncenc0 --ncenc 1 --nquant 1 --nresenc $NRES_CAE




### ECCV baseline LSUN
BASE_COMMAND=python\ wae_comp.py\ --dataset\ lsun\ --dataroot\ ./data/lsun/\ --testroot\ ./data/lsun/\ --cuda\ --nz\ 512\ --sigmasqz\ 1.0\ --lr_eg\ 0.0001\ --lr_di\ 0.0001\ --beta1\ 0.5\ --beta2\ 0.9\ --niter\ 22\ --decay_steps\ 15\ 20\ --decay_gamma\ 0.4\ --check_every\ 5\ --workers\ 8\ --recloss\ --lbd\ 0.0\ --batchSize\ 256\ --nresenc\ 4\ --comp\ --useenc\ --test_every\ 10\ --addsamples\ 10000\ --manualSeed\ 321\ --wganloss\ --useencdist\ --upencwgan\ --lsun_custom_split
$BASE_COMMAND --outf ./results3/lsun_eccv_ncenc0 --ncenc 1 --nquant 1 --lbd_di 0.002792
$BASE_COMMAND --outf ./results3/lsun_eccv_ncenc2_2 --ncenc 2 --lbd_di 0.001176
$BASE_COMMAND --outf ./results3/lsun_eccv_ncenc8_2 --ncenc 8 --lbd_di 0.000635
$BASE_COMMAND --outf ./results3/lsun_eccv_ncenc32 --ncenc 32 --lbd_di 0.000307
$BASE_COMMAND --outf ./results3/lsun_eccv_ncenc128 --ncenc 128 --lbd_di 0.000172
