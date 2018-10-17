
### MNIST
## Train WAE

# Bu WAE paperindaki sistem
python3 wae_comp.py --dataset mnist --dataroot ./data/ --testroot ./data/ --cuda --nz 8 --sigmasqz 1.0 --lr_eg 0.001 --niter 55 --decay_steps 30 50 --decay_gamma 0.4 --check_every 100 --workers 8 --recloss --mmd --bnz --outf ./results/mnist_wae_nz8_paper --lbd 10 --batchSize 256 --detenc --useenc --test_every 20 --addsamples 10000 --manualSeed 325 --imageSize 32 --wenc --paper


# encode
BASE_COMMAND=python3\ wae_comp.py\ --dataset\ mnist\ --dataroot\ ./data/\ --testroot\ ./data/\ --cuda\ --nz\ 8\ --sigmasqz\ 1.0\ --lr_eg\ 0.001\ --niter\ 55\ --decay_steps\ 30\ 50\ --decay_gamma\ 0.4\ --check_every\ 100\ --workers\ 6\ --recloss\ --mmd\ --bnz\ --batchSize\ 256\ --useenc\ --comp\ --freezedec\ --test_every\ 100\ --addsamples\ 10000\ --manualSeed\ 324\ --imageSize\ 32\ --wenc\  --paper

## ... into WAE
NRES_WAE=2
NETG_WAE=./results/mnist_wae_nz8_paper/netG_epoch_54.pth

$BASE_COMMAND --outf ./results/mnist_waedec_ncenc8_lbd10_nquant2_paper --lbd 10 --ncenc 8 --nquant 2 --nresenc $NRES_WAE --netG $NETG_WAE
$BASE_COMMAND --outf ./results/mnist_waedec_ncenc4_lbd10_nquant2_paper --lbd 10 --ncenc 4 --nquant 2 --nresenc $NRES_WAE --netG $NETG_WAE



# Bu da normal
python3 wae_comp.py --dataset mnist --dataroot ./data/ --testroot ./data/ --cuda --nz 8 --sigmasqz 1.0 --lr_eg 0.001 --niter 55 --decay_steps 30 50 --decay_gamma 0.4 --check_every 100 --workers 8 --recloss --mmd --bnz --outf ./results/mnist_wae_nz8 --lbd 10 --batchSize 256 --detenc --useenc --test_every 20 --addsamples 10000 --manualSeed 325 --imageSize 32 --wenc 


# encode
BASE_COMMAND=python3\ wae_comp.py\ --dataset\ mnist\ --dataroot\ ./data/\ --testroot\ ./data/\ --cuda\ --nz\ 8\ --sigmasqz\ 1.0\ --lr_eg\ 0.001\ --niter\ 55\ --decay_steps\ 30\ 50\ --decay_gamma\ 0.4\ --check_every\ 100\ --workers\ 6\ --recloss\ --mmd\ --bnz\ --batchSize\ 256\ --useenc\ --comp\ --freezedec\ --test_every\ 100\ --addsamples\ 10000\ --manualSeed\ 323\ --imageSize\ 32\ --wenc

## ... into WAE
NRES_WAE=2
NETG_WAE=./results/mnist_wae_nz8/netG_epoch_54.pth

$BASE_COMMAND --outf ./results/mnist_waedec_ncenc8_lbd10_nquant2 --lbd 10 --ncenc 8 --nquant 2 --nresenc $NRES_WAE --netG $NETG_WAE
$BASE_COMMAND --outf ./results/mnist_waedec_ncenc4_lbd10_nquant2 --lbd 10 --ncenc 4 --nquant 2 --nresenc $NRES_WAE --netG $NETG_WAE


#### NOW TRY EVERYTING BY REMOVING QUANTIZATION #############

# Bu WAE paperindaki sistem

# encode
BASE_COMMAND=python3\ wae_comp.py\ --dataset\ mnist\ --dataroot\ ./data/\ --testroot\ ./data/\ --cuda\ --nz\ 8\ --sigmasqz\ 1.0\ --lr_eg\ 0.001\ --niter\ 55\ --decay_steps\ 30\ 50\ --decay_gamma\ 0.4\ --check_every\ 100\ --workers\ 6\ --recloss\ --mmd\ --bnz\ --batchSize\ 256\ --useenc\ --comp\ --freezedec\ --test_every\ 100\ --addsamples\ 10000\ --manualSeed\ 324\ --imageSize\ 32\ --wenc\  --paper\ --rmvquant

## ... into WAE
NRES_WAE=2
NETG_WAE=./results/mnist_wae_nz8_paper/netG_epoch_54.pth

$BASE_COMMAND --outf ./results/mnist_waedec_ncenc8_lbd10_nquant2_paper_rmvquant --lbd 10 --ncenc 8 --nquant 2 --nresenc $NRES_WAE --netG $NETG_WAE
$BASE_COMMAND --outf ./results/mnist_waedec_ncenc4_lbd10_nquant2_paper_rmvquant --lbd 10 --ncenc 4 --nquant 2 --nresenc $NRES_WAE --netG $NETG_WAE



# Bu da normal

# encode
BASE_COMMAND=python3\ wae_comp.py\ --dataset\ mnist\ --dataroot\ ./data/\ --testroot\ ./data/\ --cuda\ --nz\ 8\ --sigmasqz\ 1.0\ --lr_eg\ 0.001\ --niter\ 55\ --decay_steps\ 30\ 50\ --decay_gamma\ 0.4\ --check_every\ 100\ --workers\ 6\ --recloss\ --mmd\ --bnz\ --batchSize\ 256\ --useenc\ --comp\ --freezedec\ --test_every\ 100\ --addsamples\ 10000\ --manualSeed\ 323\ --imageSize\ 32\ --wenc\ --rmvquant

## ... into WAE
NRES_WAE=2
NETG_WAE=./results/mnist_wae_nz8/netG_epoch_54.pth

$BASE_COMMAND --outf ./results/mnist_waedec_ncenc8_lbd10_nquant2_rmvquant --lbd 10 --ncenc 8 --nquant 2 --nresenc $NRES_WAE --netG $NETG_WAE
$BASE_COMMAND --outf ./results/mnist_waedec_ncenc4_lbd10_nquant2_rmvquant --lbd 10 --ncenc 4 --nquant 2 --nresenc $NRES_WAE --netG $NETG_WAE



























