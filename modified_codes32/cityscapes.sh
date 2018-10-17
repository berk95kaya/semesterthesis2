BASE_COMMAND=python\ wae_comp.py\ --dataset\ cityscapes\ --dataroot\ ./data/cityscapes/\ --cuda\ --nz\ 256\ --sigmasqz\ 1.0\ --avbtrick\ --niter\ 50\ --check_every\ 20\ --workers\ 8\ --imageSize\ 64
$BASE_COMMAND --outf ./results/city_wgan_64 --lbd 400 --wganloss --lr_eg 0.0001 --beta2 0.9 --lr_di 0.0001 --vis_every 50 --batchSize 64
