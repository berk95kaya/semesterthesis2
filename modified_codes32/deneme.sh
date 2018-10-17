### CELEB A
## Train WAE
python3 wae_comp.py --dataset mnist --dataroot ./data/ --testroot ./data/ --cuda --nz 128 --sigmasqz 1.0 --lr_eg 0.001 --niter 55 --decay_steps 30 50 --decay_gamma 0.4 --check_every 100 --workers 8 --recloss --mmd --bnz --outf ./output_files/x --lbd 100 --batchSize 256 --detenc --useenc --test_every 20 --addsamples 10000 --manualSeed 321

