""" © 2018, ETH Zurich """

import argparse
import numpy as np
import glob
import os
import scipy.misc
import scipy.ndimage
from skimage.measure import compare_psnr
from skimage.measure import compare_mse
from skimage.measure import compare_ssim
from ms_ssim_np import MultiScaleSSIM
import functools

# python compare_imgs.py ../results2/celeba_wganaedec_ncenc2_lbd600/test/test_rec/\*.png ../data/celeba_waecrop/resized_celebA_test/celebA/\*.jpg


make_batched = functools.partial(np.expand_dims, axis=0)


def calc_and_print_ssim_and_psnr(inp_img_ps, out_img_ps):
    assert len(inp_img_ps) == len(out_img_ps), \
        'Expected same number of input and output images, got {} and {}'.format(
                len(inp_img_ps), len(out_img_ps))
    assert len(inp_img_ps) > 0, 'No images'
    print('SSIM, MS-SSIM, PSNR')
    for inp_img, out_img in zip(inp_img_ps, out_img_ps):
        print(compare(inp_img, out_img, calc_ssim=True, calc_msssim=False, calc_psnr=False, calc_mse=False))

def calc_and_print_ssim_and_mse(inp_img_ps, out_img_ps):
    assert len(inp_img_ps) == len(out_img_ps), \
        'Expected same number of input and output images, got {} and {}'.format(
                len(inp_img_ps), len(out_img_ps))
    assert len(inp_img_ps) > 0, 'No images'
    print('mean SSIM, MS-SSIM, MSE')
    ssim, msssim, mse = 0, 0, 0
    for inp_img, out_img in zip(sorted(inp_img_ps), sorted(out_img_ps)):
        # print(inp_img)
        # print(out_img)
        cssim, cmsssim, _, cmse = compare(inp_img, out_img, calc_ssim=True, calc_msssim=True, calc_psnr=False, calc_mse=True)
        ssim += cssim; msssim += cmsssim; mse += cmse
    nimg = len(inp_img_ps)
    print(ssim/nimg, msssim/nimg, mse/nimg)



def _read_if_not_array(im):
    if not isinstance(im, np.ndarray):
        assert os.path.exists(im)
        return scipy.ndimage.imread(im)
    return im

def compare_msssim(inp_img_batched, out_img_batched):
    return MultiScaleSSIM(inp_img_batched, out_img_batched)


def compare(inp_img, out_img, calc_ssim=True, calc_msssim=True, calc_psnr=True, calc_mse=True):
    inp_img = _read_if_not_array(inp_img)
    out_img = _read_if_not_array(out_img)

    assert inp_img.shape == out_img.shape

    def get_ssim():
        return compare_ssim(inp_img, out_img, multichannel=True, gaussian_weights=True, sigma=1.5)

    def get_msssim():
        return MultiScaleSSIM(make_batched(inp_img), make_batched(out_img))

    def get_psnr():
        return compare_psnr(inp_img, out_img)

    def get_mse():
        return compare_mse(inp_img, out_img)

    def _run_if(cond, fn):
        return fn() if cond else None

    return _run_if(calc_ssim, get_ssim), _run_if(calc_msssim, get_msssim), _run_if(calc_psnr, get_psnr), _run_if(calc_mse, get_mse)


# speed test
# using batched input: 2.36s
# per image: 1.54s
def _speedtest(flags):
    import time
    inp_img_ps = sorted(glob.glob(flags.inp_glob))[:30]
    out_img_ps = sorted(glob.glob(flags.out_glob))[:30]
    assert len(inp_img_ps) == len(out_img_ps)

    inp = np.stack([_read_if_not_array(ip) for ip in inp_img_ps], 0)
    out = np.stack([_read_if_not_array(ip) for ip in out_img_ps], 0)

    m = []
    t = []
    for n in range(inp.shape[0]):
        s = time.time()
        v = compare_msssim(make_batched(inp[n, ...]), make_batched(out[n, ...]))
        t.append(time.time() - s)
        m.append(v)
    print(np.mean(m))
    print(np.sum(t))

    s = time.time()
    print(compare_msssim(inp, out))
    print(time.time() - s)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('inp_glob')
    p.add_argument('out_glob')
    flags = p.parse_args()
    # calc_and_print_ssim_and_psnr(glob.glob(flags.inp_glob), glob.glob(flags.out_glob))
    calc_and_print_ssim_and_mse(glob.glob(flags.inp_glob), glob.glob(flags.out_glob))
