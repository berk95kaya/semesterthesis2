import os
import matplotlib.pyplot as plt
from scipy.misc import imresize, toimage

# root path depends on your computer
indir = '../data/celeba_waecrop/resized_celebA_test/celebA/'
outdir = '../bpg_baseline/celeba_waecrop_test_png'

img_list = os.listdir(indir)

# ten_percent = len(img_list) // 10

for i in range(len(img_list)):
    img = plt.imread(indir + img_list[i])
    toimage(img).save(outdir + img_list[i].replace('.jpg', '.png'))

    if (i % 1000) == 0:
        print('%d images complete' % i)
