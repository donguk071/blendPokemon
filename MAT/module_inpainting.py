from generate_image import generate_images 
import os

network_pkl = './MAT/pretrained/Places_512.pkl'
dpath = './MAT/test_sets/CelebA-HQ/images'
mpath = './MAT/test_sets/CelebA-HQ/masks'
outdir = './MAT/mytest'
# 함수 호출
generate_images(
    network_pkl=network_pkl,
    resolution =512,
    truncation_psi =1,
    noise_mode = "const",
    dpath=dpath,
    mpath=mpath,
    outdir=outdir
)
# generate_images(
#     network_pkl='C:/Users/drago/university/23.1 semester/DL/inpainting/MAT/pretrained/Places_512_FullData.pkl',
#     dpath='test_sets\CelebA-HQ\images',
#     mpath = 'test_sets\CelebA-HQ\masks',
#     outdir= "./mytest"
# )
#python generate_image.py --network C:/Users/drago/university/23.1 semester/DL/inpainting/MAT/pretrained/Places_512_FullData.pkl --dpath C:/Users/drago/university/23.1 semester/DL/inpainting/MAT/mytest/pockmontestimg2.jpg --outdir ./mytest