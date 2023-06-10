from generate_image import generate_images 
import os


def generate_images_ours(network_pkl = '../MAT/pretrained/Places_512.pkl'
                         , dpath = '../MAT/test_sets/images'
                         ,mpath = '../MAT/test_sets/masks'
                         ,outdir = '../MAT/mytest'):
    
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
    
# generate_images_ours()
