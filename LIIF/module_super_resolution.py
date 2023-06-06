# -*- coding: utf-8 -*-
import torch
import os
from PIL import Image
import numpy as np
import demo

# for test
img1_path = 'LIIF/images/image1.jpg'
img2_path = 'LIIF/images/image2.jpg'
output_path = 'LIIF/outputs/output1.jpg'

def make_high_resolution(img_path, resolution_width, resolution_height):
    demo.run(img_path = img_path, output_path = output_path, model_path = 'LIIF/edsr-baseline-liif.pth', resolution_width = resolution_width, resolution_height = resolution_height)

make_high_resolution(img_path = img2_path, resolution_width = 300, resolution_height = 600)