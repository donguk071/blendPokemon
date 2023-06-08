import argparse
import os
from PIL import Image

import torch
from torchvision import transforms

import models
from utils import make_coord
from test_liif import batched_predict


def run(img_path, model_path, output_path, resolution_width, resolution_height):
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', default= img_path)
    parser.add_argument('--model', default = model_path)
    parser.add_argument('--resolution', default = (resolution_height,resolution_width))
    parser.add_argument('--output', default=output_path)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    img = transforms.ToTensor()(Image.open(args.input).convert('RGB'))

    model = models.make(torch.load(args.model, map_location=torch.device('cpu'))['model'], load_sd=True)#.cuda()
    
    h, w = int(args.resolution[0]), int(args.resolution[1])
    #h, w = list(map(int, args.resolution.split(',')))
    coord = make_coord((h, w))#.cuda()
    cell = torch.ones_like(coord)
    cell[:, 0] *= 2 / h
    cell[:, 1] *= 2 / w
    pred = batched_predict(model, ((img - 0.5) / 0.5).unsqueeze(0),
        coord.unsqueeze(0), cell.unsqueeze(0), bsize=30000)[0]
    pred = (pred * 0.5 + 0.5).clamp(0, 1).view(h, w, 3).permute(2, 0, 1).cpu()
    transforms.ToPILImage()(pred).save(args.output)
