import os

import models
import torch
from PIL import Image
from test_liif import batched_predict
from torchvision import transforms
from utils import make_coord


def make_high_resolution(img_path, output_path, resolution_width, resolution_height):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    img = transforms.ToTensor()(Image.open(img_path).convert("RGB"))

    model = models.make(
        torch.load(f="../LIIF/edsr-baseline-liif.pth", map_location=torch.device("cpu"))["model"], load_sd=True
    )

    h, w = int((resolution_height, resolution_width)[0]), int((resolution_height, resolution_width)[1])

    coord = make_coord((h, w))
    cell = torch.ones_like(coord)
    cell[:, 0] *= 2 / h
    cell[:, 1] *= 2 / w
    pred = batched_predict(
        model=model,
        inp=((img - 0.5) / 0.5).unsqueeze(0),
        coord=coord.unsqueeze(0),
        cell=cell.unsqueeze(0),
        bsize=30000,
    )[0]
    pred = (pred * 0.5 + 0.5).clamp(0, 1).view(h, w, 3).permute(2, 0, 1).cpu()
    transforms.ToPILImage()(pred).save(output_path)
