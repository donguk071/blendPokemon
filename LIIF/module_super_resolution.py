import os
import models
import torch
from PIL import Image
from test_liif import batched_predict
from torchvision import transforms
from utils import make_coord


def make_high_resolution(img_path, output_path, resolution_width, resolution_height):
    '''
    img_path(str): 입력 이미지 경로(ex: 'input.jpg', or 'input.png')
    output_path(str): super resolution 진행 한 결과물 저장 경로(ex: 'output.jpg' or 'output.png')
    resolution_width(int): super resolution 적용 후 결과물의 width
    resolution_height(int): super resolution 적용 후 결과물의 height(int)
    '''
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # 입력 이미지 불러오기
    img = transforms.ToTensor()(Image.open(img_path).convert("RGB"))

    # LIIF 폴더 내에 저장된 EDSR-baseline-liif.pth 모델 불러오기
    model = models.make(
        torch.load(f="../LIIF/edsr-baseline-liif.pth", map_location=torch.device("cpu"))["model"], load_sd=True
    )
    
    h, w = int((resolution_height, resolution_width)[0]), int((resolution_height, resolution_width)[1])
    
    '''
    LIIF/utils.py의 make_coord(shape, ranges=None, flatten=True)
    주어진 형태에 맞는 grid에서 각 grid cell의 중심좌표를 생성하는 함수

    shape: grid shape
    ranges: grid의 height, width
    flatten(True or False): True 설정 시 output(각 grid cell의 중심좌표)의 flatten값 return

    '''

    # super resolution 적용 한 후 나오게 될 이미지의 height, width에 맞는 grid에서 각 grid cell의 중심좌표 계산
    coord = make_coord((h, w))
    
    # 각 grid cell의 height와 width
    cell = torch.ones_like(coord)
    cell[:, 0] *= 2 / h
    cell[:, 1] *= 2 / w

    # LIIF/test_liif.py의 batched_predict 함수를 통해 입력 이미지의 super resolution 결과 predict

    '''
    LIIF/test_liif.py의 batched_predict(model, inp, coord, cell, bsize)

    model: LIIF 모델
    inp: 입력 이미지
    coord: 결과물 이미지의 height, width에 맞는 grid에서 각 grid cell의 중심좌표
    cell: 각 grid cell의 height와 width
    bsize: 한번의 predict 과정에서 예측할 grid cell의 개수(grid cell의 batch size)
    '''
    pred = batched_predict(
        model=model,
        inp=((img - 0.5) / 0.5).unsqueeze(0),
        coord=coord.unsqueeze(0),
        cell=cell.unsqueeze(0),
        bsize=30000,
    )[0]

    # super resoluion 결과 이미지 저장
    pred = (pred * 0.5 + 0.5).clamp(0, 1).view(h, w, 3).permute(2, 0, 1).cpu()
    transforms.ToPILImage()(pred).save(output_path)
