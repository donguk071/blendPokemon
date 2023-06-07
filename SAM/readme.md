# Segment Anything

## Installation

The code requires `python>=3.8`, as well as `pytorch>=1.7` and `torchvision>=0.8`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.

Install Segment Anything:

```
pip install git+https://github.com/facebookresearch/segment-anything.git
```

The following optional dependencies are necessary for mask post-processing.

```
pip install opencv-python
```

## <a name="GettingStarted"></a>Getting Started

First download a [model checkpoint](#model-checkpoints) and put them in `SAM/`. Then masks can be generated for images from the command line:

```bash
cd SAM
python generate_mask.py --input samples --output samples/outputs --model-type vit_h --checkpoint sam_vit_h_4b8939.pth
```

After this, a window will open. Select the desired points with the left mouse click. Once the desired mask has been created, you can save the file by pressing 's'.

## <a name="Models"></a>Model Checkpoints

Three model versions of the model are available with different backbone sizes. These models can be instantiated by running

Click the links below to download the checkpoint for the corresponding model type.

- **`default` or `vit_h`: [ViT-H SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)**
- `vit_l`: [ViT-L SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth)
- `vit_b`: [ViT-B SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)
