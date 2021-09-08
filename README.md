# FuseFormer: Fusing Fine-Grained Information in Transformers for Video Inpainting

By [Rui Liu](https://ruiliu-ai.github.io), Hanming Deng, Yangyi Huang, Xiaoyu Shi, [Lewei Lu](https://dblp.org/pid/247/6438.html), [Wenxiu Sun](http://wenxiusun.com/), [Xiaogang Wang](https://www.ee.cuhk.edu.hk/~xgwang/), [Jifeng Dai](https://jifengdai.org/), [Hongsheng Li](https://www.ee.cuhk.edu.hk/~hsli/). 

This repo is the official Pytorch implementation of [FuseFormer: Fusing Fine-Grained Information in Transformers for Video Inpainting](https://arxiv.org/abs/2109.02974).

## Introduction
<img src='imgs/pipeline.png' width="900px">

## Usage

### Prerequisites
- Python >= 3.6
- Pytorch >= 1.0 and corresponding torchvision (https://pytorch.org/)

### Install
- Clone this repo:
```
git clone https://github.com/ruiliu-ai/FuseFormer.git
```
- Install other packages:
```
cd FuseFormer
pip install -r requirements.txt
```

## Training

### Dataset preparation
Download datasets ([YouTube-VOS](https://competitions.codalab.org/competitions/19544) and [DAVIS](https://davischallenge.org/davis2017/code.html)) into the data folder.
```
mkdir data
```

### Training script
```
python train.py -c configs/youtube-vos.json
```

## Test
Download [pre-trained model](https://drive.google.com/file/d/1BuSE42QAAUoQAJawbr5mMRXcqRRKeELc/view?usp=sharing) into checkpoints folder.
```
mkdir checkpoints
```

### Test script
```
python test.py -c checkpoints/fuseformer.pth -v data/DAVIS/JPEGImages/blackswan -m data/DAVIS/Annotations/blackswan
```

## Citing FuseFormer
If you find FuseFormer useful in your research, please consider citing:
```
@InProceedings{Liu_2021_FuseFormer,
  title={FuseFormer: Fusing Fine-Grained Information in Transformers for Video Inpainting},
  author={Liu, Rui and Deng, Hanming and Huang, Yangyi and Shi, Xiaoyu and Lu, Lewei and Sun, Wenxiu and Wang, Xiaogang and Dai, Jifeng and Li, Hongsheng},
  booktitle = {International Conference on Computer Vision (ICCV)},
  year={2021}
}
```

## Acknowledement
This code borrows heavily from the video inpainting framework [spatial-temporal transformer net](https://github.com/researchmm/STTN). 
