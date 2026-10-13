# Causal Decoupling Domain Generalization for Remote Sensing Change Detection

This repository provides the official training and evaluation code for 《Causal Decoupling Domain Generalization for Remote Sensing Change Detection》.

## Dependencies

```
torch >= 2.0.0
torchvision >= 0.15.0
numpy
opencv-python
pillow
pywavelets
dropblock
tqdm
matplotlib
seaborn
```
## Pre-trained Checkpoints

We provide the checkpoints of all tasks for cross-domain change detection. 

 Source → Target | F1 | mIoU | OA | Baidu Netdisk |
|---|---|---|---|---|
| LEVIR-CD → WHU | 79.72 | 66.27 | 98.21 | [Download](https://pan.baidu.com/s/1GR7Jcpx7oGm9siv0ss74og?pwd=6nbm) |
| LEVIR-CD+ → WHU | 83.05 | 71.01 | 98.46 | [Download](https://pan.baidu.com/s/1VlJhdGAjIehozaIAn_ESvg?pwd=387j) |
| SYSU → CDD | 60.78 | 43.64 | 89.38 | [Download](https://pan.baidu.com/s/1Mob4NEn76oEUpEjehoe8Sw?pwd=r4nb) |
| CDD → SYSU | 67.27 | 50.68 | 84.71 | [Download](https://pan.baidu.com/s/1n4FIilh4CRWrEofFSArhbA?pwd=rjyt) |
| CDD → GBF-CD | 47.29 | 30.96 | 92.56 | [Download](https://pan.baidu.com/s/1HwcH67ov9haV05CEn0TQWg?pwd=k47c) |
| CDD → Yellow | 67.27 | 50.68 | 95.10 | [Download](https://pan.baidu.com/s/1rCLEd-Uni7QzztswE_aKcA?pwd=vrgk) |

## Training

```bash
python train.py
```

## Evaluation

```bash
python eval.py
```

## Important Notes

1. Since the method described in this paper relies on training small models, the training results tend to be unstable. To ensure stable training, it is recommended to use appropriately constructed pre-trained weights. Preliminary pre-training can be performed following the approach outlined in the paper 《SeaMo: A Multi-Seasonal and Multimodal Remote Sensing Foundation Model》. The weights provided above are the results of this pre-training.

2. The weights in the repository are the result of retraining following code restructuring; therefore, slight differences from the results reported in the paper are to be expected.

3. Please determine the optimal settings for the series of hyperparameters in the code based on your specific use case and training environment.


## Citation

If you are using the code/model/data provided here in a publication, please consider citing our works:

```bash

@inproceedings{aaai2026CDDGNet,
  title={Causal Decoupling Domain Generalization for Remote Sensing Change Detection},
  author={Zhao, Jiaqi and Xie, Jianpeng and Zhou, Yong and Du, Wen-Liang and Zhu, Hancheng and Yao, Rui},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={15},
  pages={13135--13143},
  year={2026}
}

```









