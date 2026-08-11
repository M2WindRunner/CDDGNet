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
| LEVIR-CD → WHU | 80.53 | 67.40 | 98.24 | [Download](https://pan.baidu.com/s/13tZKbeyDa_V1vH2y_YIvLQ?pwd=qcsm) |
| LEVIR-CD+ → WHU | | | | [Download](https://pan.baidu.com/s/PLACEHOLDER_LEVIR+_to_WHU) |
| SYSU → CDD | | | | [Download](https://pan.baidu.com/s/PLACEHOLDER_SYSU_to_CDD) |
| CDD → SYSU | 68.79 | 52.43 | 84.67 | [Download](https://pan.baidu.com/s/1eOCuG3-8FY5qFHyOWVFc4A?pwd=f5hm) |
| CDD → GBF-CD | 49.07 | 32.51 | 93.35 | [Download](https://pan.baidu.com/s/1vJU-FZb32-D8gaw13YJsnw?pwd=hppj) |
| CDD → Yellow | | | | [Download](https://pan.baidu.com/s/PLACEHOLDER_CDD_to_yellow) |

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

4. A bug occurred while uploading the "untils" folder. Please rename the "until" folder in the repository to "utils" and then run the code.

5. Specific weights are currently being uploaded.












