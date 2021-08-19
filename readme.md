# HAWP (Experimental branch for inference-only usage)

[**News**] We are refactorizing our HAWP as a python package for inference. 

## Highlights
- We propose **a fast and parsimonious parsing method HAWP** to accurately and robustly detect a vectorized wireframe in an input image with a single forward pass.
- The proposed HAWP is **fully end-to-end**.
- The proposed HAWP **does not require the squeeze module**.
- **State-of-the-art performance** on the Wireframe dataset and YorkUrban dataset.
- The proposed HAWP achieves **29.5 FPS** on a GPU (Tesla V100) for 1-batch inference.

<p align="center">
<img src="figures/teaser.png" height="400" >
<p>

## Installation
1. Install PyTorch from https://pytorch.org/get-started/locally/
2. Install hawp to your python environment by
```
pip install git+https://github.com/cherubicXN/hawp.git@inference
```

## Citations
If you find our work useful in your research, please consider citing:
```
@inproceedings{HAWP,
title = "Holistically-Attracted Wireframe Parsing",
author = "Nan Xue and Tianfu Wu and Song Bai and Fu-Dong Wang and Gui-Song Xia and Liangpei Zhang and Philip H.S. Torr
",
booktitle = "IEEE Conference on Computer Vision and Pattern Recognition (CVPR)",
year = {2020},
}
```

## Acknowledgment
- We acknowledge the effort from the authors of the Wireframe dataset and the YorkUrban dataset. These datasets make accurate line segment detection and wireframe parsing possible.

- This branch is partially based on [openpifpaf](https://github.com/openpifpaf/openpifpaf).
