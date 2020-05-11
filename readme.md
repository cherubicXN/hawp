# Holistically-Attracted Wireframe Parsing (CVPR 2020)

This is the offical implementation for our [CVPR paper](https://arxiv.org/pdf/2003.01663).

## Highlights
- We propose **a fast and parsimonious parsing method HAWP** to accurately and robustly detect a vectorized wireframe in an input image with a single forward pass. 
- The proposed HAWP is **fully end-to-end**.
- The proposed HAWP **does not require squeeze module**.
- **State-of-the-art performance** on the Wireframe dataset and YorkUrban dataset.
- The proposed HAWP achievs **29.5 FPS** on a GPU (Tesla V100) for 1-batch inference.

<p align="center">
<img src="figures/teaser.png" height="400" >
<p>

## Quantitative Results


### Precison-Recall curves for structural correctness
<p align="center">
<img src="figures/sAP10-wireframe.png" height="200" >
<img src="figures/sAP10-york.png" height="200" >
<p>

### Precison-Recall curves for heatmap-based correctness
<p align="center">
<img src="figures/APH-wireframe.png" height="200" >
<img src="figures/APH-york.png" height="200" >
<p>




## Installation

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

## Acknoledgement
We acknowledge the effort from the authors of the Wireframe dataset and the YorkUrban dataset. These datasets make accurate line segment detection and wireframe parsing possible.