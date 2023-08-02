# Training Recipes of HAWPv3 

*HAWPv3 consists of multiple training phases including the synthetic training phase and the real data training.*

## Synthetic learning

```
 python -m hawp.ssl.train \
    --datacfg hawp/ssl/config/synthetic_dataset.yaml \
    --modelcfg hawp/ssl/config/hawpv3.yaml \
    --base-lr 0.0004 \
    --epochs 10  \
    --batch-size 6 \
    --name hawpv3-round0
```

## Homographic Adaptation for Pseudo Wireframe Generation

If you prefer single-image mode to minimize the usage of GPU memory, please use the following command to obtain pseudo labels
```
python -m hawp.ssl.homoadp --metarch HAWP-heatmap \
    --datacfg hawp/ssl/config/export/wireframe-10iters.yaml \
    --workdir exp-ssl/hawpv3-round0 \
    --epoch 10 \
    --modelcfg exp-ssl/hawpv3-round0/model.yaml \
    --min_score 0.75 

```

For the batch mode, please use the following command 
```
python -m hawp.ssl.homoadp-bm --metarch HAWP-heatmap \
    --datacfg hawp/ssl/config/exports/wireframe-10iters.yaml \
    --workdir exp-ssl/hawpv3-round0 \
    --epoch 10 \
    --modelcfg exp-ssl/hawpv3-round0/model.yaml \
    --min-score 0.5   --batch-size=16
```
*On my machine (NVIDIA A6000), the batch size of 16 will take 40G GPU memory in 43 minutes to generate the wireframe labels for 20k images of the training images in the Wireframe dataset (Huang et al., CVPR 2018)*

## Learning from Real-World images
