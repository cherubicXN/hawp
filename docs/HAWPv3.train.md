# Training Recipes of HAWPv3 

*HAWPv3 consists of multiple training phases including the synthetic training phase and the real data training.*

## Step 0: Synthetic learning

```
 python -m hawp.ssl.train \
    --datacfg hawp/ssl/config/synthetic_dataset.yaml \
    --modelcfg hawp/ssl/config/hawpv3.yaml \
    --base-lr 0.0004 \
    --epochs 10  \
    --batch-size 6 \
    --name hawpv3-round0
```

## Step 1: Homographic Adaptation for Pseudo Wireframe Generation

If you prefer single-image mode to minimize the usage of GPU memory footprint, please use the following command to obtain pseudo labels
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
    --min-score 0.75   --batch-size=16
```
*On my machine (NVIDIA A6000), the batch size of 16 will take 40G GPU memory in 43 minutes to generate the wireframe labels for 20k images of the training images in the Wireframe dataset (Huang et al., CVPR 2018)*

### Remarks
1. After the homographic adaptation step finished, three files will be generated and stored at the directory of ```data-ssl/{name}```, where ```{name}``` is the name for the last round of training. For example, if we train HAWPv3 with the name of ```hawpv3-round0```, the exported data will be saved at ```data-ssl/hawpv3-round0```.

2. For each generated wireframe and its auxiluary files, their names are started with ```{hash}-model-{epoch:05d}```.

3. In sum, the generated datacfg ``YAML`` file is located at ``data-ssl/{name}/{hash}-model-{epoch:05d}.yaml``.
## Step 2: Learning from Real-World images

- Once we have the pseudo wireframe labels, we can train HAWPv3 on the real-world images. An example usage is in the below command line:
    ```
    python -m hawp.ssl.train  --datacfg data-ssl/export_datasets/{name}/{hash}-model-00010.yaml --modelcfg hawp/ssl/config/hawpv3.yaml --base-lr 0.0004 --epochs 30  --name hawpv3-round1 --batch-size 6
    ```

- Then, we can run the homographic adaptation with new model checkpoints trained on real-world images, and then train/fine-tune a new model to further improve the repeatibility.
