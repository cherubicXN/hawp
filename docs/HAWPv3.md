# HAWPv3: Learning Wireframes via Self-Supervised Learning

*The codes of HAWPv3 are placed in the directory of [hawp/ssl](../hawp/ssl).*

|Model Name|Comments|MD5|
|---|---|---|
|[hawpv3-fdc5487a.pth](https://github.com/cherubicXN/hawp-torchhub/releases/download/HAWPv3/hawpv3-fdc5487a.pth)| Trained on the images of Wireframe dataset | fdc5487a43e3d42f6b2addf79d8b930d
|[hawpv3-imagenet-03a84.pth](https://github.com/cherubicXN/hawp-torchhub/releases/download/HAWPv3/hawpv3-imagenet-03a84.pth)| Trained on 100k images of ImageNet dataset| 03a8400e9474320f2b42973d1ba19487|

### Inference on your own images

- Run the following command line to obtain wireframes from HAWPv3 model
    <details>
        <summary><b>hawpv3-fdc5487a.pth</b></summary>
        
        python -m hawp.ssl.predict --ckpt checkpoints/hawpv3-fdc5487a.pth \
            --threshold 0.05 \
            --img {filename.png}
    </details>

    <details>
        <summary><b>hawpv3-imagenet-03a84.pth</b></summary>

        python -m hawp.ssl.predict --ckpt checkpoints/hawpv3-imagenet-03a84.pth \
            --threshold 0.05 \
            --img {filename.png}
    </details>

- A running example on the DTU-24 images
  ```bash
  python -m hawp.ssl.predict --ckpt checkpoints/hawpv3-imagenet-03a84.pth  \
    --threshold 0.05  \
    --img ~/datasets/DTU/scan24/image/*.png \ 
    --saveto docs/figures/dtu-24 --ext png \
  ```
    <p align="center">
    <!-- <img src="figures/teaser.png" height="400" >
    -->
    <img src="figures/dtu-24/000000.png" width="200">
    <img src="figures/dtu-24/000001.png" width="200">
    <img src="figures/dtu-24/000002.png" width="200">
    <img src="figures/dtu-24/000003.png" width="200">
    <img src="figures/dtu-24/000004.png" width="200">
    <img src="figures/dtu-24/000005.png" width="200">
    <img src="figures/dtu-24/000009.png" width="200">
    <img src="figures/dtu-24/000015.png" width="200">
    </p>
   
## Training

```bash

python -m hawp.ssl.train --help
usage: train.py [-h] --datacfg DATACFG --modelcfg MODELCFG --name NAME
                [--pretrained PRETRAINED] [--overwrite] [--tf32]
                [--dtm {True,False}] [--batch-size BATCH_SIZE]
                [--num-workers NUM_WORKERS] [--base-lr BASE_LR]
                [--steps STEPS [STEPS ...]] [--gamma GAMMA] [--epochs EPOCHS]
                [--seed SEED] [--iterations ITERATIONS]

optional arguments:
  -h, --help            show this help message and exit
  --datacfg DATACFG     filepath of the data config
  --modelcfg MODELCFG   filepath of the model config
  --name NAME           the name of experiment
  --pretrained PRETRAINED
                        the pretrained model
  --overwrite           [Caution!] the option to overwrite an existed
                        experiment
  --tf32                toggle on the TF32 of pytorch
  --dtm {True,False}    toggle the deterministic option of CUDNN. This option
                        will affect the replication of experiments

training recipe:
  --batch-size BATCH_SIZE
                        the batch size of training
  --num-workers NUM_WORKERS
                        the number of workers for training
  --base-lr BASE_LR     the initial learning rate
  --steps STEPS [STEPS ...]
                        the steps of the scheduler
  --gamma GAMMA         the lr decay factor
  --epochs EPOCHS       the number of epochs for training
  --seed SEED           the random seed for training
  --iterations ITERATIONS
                        the number of training iterations

```