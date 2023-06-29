# HAWPv2:  Learning Wireframes via Fully-Supervised Learning

*The codes of HAWPv2 are placed in the directory of [hawp/fsl](../hawp/fsl).*
## Quickstart & Evaluation
- Please download the dataset and checkpoints as in [readme.md](../readme.md).
- Run the following command line(s) to evaluate the offical model on the Wireframe dataset and YorkUrban dataset by
  
  <details>
  <summary>Evaluation on the Wireframe dataset.</summary>

  ```bash
  python -m hawp.fsl.benchmark configs/hawpv2.yaml \
    --ckpt checkpoints/hawpv2-edb9b23f.pth \
    --dataset wireframe
  ``` 
  </details>
  
  <details>
  <summary>Evaluation on the YorkUrban dataset.</summary>

  ```bash
  python -m hawp.fsl.benchmark configs/hawpv2.yaml \
    --ckpt checkpoints/hawpv2-edb9b23f.pth \
    --dataset wireframe
  ``` 
  </details>

## Evaluation Results

|Dataset|sAP-5|sAP-10|sAP-15|command line|comment|
|--|--|--|--|--|--|
|Wireframe| 65.8 | 69.8 |71.4|``python -m hawp.fsl.benchmark configs/hawpv2.yaml --ckpt checkpoints/hawpv2-edb9b23f.pth --dataset wireframe --jhm=0.001``|jhm = 0.001|
|Wireframe| 65.7 | 69.8 |71.4|``python -m hawp.fsl.benchmark configs/hawpv2.yaml --ckpt checkpoints/hawpv2-edb9b23f.pth --dataset wireframe --jhm=0.005``|jhm = 0.005|
|Wireframe| 65.7 | 69.7 |71.3|``python -m hawp.fsl.benchmark configs/hawpv2.yaml --ckpt checkpoints/hawpv2-edb9b23f.pth --dataset wireframe --jhm=0.008``|jhm = 0.008 (default setting)|
|YorkUrban|29.0|31.4|32.8|``python -m hawp.fsl.benchmark configs/hawpv2.yaml --ckpt checkpoints/hawpv2-edb9b23f.pth --dataset york --jhm=0.005``|jhm = 0.001|jhm=0.001
|YorkUrban|28.9|31.4|32.7|``python -m hawp.fsl.benchmark configs/hawpv2.yaml --ckpt checkpoints/hawpv2-edb9b23f.pth --dataset york --jhm=0.005``|jhm = 0.005|jhm = 0.005
|YorkUrban|28.8|31.3|32.6|``python -m hawp.fsl.benchmark configs/hawpv2.yaml --ckpt checkpoints/hawpv2-edb9b23f.pth --dataset york --jhm=0.005``|jhm = 0.008|jhm = 0.008 (default setting)

# Training
- Run the following command line to train the HAWPv2 on the Wireframe dataset.
  ```
  python -m hawp.fsl.train configs/hawpv2.yaml --logdir outputs
  ```

- The usage of [hawp.fsl.train](hawp/fsl/../../../hawp/fsl/train.py) is as follow:
  ```dotnetcli
  HAWPv2 Training

  positional arguments:
    config              path to config file

  optional arguments:
    -h, --help          show this help message and exit
    --logdir LOGDIR
    --resume RESUME
    --clean
    --seed SEED
    --tf32              toggle on the TF32 of pytorch
    --dtm {True,False}  toggle the deterministic option of CUDNN. This option will affect the replication of experiments

  ```