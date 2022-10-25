# HAWPv3: Learning Wireframes via Self-Supervised Learning

|Model Name|Checkpoint|MD5|
|---|---|---|
|imagenet-small| [url](https://hawp-models.s3.us-east-2.amazonaws.com/hawpv3-models/imagenet-train-small.zip)   | 03a8400e9474320f2b42973d1ba19487|


## Quickstart
### 1. Download Checkpoints
- Go to the project directory of HAWP (e.g., in my own machine, it is ``/home/xn/repo/hawp-private
``)
- Create a directory to save the checkpoints (e.g., in my own machine, I created a folder named ``hawpv3-models``. The absolute directory should be ``/home/xn/repo/hawp-private/hawpv3-models``)
- Download the checkpoint from the url listed in the top table. For example, you can run the following command lines to download the model [imagenet-small](https://hawp-models.s3.us-east-2.amazonaws.com/hawpv3-models/imagenet-train-small.zip):

```shell
#E.g., Working dir is /home/xn/repo/hawp-private
cd hawpv3-models # or your favorite directory
wget https://hawp-models.s3.us-east-2.amazonaws.com/hawpv3-models/imagenet-train-small.zip 
unzip imagenet-train-small.zip
cd ..
```

### 2. Inference on your own images
```python
python -m sslib.predict --modelcfg hawpv3-models/imagenet-train-small/model.yaml \
    --ckpt hawpv3-models/imagenet-train-small/model-final.pth \
    --img {filename.png} -t=0.5
```

