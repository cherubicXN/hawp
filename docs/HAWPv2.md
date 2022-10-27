# Quickstart


<details>
<summary><b>Step 1: Download the official checkpoint from <a href="https://github.com/cherubicXN/hawp-torchhub/releases/tag/HAWPv2">torchub</a></b> by running the <a href="downloads.sh">downloads.sh</a>.
</summary>

```bash
wget https://github.com/cherubicXN/hawp-torchhub/releases/download/HAWPv2/hawpv2-edb9b23f.pth -P checkpoints
```
</details>

Step 1: 
# Training

```dotnetcli
HAWPv2 Training

positional arguments:
  config              path to config file

optional arguments:
  -h, --help          show this help message and exit
  --logdir LOGDIR
  --resume RESUME
  --clean
  --eval-off
  --seed SEED
  --tf32              toggle on the TF32 of pytorch
  --dtm {True,False}  toggle the deterministic option of CUDNN. This option will affect the replication of experiments

```