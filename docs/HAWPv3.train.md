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

## Homographic Adaptation

```
python -m hawp.ssl.homoadp --metarch HAWP-heatmap \
    --datacfg hawp/ssl/config/export/wireframe-10iters.yaml \
    --workdir exp-ssl/hawpv3-round0 \
    --epoch 10 \
    --modelcfg exp-ssl/hawpv3-round0/model.yaml \
    --min_score 0.75 

```
## Realdata learning