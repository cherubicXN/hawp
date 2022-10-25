python -m evaluation.draw-json \
    --pred precomputed-results/benchmark/LETR-R101-york.json \
    --benchmark york \
    --topk 50 \
    --dest precomputed-results/vis-fsl-york/LETR-R101

python -m evaluation.draw-json \
    --pred outputs/ihawp-train-rot-v2-full/220625-162909/york_test.json \
    --benchmark york \
    --threshold 0.9 \
    --dest precomputed-results/vis-fsl-york/HAWPv2


python -m evaluation.draw-json \
    --pred precomputed-results/benchmark/HAWPv1-york.json  \
    --benchmark york \
    --threshold 0.97 \
    --dest precomputed-results/vis-fsl-york/HAWPv1

python -m evaluation.draw-json \
    --pred precomputed-results/benchmark/afmpp-york.json \
    --benchmark york \
    --threshold=0.2 \
    --cmp=l \
    --dest precomputed-results/vis-fsl-york/afmpp

python -m evaluation.draw-json \
    --pred precomputed-results/benchmark/FClip-HG2-LB-york.json \
    --benchmark york \
    --topk 100 \
    --dest precomputed-results/vis-fsl-york/FClip


