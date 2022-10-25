python -m evaluation.draw-json \
    --pred precomputed-results/benchmark/LETR-R101-wireframe.json \
    --benchmark wireframe \
    --topk 50 \
    --dest precomputed-results/vis-fsl/LETR-R101

python -m evaluation.draw-json \
    --pred outputs/ihawp-train-rot-v2-full/220625-162909/wireframe_test.json \
    --benchmark wireframe \
    --threshold 0.97 \
    # --topk 50 \
    --dest precomputed-results/vis-fsl/HAWPv2

python -m evaluation.draw-json \
    --pred precomputed-results/benchmark/HAWPv1-wireframe.json \
    --benchmark wireframe \
    --threshold 0.97 \
    --dest precomputed-results/vis-fsl/HAWPv2


python -m evaluation.draw-json \
    --pred precomputed-results/benchmark/afmpp-wireframe.json \
    --benchmark wireframe \
    --threshold=0.2 \
    --cmp=l \
    --dest precomputed-results/vis-fsl/afmpp

python -m evaluation.draw-json \
    --pred precomputed-results/benchmark/FClip-HG2-LB-wireframe.json \
    --benchmark wireframe \
    --topk 50 \
    --dest precomputed-results/vis-fsl/FClip


