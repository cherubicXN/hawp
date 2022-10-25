FNAME='00031811.png'
FNAME='00110785.png'
# FNAME='00255368.png'
# FNAME='00031546.png'
# FNAME='00031811.png'
# FNAME='00034439.png'
FNAME='00053549.png'

python -m evaluation.draw-json \
    --pred precomputed-results/benchmark/HAWPv1-wireframe.json \
    --benchmark wireframe \
    --threshold 0.97 \
    --fname $FNAME \
    --dest precomputed-results/vis-fsl-sel/HAWPv1

python -m evaluation.draw-json \
    --pred precomputed-results/benchmark/LETR-R101-wireframe.json \
    --benchmark wireframe \
    --topk 50 \
    --fname $FNAME \
    --dest precomputed-results/vis-fsl-sel/LETR-R101

python -m evaluation.draw-json \
    --pred outputs/ihawp-train-rot-v2-full/220625-162909/wireframe_test.json \
    --benchmark wireframe \
    --threshold 0.9 \
    --fname $FNAME \
    --dest precomputed-results/vis-fsl-sel/HAWPv2


python -m evaluation.draw-json \
    --pred precomputed-results/benchmark/afmpp-wireframe.json \
    --benchmark wireframe \
    --threshold=0.2 \
    --fname $FNAME \
    --cmp=l \
    --dest precomputed-results/vis-fsl-sel/afmpp

python -m evaluation.draw-json \
    --pred precomputed-results/benchmark/FClip-HG2-LB-wireframe.json \
    --benchmark wireframe \
    --fname $FNAME \
    --topk 50 \
    --dest precomputed-results/vis-fsl-sel/FClip


