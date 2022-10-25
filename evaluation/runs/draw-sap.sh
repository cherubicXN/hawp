# python -m evaluation.draw-sap --path \
#     outputs/ihawp-train-rot-v2-full/220625-162909/wireframe_test.json.sap \
#     precomputed-results/benchmark/HAWPv1-wireframe.json.sap  \
#     precomputed-results/benchmark/FClip-HG2-LB-wireframe.json.sap \
#     precomputed-results/benchmark/LETR-R101-wireframe.json.sap \
#     precomputed-results/benchmark/LETR-R50-wireframe.json.sap \
#     precomputed-results/benchmark/LCNN-wireframe.json.sap \
#     --threshold=10 \
#     --dest figures/wireframe-sAP-10.pdf

python -m evaluation.draw-sap --path \
    outputs/ihawp-train-rot-v2-full/220625-162909/wireframe_test.json.sap \
    precomputed-results/benchmark/HAWPv1-wireframe.json.sap  \
    precomputed-results/benchmark/FClip-HG2-LB-wireframe.json.sap \
    precomputed-results/benchmark/LETR-R101-wireframe.json.sap \
    precomputed-results/benchmark/LETR-R50-wireframe.json.sap \
    precomputed-results/benchmark/LCNN-wireframe.json.sap \
    --threshold=5 \
    --dest figures/wireframe-sAP-05.pdf
# outputs/ihawp-train-rot-v2-full/220625-162909/wireframe_test.json.sap \
    #precomputed-results/benchmark/HAWPv1-wireframe.json.sap  \

python -m evaluation.draw-sap --path \
    precomputed-results/benchmark/LETR-R101-york.json.sap \
    precomputed-results/benchmark/LETR-R50-york.json.sap \
    precomputed-results/benchmark/FClip-HG2-LB-york.json.sap \
    precomputed-results/benchmark/HAWPv1-york.json.sap  \
    outputs/ihawp-train-rot-v2-full/220625-162909/york_test.json.sap \
    --threshold=5 \
    --dest figures/york-sAP-05.pdf
    # precomputed-results/benchmark/LCNN-york.json.sap \