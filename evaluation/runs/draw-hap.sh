python -m evaluation.draw-hap --path \
    precomputed-results/benchmark/DWP-wireframe.mat \
    precomputed-results/benchmark/LCNN-wireframe.mat \
    precomputed-results/benchmark/LETR-R101-wireframe-aph.mat \
    precomputed-results/benchmark/LETR-R50-wireframe-aph.mat \
    precomputed-results/benchmark/FClip-HG2-LB-wireframe-aph.mat \
    precomputed-results/benchmark/afmpp-wireframe-aph.mat \
    precomputed-results/benchmark/HAWPv1-wireframe.mat \
    outputs/ihawp-train-rot-v2-full/220625-162909/wireframe_test-aph.mat \
    --dest figures/APH-wireframe.pdf


python -m evaluation.draw-hap --path \
    precomputed-results/benchmark/DWP-york.mat \
    precomputed-results/benchmark/LCNN-york.mat \
    precomputed-results/benchmark/LETR-R101-york-aph.mat \
    precomputed-results/benchmark/LETR-R50-york-aph.mat \
    precomputed-results/benchmark/FClip-HG2-LB-york-aph.mat \
    precomputed-results/benchmark/afmpp-york-aph.mat \
    precomputed-results/benchmark/HAWPv1-york.mat \
    outputs/ihawp-train-rot-v2-full/220625-162909/york_test-aph.mat \
    --dest figures/APH-york.pdf
    # precomputed-results/benchmark/DWP-wireframe.mat \