#LETR-R50
python -m evaluation.eval-sap --pred precomputed-results/benchmark/LETR-R50-wireframe.json --benchmark wireframe --label LETR-R50
python -m evaluation.eval-sap --pred precomputed-results/benchmark/LETR-R50-york.json --benchmark york --label LETR-R50
#LETR-R101
python -m evaluation.eval-sap --pred precomputed-results/benchmark/LETR-R101-wireframe.json --benchmark wireframe --label LETR-R101
python -m evaluation.eval-sap --pred precomputed-results/benchmark/LETR-R101-york.json --benchmark york --label LETR-R101


#HAWPv2
python -m evaluation.eval-sap --pred outputs/ihawp-train-rot-v2-full/220625-162909/wireframe_test.json --benchmark wireframe --label HAWPv2
python -m evaluation.eval-sap --pred outputs/ihawp-train-rot-v2-full/220625-162909/york_test.json --benchmark york --label HAWPv2
