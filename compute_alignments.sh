# Save stats of training data

# Exit in case of errors
set -e

# Paths (adjust if needed)
TRAINCFG=/home/bracke/code/transnormer/models/models_2024-03-21/training_config.toml
OUT1=/home/bracke/code/transnormer/hidden/statistics/typestats-orig2norm-dtak-v03-1600-1699-750k-maxlen512byt5
OUT2=/home/bracke/code/transnormer/hidden/statistics/typestats-norm2orig-dtak-v03-1600-1699-750k-maxlen512byt5

# Prepare conda environment
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate gpu-venv-transnormer
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
export CUDA_VISIBLE_DEVICES=1

# Create generations based on ./test_config.toml
python3 src/transnormer/evaluation/dataset_stats.py -c $TRAINCFG -o $OUT1 --base-layer orig
python3 src/transnormer/evaluation/dataset_stats.py -c $TRAINCFG -o $OUT2 --base-layer norm
