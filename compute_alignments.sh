# Save stats of training data

# Exit in case of errors
set -e

# Adjust model name
MODELSDIRNAME=models_2024-03-21


# Paths
TRAINCFG=/home/bracke/code/transnormer/models/$MODELSDIRNAME/training_config.toml
OUT1=/home/bracke/code/transnormer/hidden/statistics/typestats-orig2norm-$MODELSDIRNAME.pkl
# uncomment this and final line, if needed
# OUT2=/home/bracke/code/transnormer/hidden/statistics/typestats-norm2orig-$MODELSDIRNAME.pkl

# Prepare conda environment
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate gpu-venv-transnormer
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
export CUDA_VISIBLE_DEVICES=1

python3 src/transnormer/evaluation/dataset_stats.py -c $TRAINCFG -o $OUT1 --base-layer orig
# python3 src/transnormer/evaluation/dataset_stats.py -c $TRAINCFG -o $OUT2 --base-layer norm
