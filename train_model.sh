# Exit in case of errors
set -e

# Prepare conda environment
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate gpu-venv-transnormer
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
export CUDA_VISIBLE_DEVICES=1

python3 src/transnormer/models/train_model.py
