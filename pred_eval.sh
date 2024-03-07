# Run prediction and evaluation scripts and save the configurations and results

# Exit in case of errors
set -e

# Paths (adjust if needed)
DIR_PRED=hidden/predictions
DIR_TESTCFG=hidden/test_configs
DIR_SENTSCORES=hidden/sent_scores
BASENAME_PRED=preds.jsonl
PATH_PRED=$DIR_PRED/$BASENAME_PRED


# Prepare conda environment
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate gpu-venv-transnormer
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
export CUDA_VISIBLE_DEVICES=1

# Create generations based on ./test_config.toml
python3 src/transnormer/models/generate.py -c test_config.toml --out $PATH_PRED

# Rename test_config file to a unique name and copy
fname_testcfg=`md5sum test_config.toml | head -c 8`.toml
cp test_config.toml $DIR_TESTCFG/$fname_testcfg
echo "Test config stored here: $DIR_TESTCFG/$fname_testcfg"

# Rename the predictions file to a unique name
fname_preds=`md5sum $PATH_PRED | head -c 8`.jsonl
mv $PATH_PRED $DIR_PRED/$fname_preds
echo "Predictions stored here: $DIR_PRED/$fname_preds"

# Call evaluation script
python3 src/transnormer/evaluation/evaluate.py \
  --input-type jsonl \
  --ref-file $DIR_PRED/$fname_preds \
  --pred-file $DIR_PRED/$fname_preds \
  --ref-field=norm --pred-field=pred -a both \
  --sent-wise-file $DIR_SENTSCORES/sent_scores_${fname_preds%.*}.pkl \
  --test-config $DIR_TESTCFG/$fname_testcfg \
  >> hidden/eval.jsonl

# Store sentence scores with the predictions
python3 src/transnormer/evaluation/add_sent_scores.py $DIR_SENTSCORES/sent_scores_${fname_preds%.*}.pkl $DIR_PRED/$fname_preds
