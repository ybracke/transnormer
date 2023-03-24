# `transnormer`

A lexical normalizer for historical spelling variants using a transformer architecture.

This project provides code for training encoder-decoder transformer models,
applying a model and inspecting and evaluating a model's performance.



## Project Organization

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train` [not in use yet]
    ├── README.md          
    ├── requirements.txt   
    ├── requirements-dev.txt   
    ├── pyproject.toml     <- makes project pip installable 
    |
    ├── data                  [not in use yet]
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained models
    │
    ├── notebooks          <- Jupyter notebooks
    │
    ├── references         <- Manuals and other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    └── src                <- Source code for use in this project. [TODO]
        └── transnormer        
            ├── __init__.py    <- Makes src a Python module
            │
            ├── data           <- Scripts to download or generate data
            │   └── make_dataset.py
            │
            ├── features       <- Scripts to turn raw data into features for modeling
            │   └── build_features.py
            │
            ├── models         <- Scripts to train models and then use trained models to make
            │   │                 predictions
            │   ├── predict_model.py
            │   └── train_model.py
            |
            └── tests  <- Testing facilities for source code
            │
            └── visualization  <- Scripts to create exploratory and results oriented visualizations
                └── visualize.py


--------

Project structure is based on the [cookiecutter data science project template](https://drivendata.github.io/cookiecutter-data-science/).


## Installation

Create a conda environment and install dependencies.

```bash
# Install conda environment
conda install -y pip
conda create -y --name <environment-name> python=3.9 pip
conda activate <environment-name>

conda install -y cudatoolkit=11.3.1 cudnn=8.3.2 -c conda-forge
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
pip install torch==1.12.1+cu113 torchvision torchaudio -f https://download.pytorch.org/whl/torch_stable.html

# Install dependencies 
pip install -r requirements.txt
pip install -r requirements-dev.txt

# and/or install the transnormer package (-e for editable mode)
pip install -e . 
```

**Hints**

* Do `export TOKENIZERS_PARALLELISM=false` to get rid of parallelism warning
  messages (see `stdout-with-erros.md`)
* If you use the `Trainer()`, do `export CUDA_VISIBLE_DEVICES=1` and 
  `gpu =  "cuda:0"` in config file. Otherwise it will use both GPUs automatically.


## Required resources

* A dataset of historical language documents with (gold-)normalized labels
* Encoder and decoder models (available on the [Huggingface Model Hub](huggingface.co/models))
* Tokenizers that belong to the models (also available via Huggingface)

## Config file

To run the training script, you also need a toml file called `training_config.toml`
with training configuration parameters. The file `training_config_TEMPLATE.toml`
is a template for this.  
**NOTE: This will probably be changed in later versions, see issues [6](https://github.com/ybracke/transnormer/issues/6) and [7](https://github.com/ybracke/transnormer/issues/7)**


## Usage

Run the training script:

```bash
cd src/transnormer/models
python3 model_train.py
```


## Intuition 

Historical text normalization is treated as a seq2seq task, like machine
translation. We use a transformer encoder-decoder model. The encoder-decoder
gets warm-started with pre-trained models and fine-tuned on a dataset for
lexical normalization. 

1. Encoder for historic German 
2. Decoder for modern German 
3. Encoder-decoder wired together
   * Supervised learning with labeled data


## Motivation

* Transformers are state of the art in NLP
  * By using pre-trained transformer models we can leverage linguistic knowledge
  from large quantities of data
* There exists more historical text that is not normalized than (gold) normalized text
  * An encoder (LM) can be learned for historical text without a normalization layer
* Intuition: We create a model from an encoder that knows a lot about historical
  language, and a decoder that knows a lot about modern language and plug them
  together by training them on gold-normalized data.


## Background

**[Perhaps this should go somewhere else, e.g. into `./references`]**

### References

* For a blogpost on warm-starting encoder-decoder, see [here](https://huggingface.co/blog/warm-starting-encoder-decoder)
  * Corresponding [colab notebook](https://colab.research.google.com/drive/1Ekd5pUeCX7VOrMx94_czTkwNtLN32Uyu?usp=sharing)
* Paper ["Leveraging Pre-trained Checkpoints for Sequence Generation Tasks"](https://arxiv.org/abs/1907.12461)


## DVC 

This project uses [DVC](https://dvc.org/doc) for (1) versioning data and model
files (2) tracking experiments. 

### Versioning data and models

Large data and model files are versioned with DVC and do not get tracked by git.
Instead, only a hash (stored in a text file) is tracked by git, either in a
`<dataname>.dvc` or in `dvc.lock` are tracked by git. 

`dvc list . --dvc-only --recursive`  
-> shows the files tracked by DVC

`dvc push` moves a specific version of the data to the remote storage.


### Tracking experiments

DVC is also used for tracking experiments to make models reproducible and to
separate code-development from experiments. Each DVC experiment is a snapshot of
the state of the code, the configs, the trained model resulting from these, and
possibly evaluation metrics at a specific point in time.

#### Workflow

1. Make sure any recent changes to the code are committed
2. Set parameters in the config file (`training_config.toml`)
3. Make sure `dvc.yaml` is still up-to-date (i.e. contains all dependencies,
  parameters, output locations, etc.)
4. Run the training with `dvc exp run` 
   * You can also set parameters in the config file at this stage with
     `--set-params|-S`. Example:   
     `dvc exp run -S 'training_config.toml:training_hyperparams.save_steps=50'`)
5. `dvc exp run` (1) creates a new version of the model and (2) modifies
  `dvc.lock`. Instead of an individual `model.dvc` file for the model, its path,
  md5, etc. are stored in `dvc.lock` under `outs`, (3) creates a hidden commit
  that also contains the config and `dvc.lock` (i.e. a link to the updated
  model).
6. To push the new model version to the remote, do: `dvc push models/model`.
7. `git restore .` will restore the updated `dvc.lock` and config files. You
   don't have to git-commit these separately to git because this was already done
   automatically in step 5.


#### Background

An experiment in DVC is a set of changes to your data, code, and configuration
that you want to track and reproduce. When you run `dvc exp run` command, DVC
will create a snapshot of your code and data, and save it as a hidden git
commit. Technically: Experiments are custom Git references (found in
`.git/refs/exps`) with one or more commits based on HEAD. These commits are
hidden and not checked out by DVC and not pushed to git remotes either.


### Checking out a specific experiment

`dvc exp show`  
-> shows the experiments done on the current HEAD (add `-A` for all commits)

`dvc exp apply <name>`    
-> puts the version of configs, code and model for experiment `<name>` into the
current workspace

`dvc exp apply HEAD`   
-> goes back to status of HEAD
