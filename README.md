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
is a template for this. [**NOTE: This will probably be changed in later versions**]


## Usage

Run the training script.

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


## TODOs and future lines of inquiry (possible issues)

* Make GitHub issues out of these points
* See `#TODO` in `model_train.py`
* Remove codecarbon again
* Remove all params from `model_train.py` and put them into `training_config.toml`
* Use a [dvc pipeline](https://dvc.org/doc/start/data-management/data-pipelines) to track experiment runs and parameter configurations
* Use character-based models instead of subword-tokens
* Can additional information be leveraged like the publication year or era
* How does the model deal with names?
* Experiment with a loss function that punishes wrongly changed words more than
  wrongly unchanged words (see Bollmann?) to get a "conservative" normalization 
* How well does the model work if we replace the pre-trained decoder with a
  randomly initialized one (BERT2Rnd)?
* Fine-tune encoder
* Fine-tune decoder

### Tokenizer consistency

I am currently using two different tokenizers, one for historical language and
one for modern language. It seems to be common (or the only possible way?) to
use only a single tokenizer if you want to create a huggingface model
(`transformers.EncoderDecoderModel`) including a tokenizer. 

---


