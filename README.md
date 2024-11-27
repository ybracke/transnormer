# `transnormer`

A lexical normalizer for historical spelling variants using a transformer architecture.


- [`transnormer`](#transnormer)
  - [Models](#models)
  - [Installation](#installation)
    - [1. Set up environment](#1-set-up-environment)
    - [2.a Install package from GitHub](#2a-install-package-from-github)
    - [2.b Editable install for developers](#2b-editable-install-for-developers)
    - [3. Requirements](#3-requirements)
  - [Usage](#usage)
    - [Quickstart](#quickstart)
      - [Quickstart Training](#quickstart-training)
      - [Quickstart Generation and Evaluation](#quickstart-generation-and-evaluation)
    - [Preparation 1: Virtual environment](#preparation-1-virtual-environment)
    - [Preparation 2: Data preparation](#preparation-2-data-preparation)
    - [1. Model training](#1-model-training)
      - [Training config file](#training-config-file)
      - [Resume training a model](#resume-training-a-model)
    - [2. Generating normalizations](#2-generating-normalizations)
      - [Test config file](#test-config-file)
      - [Unique names for config and prediction files](#unique-names-for-config-and-prediction-files)
    - [3. Evaluation](#3-evaluation)
      - [3.1 Metrics](#31-metrics)
      - [3.2 Inspecting and analyzing outputs](#32-inspecting-and-analyzing-outputs)
  - [Project](#project)
  - [License](#license)



## Models

**Note:** This section is continously updated.

We release *transnormer* models and evaluation results on the Hugging Face Hub.


| Model | Test set | Time period | WordAcc | WordAcc (-i) |
| --- | --- | --- | --- | --- |
| [transnormer-19c-beta-v02](https://huggingface.co/ybracke/transnormer-19c-beta-v02) | [DTA reviEvalCorpus-v1](https://huggingface.co/datasets/ybracke/dta-reviEvalCorpus-v1) | 1780-1899 | 98.88 | 99.34 |

The metric *WordAcc* is the harmonized word accurracy (Bawden et al. 2022) explained [below](#31-metrics); *-i* denotes a case insensitive version (i.e. deviations in casing between prediction and gold normalizaiton are ignored).


## Installation

### 1. Set up environment

#### 1.a On a GPU <!-- omit in toc -->

If you have a GPU available, you should first install and set up a conda environment.

```bash
conda install -y pip
conda create -y --name <environment-name> python=3.9 pip
conda activate <environment-name>

conda install -y cudatoolkit=11.3.1 cudnn=8.3.2 -c conda-forge
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
pip install torch==1.12.1+cu113 torchvision torchaudio -f https://download.pytorch.org/whl/torch_stable.html
```

#### 1.b On a CPU <!-- omit in toc -->

Set up a virtual environment, e.g. like this

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

### 2.a Install package from GitHub

```bash
pip install git+https://github.com/ybracke/transnormer.git
```

### 2.b Editable install for developers

```bash
# Clone repo from GitHub
git clone git@github.com:ybracke/transnormer.git
cd ./transnormer
# install package in editable mode
pip install -e .
# install development requirements
pip install -r requirements-dev.txt
```

### 3. Requirements

To train a model you need the following resources:

* Encoder and decoder models (available on the [Huggingface Model Hub](huggingface.co/models))
* Tokenizers that belong to the models (also available via Huggingface)
* A dataset of historical language documents with (gold-)normalized labels
* A file specifying the training configurations, see [Training config file](#training-config-file)


## Usage

### Quickstart

1. Prepare environment (see [below](#preparation-1-virtual-environment))
2. Prepare data (see [below](#preparation-2-data-preprocessing))

#### Quickstart Training

1. Specify the training parameters in the [training config file](#training-config-file)
2. Run training script: `$ python3 src/transnormer/models/model_train.py`.

For more details, see [below](#1-model-training)

#### Quickstart Generation and Evaluation

1. Specify the generation parameters in the [test config file](#test-config-file)
2. If necessary adjust paths in `pred_eval.sh`. Then run: `bash pred_eval.sh`

For more details, see sections on [Generation](#2-generating-normalizations) and [Evaluation](#3-evaluation).


### Preparation 1: Virtual environment

#### `venv` <!-- omit in toc -->

```bash
source .venv/bin/activate
```

#### Conda <!-- omit in toc -->

```bash
conda activate <environment-name>
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
```

* If you have multiple GPUs available and want to use only one (here: the GPU with index `1`):
  * `export CUDA_VISIBLE_DEVICES=1`
  * Set `gpu = "cuda:0"` in [config file](#1-select-gpu)
* `export TOKENIZERS_PARALLELISM=false` to get rid of parallelism warning messages


### Preparation 2: Data preparation

See repository [transnormer-data](https://github.com/ybracke/transnormer-data)


### 1. Model training

1. Specify the training parameters in the [config file](#training-config-file)

2. Run training script: `$ python3 src/transnormer/models/model_train.py`. (Don't forget to start your virtual environment first; see [Installation](#installation).) Training can take multiple hours, so consider using `nohup`: `$ nohup nice python3 src/transnormer/models/train_model.py &`



#### Training config file

[TODO]

The file `training_config.toml` serves as a comprehensive configuration guide for customizing and fine-tuning the training process of a language model using the specified parameters.

Please note that the provided configuration settings and parameters are examples. You can customize them to fit your specific training requirements. Refer to the comments within the configuration file for additional information and guidance on modifying these parameters for optimal training outcomes.

The following paragraphs provide detailed explanations of each section and parameter within the configuration file to facilitate effective model training.

##### 1. Select GPU <!-- omit in toc -->

The `gpu` parameter allows you to specify the GPU device for training. You can set it to the desired GPU identifier, such as `"cuda:0"`, ensuring compatibility with the CUDA environment. Remember to set the appropriate CUDA visible devices beforehand using if required (e.g. `export CUDA_VISIBLE_DEVICES=1` to use only the GPU with index `1`).

##### 2. Random Seed (Reproducibility) <!-- omit in toc -->

The `random_seed` parameter defines a fixed random seed (`42` in the default settings) to ensure reproducibility of the training process. This enables consistent results across different runs.

##### 3. Data Paths and Subset Sizes <!-- omit in toc -->

The `[data]` section includes paths to training, validation, and test datasets. The `paths_train`, `paths_validation`, and `paths_test` parameters provide paths to respective JSONL files containing data examples. Additionally, `n_examples_train`, `n_examples_validation`, and `n_examples_test` specify the number of examples to be used from each dataset split during training.

Both `paths_{split}` and `n_examples_{split}` are lists. The number at `n_examples_{split}[i]` refers to the number of examples to use from the data specified at `paths_{split}[i]`. Hence `n_examples_{split}` must be the same length as `paths_{split}`. Setting `n_examples_{split}[i]` to a value higher than the number of examples in `paths_{split}[i]` ensures that all examples in this split will be used, but no oversampling is applied.

##### 4. Tokenizer Configuration <!-- omit in toc -->

The `[tokenizer]` section holds settings related to tokenization of input and output sequences. You can specify `tokenizer_input` and `tokenizer_output` models. If you omit `tokenizer_output`, `tokenizer_input` will be used as the output tokenizer as well. If you omit `tokenizer_input`, the program will try to use the tokenizer of the checkpoint given under `language_model`.

You can specify an `input_transliterator` for data preprocessing. This option is not implemented for the byte-based models and might be removed in the future.
You can adjust `min_length_input` and `max_length_input` to filter inputs before traing. You can set `max_length_output` to define the maximum token lengths of output sequences, though this is not recommended and the property might be removed.

##### 5. Language Model Selection <!-- omit in toc -->

Under `[language_models]`, you can choose the language model(s) to be retrained. It is possible to either use a byte-based encoder-decoder as the base model **or** two subword-based models (encoder and decoder). Accordingly the config file must either specify a `checkpoint_encoder_decoder` parameter, which points to the checkpoint of the chosen encoder-decoder model **or** two parameters, `checkpoint_encoder` (for historic language) **and** `checkpoint_decoder` (for modern language).

This section may change in the future, see this [issue](https://github.com/ybracke/transnormer/issues/67).

##### 6. Training Hyperparameters <!-- omit in toc -->

The `[training_hyperparams]` section encompasses essential training parameters, such as `batch_size` (determines the number of examples in each training batch), `epochs` (indicates the number of training epochs) ~~, and `learning_rate`~~ (not actually used). You can control the frequency of logging, evaluation, and model saving using `logging_steps`, `eval_steps`, and `save_steps` respectively. `eval_strategy` defines how often evaluation occurs, and `fp16` toggles half-precision training.

This section may change in the future, see this [issue](https://github.com/ybracke/transnormer/issues/88).

##### 7. Beam Search Decoding Parameters <!-- omit in toc -->

The `[beam_search_decoding]` section contains parameters related to beam search decoding during inference. `no_repeat_ngram_size` prevents n-grams of a certain size from repeating. (Note that what is a sensible value for this parameter is different depending on the tokenization. For a char/byte-based (aka "tokenizer-free") model, set this to higher value than for subword-based models.) `early_stopping` enables stopping decoding when early stopping criteria are met. `length_penalty` controls the trade-off between sequence length and probability. `num_beams` specifies the number of beams to use in beam search.

This section may change in the future, see this [issue](https://github.com/ybracke/transnormer/issues/89).

#### Resume training a model

[TODO]

We may want to fine-tune a model that is already the product of fine-tuning. We call the first fine-tuned model 'checkpoint-X' and the second model 'checkpoint-Y'. To train checkpoint-Y from checkpoint-X simply add the path to checkpoint-X under `language_models` in `training_config.toml`.

To clarify, checkpoint-Y was created like this:
```original pretrained model (e.g. byt5-small) -> checkpoint-X -> checkpoint-Y```

Thus, in order to keep track of the full provenance of checkpoint-Y, we must not only keep checkpoint-Y's `training_config.toml` but also keep the directory where checkpoint-X and its `training_config.toml` is located.


### 2. Generating normalizations

The fastest way to create normalizations and get evaluation metrics is to run the bash script:
`bash pred_eval.sh`
This runs the scripts for generation and evaluation and performs the copy/rename operations described in the following.

---

The script `src/transnormer/models/generate.py` generates normalizations given a [config file](#test-config-file). This produces at JSONL file with generated normalizations.

```
usage: generate.py [-h] [-c CONFIG] [-o OUT]

Generates normalizations given a configuration file that specifies the model, the data and parameters.

optional arguments:
  -h, --help            show this help message and exit
  -c CONFIG, --config CONFIG
                        Path to the config file (TOML)
  -o OUT, --out OUT     Path to the output file (JSONL)
```

Example call:

```
python3 src/transnormer/models/generate.py -c test_config.toml --out <path>
```

#### Test config file

The test config file configures which device, data, tokenizer, model and generation parameters are used when generating normalizations. Refer to `test_config.toml` for a template and the description of the [training config file](#training-config-file) for a detailed description of the sections. Note that, currently, only a single test data file is allowed as input.

#### Unique names for config and prediction files

[TODO]

Rename and copy current `test_config.toml`:

```bash
# in the transformer directory call
filename=`md5sum test_config.toml | head -c 8`.toml
cp test_config.toml hidden/test_configs/$filename
```

Rename the predictions file (e.g. `hidden/predictions/preds.jsonl`) to a unique name like this:

```bash
# go to predictions directory
cd hidden/predictions
# rename pred file
filename=`md5sum preds.jsonl | head -c 8`.jsonl
mv preds.jsonl $filename
```


### 3. Evaluation

#### 3.1 Metrics

The script `src/transnormer/evaluation/evaluate.py` computes a harmonized accuracy score and the normalized Levenshtein distance. The metric and its computation are adopted from [Bawden et al. (2022)](https://github.com/rbawden/ModFr-Norm).

```
usage: evaluate.py [-h] --input-type {jsonl,text} [--ref-file REF_FILE] [--pred-file PRED_FILE]
                   [--ref-field REF_FIELD] [--pred-field PRED_FIELD] -a ALIGN_TYPES [--sent-wise-file SENT_WISE_FILE]
                   [--test-config TEST_CONFIG]

Compute evaluation metric(s) for string-to-string normalization (see Bawden et al. 2022). Choose --align-type=both for a harmonized accuracy score.

optional arguments:
  -h, --help            show this help message and exit
  --input-type {jsonl,text}
                        Type of input files: jsonl or text
  --ref-file REF_FILE   Path to the input file containing reference normalizations (typically a gold standard)
  --pred-file PRED_FILE
                        Path to the input file containing predicted normalizations
  --ref-field REF_FIELD
                        Name of the field containing reference (for jsonl input)
  --pred-field PRED_FIELD
                        Name of the field containing prediction (for jsonl input)
  -a ALIGN_TYPES, --align-types ALIGN_TYPES
                        Which file's tokenisation to use as reference for alignment. Valid choices are 'both', 'ref',
                        'pred'. Multiple choices are possible (comma separated)
  --sent-wise-file SENT_WISE_FILE
                        Path to a file where the sentence-wise accuracy scores get saved. For pickled output (list),
                        the path must match /*.pkl/. Textual output is a comma-separated list
  --test-config TEST_CONFIG
                        Path to the file containing the test configurations
```


Example call:

```bash
python3 src/transnormer/evaluation/evaluate.py \
  --input-type jsonl --ref-file hidden/predictions/d037b975.jsonl \
  --pred-file hidden/predictions/d037b975.jsonl \
  --ref-field=norm --pred-field=pred -a both \
  --sent-wise-file hidden/sent_scores/sent_scores_d037b975.pkl \
  --test-config hidden/test_configs/d1b1ea77.toml \
  >> hidden/eval.jsonl
```

In this case, the gold normalizations ("ref") and auto-generated normalizations ("pred") are stored in the same JSONL file, therefore `--ref-file` and `--pred-file` take the same argument. If `ref` and `pred` texts are stored in different files, the files must be in the same order (i.e. example in line 1 of the ref-file refers to the example in line 1 of the pred-file, etc.). Global evaluation metrics are printed to stdout by default and can be redirected, as in the example above.

If you have a single JSONL file with original input, predictions and gold labels, you probably want to write the sentence-wise accuracy scores to this file, that have been computed by `evaluate.py`. This can be done with `src/transnormer/evaluation/add_sent_scores.py`:

```
usage: add_sent_scores.py [-h] [-p PROPERTY] scores data

Write sentence-wise accuracy scores stored SCORES to DATA (jsonl file)

positional arguments:
  scores                Scores file (either pickled (*.pkl) or comma-separated plain-text).
  data                  Data file (JSONL)

optional arguments:
  -h, --help            show this help message and exit
  -p PROPERTY, --property PROPERTY
                        Name for the property in which the score gets stored (default: 'score')
```

Example call:

```bash
python3 src/transnormer/evaluation/add_sent_scores.py hidden/sent_scores.pkl hidden/predictions/8ae3fd47.jsonl
```

```bash
python3 src/transnormer/evaluation/add_sent_scores.py hidden/sent_scores.pkl hidden/predictions/8ae3fd47.jsonl -p score_i
```

#### 3.2 Inspecting and analyzing outputs

[TODO]

**TODO**, see [this issue](https://github.com/ybracke/transnormer/issues/93)

Use `jq` to create a text-only version from the JSONL files containing the predictions and then call `diff` on that. Example:
```bash
jq -r '.norm' ./8ae3fd47.jsonl > norm
jq -r '.pred' ./8ae3fd47.jsonl > pred
code --diff norm pred
```


## Project

This project is developed at the [Berlin-Brandenburg Academy of Sciences and Humanities](https://www.bbaw.de) (Berlin-Brandenburgische Akademie der Wissenschaften, BBAW) within the national research data infrastructure (Nationale Forschungsdateninfrastruktur, NFDI) [Text+](https://www.text-plus.org/).


## License

The code in this project is licensed under [GNU General Public License v3.0](https://github.com/ybracke/transnormer/blob/dev/LICENSE).
