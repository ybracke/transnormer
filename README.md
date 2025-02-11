# `Transnormer`

`Transnormer` models are byte-level sequence-to-sequence models for normalizing historical German text.
This repository contains code for training and evaluating `Transnormer` models.

- [`Transnormer`](#transnormer)
  - [Models](#models)
    - [Using Public Models](#using-public-models)
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
    - [2. Generating normalizations](#2-generating-normalizations)
      - [Test config file](#test-config-file)
    - [3. Evaluation](#3-evaluation)
      - [3.1 Get evaluation metrics](#31-get-evaluation-metrics)
      - [3.2 `pred_eval.sh`](#32-pred_evalsh)
  - [Project](#project)
  - [License](#license)



## Models

**Note:** This section is continously updated.

We release *transnormer* models and evaluation results on the Hugging Face Hub.


| Model | Test set | Time period | WordAcc | WordAcc (-i) |
| --- | --- | --- | --- | --- |
| Identity baseline | [DTA reviEvalCorpus-v1](https://huggingface.co/datasets/ybracke/dta-reviEvalCorpus-v1) | 1780-1899 | 91.45 | 93.25 |
| [transnormer-19c-beta-v02](https://huggingface.co/ybracke/transnormer-19c-beta-v02) | [DTA reviEvalCorpus-v1](https://huggingface.co/datasets/ybracke/dta-reviEvalCorpus-v1) | 1780-1899 | 98.88 | 99.34 |

The metric *WordAcc* is the harmonized word accurracy (Bawden et al. 2022) explained [below](#31-get-evaluation-metrics); *-i* denotes a case insensitive version (i.e. deviations in casing between prediction and gold normalization are ignored). The identity baseline only replaces outdated characters by their modern counterpart (e.g. "ſ" -> "s", "aͤ" -> "ä").

### Using Public Models

Models are easy to use with the [`transformers`](https://huggingface.co/docs/transformers/index) library:

```python
from transformers import pipeline

transnormer = pipeline(model='ybracke/transnormer-19c-beta-v02')
sentence = "Die Königinn ſaß auf des Pallaſtes mittlerer Tribune."
print(transnormer(sentence, num_beams=4, max_length=128))
# >>> [{'generated_text': 'Die Königin saß auf des Palastes mittlerer Tribüne.'}]
```


## Installation

In order to reproduce model training and evaluation, install the dependencies and code as described in this section and refer to the documentation in the section on [Usage](#usage).

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

Set up a virtual environment, e.g.:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

### 2.a Install package from GitHub

```bash
pip install git+https://github.com/ybracke/transnormer.git@dev
```

### 2.b Editable install for developers

```bash
# Clone repo from GitHub
git clone git@github.com:ybracke/transnormer.git
git switch dev
cd ./transnormer
# install package in editable mode
pip install -e .
# install development requirements
pip install -r requirements-dev.txt
```

### 3. Requirements

To train a model you need the following resources:

* A pre-trained encoder-decoder model (available on the [Huggingface Model Hub](huggingface.co/models))
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
2. Specify file paths in `pred_eval.sh`, then run: `bash pred_eval.sh`

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

* If you have multiple GPUs available and want to use only one (e.g. the GPU with index `1`):
  * `export CUDA_VISIBLE_DEVICES=1`
  * Set `gpu = "cuda:0"` in [config file](#1-select-gpu)
* `export TOKENIZERS_PARALLELISM=false` to get rid of parallelism warning messages


### Preparation 2: Data preparation

The training and test data must be in [JSONL](https://jsonlines.org/) format, where each record is a parallel training sample, e.g. a sentence. The records in the files must at least have the following format:

```jsonc
{
    "orig" : "Eyn Theylſtueck", // original spelling
    "norm" : "Ein Teilstück"    // normalized spelling
}
```

See repository [transnormer-data](https://github.com/ybracke/transnormer-data) for more information.


### 1. Model training

1. Specify the training parameters in the [config file](#training-config-file)

2. Run training script: `$ python3 src/transnormer/models/model_train.py`. Training can take multiple hours, so consider using `nohup`: `$ nohup nice python3 src/transnormer/models/train_model.py &`

#### Training config file

The file `training_config.toml` specifies the training configurations, e.g. training data, base model, training hyperparameters. Update the file before fine-tuning model.

The following paragraphs provide detailed explanations of each section and parameter within the configuration file to facilitate effective model training.

##### 1. Select GPU <!-- omit in toc -->

The `gpu` parameter sets the GPU device used for training. You can set it to the desired GPU identifier, such as `"cuda:0"`, ensuring compatibility with the CUDA environment. Remember to set the appropriate CUDA visible devices beforehand, if required (e.g. `export CUDA_VISIBLE_DEVICES=1` to use only the GPU with index `1`).

##### 2. Random Seed (Reproducibility) <!-- omit in toc -->

The `random_seed` parameter defines a fixed random seed (`42` in the default settings) to ensure reproducibility of the training process.

##### 3. Data Paths and Subset Sizes <!-- omit in toc -->

The `[data]` section references the training and evaluation data. `paths_train`, `paths_validation`, and `paths_test` are lists of paths to JSONL files or to directories that only contain JSONL files. See [data preparation](#preparation-2-data-preparation) for more information on the data format. Additionally, `n_examples_train`, `n_examples_validation`, and `n_examples_test` specify the number of examples to be used from each dataset split during training.

Both `paths_{split}` and `n_examples_{split}` are lists. The number at `n_examples_{split}[i]` refers to the number of examples to use from the data specified at `paths_{split}[i]`. Hence `n_examples_{split}` must contain the same amount of elements as `paths_{split}`. Setting `n_examples_{split}[i]` to a value higher than the number of examples in `paths_{split}[i]` ensures that all examples in this split will be used, but no oversampling is applied.

Per default the samples get shuffled by the training code, set `do_shuffle = false` to prevent this. Set `reverse_labels = true` to switch the labels (`orig` and `norm`) of the training data in order to train a denormalizer.

##### 4. Tokenizer Configuration <!-- omit in toc -->

The `[tokenizer]` section holds settings related to tokenization of input and output sequences. Specify the `tokenizer` that belongs to the model, the `padding` behavior (see [huggingface reference](https://huggingface.co/docs/transformers/pad_truncation)).
If you omit `tokenizer`, the program will attempt to use the tokenizer of the checkpoint given under `language_model`.
You can specify an `input_transliterator` for data preprocessing. This option is not implemented for the byte-based models and might be removed in the future.
You can adjust `min_length_input` and `max_length_input` to filter inputs before traing.

##### 5. Language Model Selection <!-- omit in toc -->

Under `[language_models]`, specify the model that is to be fine-tuned. Currently, only encoder-decoder models of the type [ByT5](https://huggingface.co/google/byt5-small) are safely supported. Set `from_scratch = true` to do a retraining from scratch instead of fine-tuning.

##### 6. Training Hyperparameters <!-- omit in toc -->

The `[training_hyperparams]` section specifies essential training parameters, such as `batch_size` (determines the number of examples in each training batch), `epochs` (indicates the number of training epochs), `fp16` (toggles half-precision training), and `learning_rate`. Refer to [`transformers.Seq2SeqTrainingArguments`](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Seq2SeqTrainingArguments) for details. You can control the frequency of logging, evaluation, and model saving using `logging_steps`, `eval_steps`, and `save_steps` respectively.


### 2. Generating normalizations

The script `src/transnormer/models/generate.py` generates normalizations given a [config file](#test-config-file) and saves a JSONL file with the same properties as the input file, plus a `pred` property for the predicted normalization.

```
usage: generate.py [-h] [-c CONFIG] [-o OUT]

Generates normalizations given a configuration file that specifies the model, the data and parameters.

optional arguments:
  -h, --help            show this help message and exit
  -c CONFIG, --config CONFIG
                        Path to the config file (TOML)
  -o OUT, --out OUT     Path to the output file (JSONL)
```

#### Test config file

The test config file configures which data, tokenizer, model and generation parameters are used when generating normalizations with `generate.py`. The template file `test_config.toml` illustrates the usage; some sections are identical to the [training config file](#training-config-file). The following is a description of the most relevant parts of the config file.

##### Test data <!-- omit in toc -->

The `[data]` specifies information concerning the test data.
- `path_test` references either a local JSONL file (e.g. `"data/test.jsonl"`), a local directory containing JSONL files (e.g. `"data/test"`) or the name of a Hugging Face dataset (e.g. `"ybracke/dta-reviEvalCorpus-v1"`).
- Only if `path_test` is a Hugging Face dataset, `split` should be specified. It must reference any of the datasets existing splits (e.g. `"test"`).
- `max_bytelength` is optional and can be specified to set an upper boundary for the length of individual inputs, i.e. to remove all samples from the test set where the `"orig"` string exceeds the byte length specified here.
- `n_examples_test` is optional and can be included to specify the number of test examples to use. If this is specified to a number *N* lower than the total of test set samples (possibly after filtering according to `max_bytelength`), a random *N* samples will be selected for testing.

##### Generation configurations <!-- omit in toc -->

The `[generation_config]` section contains parameters related to generation, e.g. `early_stopping`, `length_penalty` (higher value favors longer sequences), `num_beams` (the number of beams to use in beam search, less is faster), `max_new_tokens` (maximum output length in bytes). Refer to [`transformer.GenerationConfig`](https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig) for more options and documentation.

### 3. Evaluation

The quickest way to generate normalizations and get evaluation metrics is to adjust the [test config file](#test-config-file) and run `$ bash pred_eval.sh` (see [below](#32-pred-evalsh)).

#### 3.1 Get evaluation metrics

The script `src/transnormer/evaluation/evaluate.py` computes a harmonized accuracy score and the normalized Levenshtein distance. The metrics and their computation are adopted from [Bawden et al. (2022)](https://github.com/rbawden/ModFr-Norm).

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
  --input-type jsonl --ref-file d037b975.jsonl \
  --pred-file d037b975.jsonl \
  --ref-field=norm --pred-field=pred -a both \
  --sent-wise-file sent_scores_d037b975.pkl \
  --test-config d1b1ea77.toml
```

In this case, the gold normalizations (*ref*) and auto-generated normalizations (*pred*) are stored in the same JSONL file, therefore `--ref-file` and `--pred-file` take the same argument.
If *ref* and *pred* texts are stored in different files, the examples in the files must be in the same order.
Global evaluation metrics are printed to stdout by default and can be redirected into a file.

If you have a single JSONL file with original input, predictions and gold labels and you want to write the sentence-wise accuracy scores (that have been computed by `evaluate.py`) to this file, you can do this with `src/transnormer/evaluation/add_sent_scores.py`:

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


#### 3.2 `pred_eval.sh`

This bash script runs the python scripts for generation and evaluation and performs copy/rename operations to automatically store config and prediction files under unique names via hashed file names.


## Project

This project is developed at the [Berlin-Brandenburg Academy of Sciences and Humanities](https://www.bbaw.de) (Berlin-Brandenburgische Akademie der Wissenschaften, BBAW) within the national research data infrastructure (Nationale Forschungsdateninfrastruktur, NFDI) [Text+](https://www.text-plus.org/).


## License

The code in this project is licensed under [GNU General Public License v3.0](https://github.com/ybracke/transnormer/blob/dev/LICENSE).
