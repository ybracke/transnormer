# `transnormer`

A lexical normalizer for historical spelling variants using a transformer architecture.

## Installation

### 1. Set up environment

#### 1.a On a GPU

If you have a GPU available, you should first install and set up a conda environment.

```bash
conda install -y pip
conda create -y --name <environment-name> python=3.9 pip
conda activate <environment-name>

conda install -y cudatoolkit=11.3.1 cudnn=8.3.2 -c conda-forge
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
pip install torch==1.12.1+cu113 torchvision torchaudio -f https://download.pytorch.org/whl/torch_stable.html
```

#### 1.b On a CPU

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
* A file specifying the training configurations, see [Configuration](#configuration)

## Usage

### Preparation 1: Virtual environment

#### `venv`

```bash
source .venv/bin/activate
```

#### Conda

```bash
conda activate <environment-name>
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
```

* If you have multiple GPUs available and want to use only one (here: the GPU with index `1`):
  * `export CUDA_VISIBLE_DEVICES=1`
  * Set `gpu = "cuda:0"` in [config file](#1-select-gpu)
* `export TOKENIZERS_PARALLELISM=false` to get rid of parallelism warning messages


### Preparation 2: Data preprocessing

Scripts and functions in `src/transnormer/data`

TODO - Describe what they do (inspiration: https://github.com/clarinsi/csmtiser#data-preprocessing)


#### `split_dataset.py`

```
usage: split_dataset.py [-h] [-o OUT] [-v VALIDATION_SET_SIZE] [-t TEST_SET_SIZE]
                        [--random-state RANDOM_STATE]
                        file

Create genre-stratified train, validation and test splits of the DTA for a specific time
frame.

positional arguments:
  file                  Input file (JSON Lines).

optional arguments:
  -h, --help            show this help message and exit
  -o OUT, --out OUT     Path to the output file (JSON Lines).
  -v VALIDATION_SET_SIZE, --validation-set-size VALIDATION_SET_SIZE
                        Size of the validation set as a fraction of the total data.
  -t TEST_SET_SIZE, --test-set-size TEST_SET_SIZE
                        Size of the test set as a fraction of the total data.
  --random-state RANDOM_STATE
                        Seed for the random state (default: 42).
```

#### `make_dataset.py`

```
usage: make_dataset.py [-h] [-t TARGET] DATASET [DATASET ...]

Convert datasets from raw format to JSON Lines

positional arguments:
  DATASET               Path(s) to dataset(s)

optional arguments:
  -h, --help            show this help message and exit
  -t TARGET, --target TARGET
                        Path to target directory
```

#### `read_*` function

In order to support reading in and converting a dataset to be used as training or test data, there has to be a `read_*` function in `loader.py` (e.g. `read_dtaeval_raw`).The `read_*` function must return a dict that looks like this:
`{ "orig" : List[str], "norm" : List[str]}`. Additional dict entries might be metadata, e.g. `"year" : List[int]`, `"document" : List[str]`.


### 1. Model training

1. Specify the training parameters in the [config file](#configuration)
2. Run training script: `$ python3 src/transnormer/models/model_train.py`. (Don't forget to start your virtual environment (see [Installation](#installation)) first.)

#### Configuration

The file `training_config.toml` serves as a comprehensive configuration guide for customizing and fine-tuning the training process of a language model using the specified parameters.

Please note that the provided configuration settings and parameters are examples. You can customize them to fit your specific training requirements. Refer to the comments within the configuration file for additional information and guidance on modifying these parameters for optimal training outcomes.

The following paragraphs provide detailed explanations of each section and parameter within the configuration file to facilitate effective model training.

##### 1. Select GPU

The `gpu` parameter allows you to specify the GPU device for training. You can set it to the desired GPU identifier, such as `"cuda:0"`, ensuring compatibility with the CUDA environment. Remember to set the appropriate CUDA visible devices beforehand using if required (e.g. `export CUDA_VISIBLE_DEVICES=1` to use only the GPU with index `1`).

##### 2. Random Seed (Reproducibility)

The `random_seed` parameter defines a fixed random seed (`42` in the default settings) to ensure reproducibility of the training process. This enables consistent results across different runs.

##### 3. Data Paths and Subset Sizes

The `[data]` section includes paths to training, validation, and test datasets. The `paths_train`, `paths_validation`, and `paths_test` parameters provide paths to respective JSONL files containing data examples. Additionally, `n_examples_train`, `n_examples_validation`, and `n_examples_test` specify the number of examples to be used from each dataset split during training.
Both `paths_{split}` and `n_examples_{split}` are lists. The number at `n_examples_{split}[i]` refers to the number of examples to use from the data specified at `paths_{split}[i]`. Hence `n_examples_{split}` must be the same length as `paths_{split}`. Setting `n_examples_{split}[i]` to a value higher than the number of examples in `paths_{split}[i]` ensures that all examples in this split will be used, but no oversampling is applied.

##### 4. Tokenizer Configuration

The `[tokenizer]` section holds settings related to tokenization of input and output sequences. You can adjust `max_length_input` and `max_length_output` to define the maximum token lengths for input and output sequences. This section also provides the option to specify an `input_transliterator` for transliteration purposes.

##### 5. Language Model Selection

Under `[language_models]`, you can choose the language model(s) to be retrained. It is possible to either use a byte-based encoder-decoder as base model **or** two subword-based models (encoder and decoder). Accordingly the config file must either specify a `checkpoint_encoder_decoder` parameter, which points to the checkpoint of the chosen encoder-decoder model **or** two parameters, `checkpoint_encoder` (for historic language) **and** `checkpoint_decoder` (for modern language).
This section may change in the near future, see this [issue](https://github.com/ybracke/transnormer/issues/67).


##### 6. Training Hyperparameters

The `[training_hyperparams]` section encompasses essential training parameters, such as `batch_size` (determines the number of examples in each training batch), `epochs` (indicates the number of training epochs), and `learning_rate`. You can control the frequency of logging, evaluation, and model saving using `logging_steps`, `eval_steps`, and `save_steps` respectively. `eval_strategy` defines how often evaluation occurs, and `fp16` toggles half-precision training.

##### 7. Beam Search Decoding Parameters

The `[beam_search_decoding]` section contains parameters related to beam search decoding during inference. `no_repeat_ngram_size` prevents n-grams of a certain size from repeating. (Note that what is a sensible value for this parameter is different depending on the tokenization. For a char/byte-based (aka "tokenizer-free") model, set this to higher value than for subword-based models.) `early_stopping` enables stopping decoding when early stopping criteria are met. `length_penalty` controls the trade-off between sequence length and probability. `num_beams` specifies the number of beams to use in beam search.

### 2. Generating normalizations

The script `src/transnormer/models/generate.py` generates normalizations.

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


### 3. Evaluation

#### 3.1 Metrics

The script `src/transnormer/evaluation/evaluate.py` computes an accuracy score and the normalized Levenshtein distance.

```
usage: evaluate.py [-h] --input-type {jsonl,text} [--ref-file REF_FILE] [--pred-file PRED_FILE]
                   [--ref-field REF_FIELD] [--pred-field PRED_FIELD] -a ALIGN_TYPES

Compute evaluation metric(s) for string-to-string normalization (see Bawden et al. 2022). Choose --align-type=both for a harmonized accuracy score.

optional arguments:
  -h, --help            show this help message and exit
  --input-type {jsonl,text}
                        Type of input files: jsonl or text
  --ref-file REF_FILE   Path to the input file containing reference normalizations (typically a gold
                        standard)
  --pred-file PRED_FILE
                        Path to the input file containing predicted normalizations
  --ref-field REF_FIELD
                        Name of the field containing reference (for jsonl input)
  --pred-field PRED_FIELD
                        Name of the field containing prediction (for jsonl input)
  -a ALIGN_TYPES, --align-types ALIGN_TYPES
                        Which file's tokenisation to use as reference for alignment. Valid choices are
                        'both', 'ref', 'pred'. Multiple choices are possible (comma separated)
```


Example call:

```
python3 src/transnormer/evaluation/evaluate.py --input-type jsonl --ref-file hidden/output/out02.jsonl --pred-file hidden/output/out02.jsonl --ref-field=norm --pred-field=pred -a ref,pred,both >> hidden/eval.jsonl
```

In this case, the gold normalizations ("ref") and auto-generated normalizations ("pred") are located in the same file, therefore `--ref-file` and `--pred-file` take the same argument. If they are located in different files, the files must be in the same order (i.e. example in line 1 of the ref-file refers to the example in line 1 of the pred-file, etc.).


#### 3.2 Inspecting and analyzing outputs

**TODO**

* Generations (or "predictions") were previously created with this Jupyter notebook: `notebooks/exploratory/inspect_predictions.ipynb`
* Now that generating normalizations is handled elsewhere, the notebook should be updated so that it reads in JSONL files containing fields like "orig", "gold" and "pred" and applies the analysis functions
* Could [Meld](https://meldmerge.org/) be helpful for a visual comparison of orig, gold and pred?


## Background

In this section you find information on the institutional and theoretical background of the project.

### Text+

This project is developed at the [Berlin-Brandenburg Academy of Sciences and Humanities](https://www.bbaw.de) (Berlin-Brandenburgische Akademie der Wissenschaften, BBAW) within the national research data infrastructure (Nationale Forschungsdateninfrastruktur, NFDI) [Text+](https://www.text-plus.org/).

### Description

We use a transformer encoder-decoder model. The encoder-decoder
gets warm-started with pre-trained models and fine-tuned on a dataset for
lexical normalization.

1. Pre-trained encoder for historic German
2. Pre-trained decoder for modern German
3. Plug encoder and decoder together by supervised learning with labeled data

Intuition: We create a model from an encoder that knows a lot about historical
  language, and a decoder that knows a lot about modern language and plug them
  together by training them on gold-normalized data. Both encoder and decoder can be pre-trained on large quantities of unlabeled data (historic/modern), which are more readily available than labeled data.

### CAB

This normalizer developed in this project is intended to become the successor of the normalizing component of the Cascaded Analysis Broker (CAB), developed at the BBAW by Bryan Jurish ([CAB webpage](https://kaskade.dwds.de/~moocow/software/DTA-CAB/), [CAB web service](https://kaskade.dwds.de/~moocow/software/DTA-CAB/), [Jurish (2012)](https://publishup.uni-potsdam.de/opus4-ubp/frontdoor/index/index/docId/5562)).

CAB is based on a combination of hand-craftet rules, edit distances and hidden markov language models. TODO

#### `transnormer` vs. CAB

This project contains some changes compared to CAB.

##### Machine learning

* Being based on the transformer architecture, this project uses state-of-the-art machine learning technology.
* Training machine learning models can be continued once more and/or better training data is available, thus allowing a continuous improvement of the models.
* Machine learning should help to find better normalizations for unknown texts and contexts.
* By using pre-trained transformer models we can leverage linguistic knowledge from large quantities of data

##### Sequence-to-sequence

The models trained with this project are sequence-to-sequence models (see [above](TODO)). This means, they take as input a string of unnormalized text and return a string of normalized text. This is different from CAB, which is a sequence tagger, where each token is assigned a single label ().

Unlike CAB our program can apply re-tokenization (normalizing word separation). TODO: Take more info from [here](https://pad.gwdg.de/KEg0QOHJQUyH5wyyANxHBQ#), tokenization of the output is not contingent of tokenization of historic data

##### Leverage large language models

TODO

* Have better knowledge of language
* Should deal with context better than CAB

##### Maintainability

* We base the program on components and libraries that are maintained by large communities, institutions or companies (e.g. Huggingface), instead of in-house developments that are less well supported.
* We move away from a C- and Perl-based program to a Python-based program, which has a larger community of users and developers.

### Roadmap

### More info

## Development

### Contributing

TODO

### Testing

TODO

### DVC

This project uses [DVC](https://dvc.org/doc) for (1) versioning data and model
files (2) tracking experiments.

#### Versioning data and models

Large data and model files are versioned with DVC and do not get tracked by git.
Instead, only a hash (stored in a text file) is tracked by git, either in a
`<dataname>.dvc` or in `dvc.lock` are tracked by git.

`dvc list . --dvc-only --recursive` shows the files tracked by DVC

`dvc push` moves a specific version of the data to the remote storage.


#### Tracking experiments

DVC is also used for tracking experiments to make models reproducible and to
separate code development from experiments. Each DVC experiment is a snapshot of
the state of the code, the configs, the trained model resulting from these, and
possibly evaluation metrics at a specific point in time.

##### Workflow: Run experiment

1. Make sure any recent changes to the code are committed
2. Set parameters in the config file (`training_config.toml`)
3. Make sure `dvc.yaml` is still up-to-date (i.e. contains all dependencies,
  parameters, output locations, etc.)
4. Run the training with `dvc exp run`
   * You can also set parameters in the config file at this stage with
     `--set-params|-S`. Example:
     `dvc exp run -S 'training_config.toml:training_hyperparams.save_steps=50'`)
5. Do `dvc exp run [--name <exp-name>]`
   (1) creates a new version of the model;
   (2) modifies `dvc.lock`. Instead of an individual `model.dvc` file for the
  model, its path, md5, etc. are stored in `dvc.lock` under `outs`;
  (3) creates a hidden commit that also contains the config and `dvc.lock` (i.e.
  a link to the updated model).
6. To push the new model version to the remote, do: `dvc push models/model`.
7. `git restore .` will restore the updated `dvc.lock` and config files. You
   don't have to git-commit these separately to git because this was already done
   automatically in step 5.

##### Workflow: Use a model from a specific experiment

1. `dvc exp branch <branch-name> <exp-name>` will create a git branch from the
   experiment. It makes sense to give the branch the same name as the
   experiment in order to associate them easily.
   (The experiment branch can be also created later from a hidden commit (see
   `dvc exp show --all-commits` to see all candidates). Note that the code in
   the experiment branch will be in the state as it was at the time of running
   the experiment.)
2. The experiment branch can now be used to analyze the model by running
   inspection notebooks.
3. Push the experiment branch to remote in order to be able to look at the
   analyses notebooks on GitHub. Remove the experiment branch from remote or
   alltogether if they are not needed them anymore (they can be recreated, see
   point 1).
4. You can reproduce the experiment (i.e. train the model again) either on the
   experiment branch or if you did `dvc exp apply <exp-name>` (on branch `dev`).

##### Experiments: Technical details / background

An "experiment" in DVC is a set of changes to your data, code, and configuration
that you want to track and reproduce. When you run `dvc exp run`, DVC
will create a snapshot of your code and data, and save it as a hidden git
commit. Technically: Experiments are custom Git references (found in
`.git/refs/exps`) with one or more commits based on HEAD. These commits are
hidden and not checked out by DVC and not pushed to git remotes either.


## License

The code in this project is licensed under [GNU General Public License v3.0](https://github.com/ybracke/transnormer/blob/dev/LICENSE).
