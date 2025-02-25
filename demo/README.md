# Demo

## Notebook `transnormer.ipynb`

The Jupyter Notebook exemplifies the basic usage of transnormer models.

## Script `process-text-file.py`

Demo script for processing a plain text file (containing an complete publication) with a transnormer model.
You can get an example text file with the bash script:

```bash
# create data/1700-1799/robins_artillerie_1745.txt
bash get-text-file.sh

# run demo
python3 process-text-file.py data/1700-1799/robins_artillerie_1745.txt
```

Notes:
* Public transnormer models were trained using a maximum input length of 512 bytes (~70 words). Inference is generally possible for longer sequences, but may give worse results than for shorter sequence. To avoid this problem, the `text_chunker` in the script splits long sequences in order to process them separately.
* The public transnormer models have been trained with entire sentences as the input and thus work best when presented with a complete sentence. The way that the file is processed in the script (#L125ff) assumes that each line in the input file contains a single sentence. If the input data comes in a different form, you may want to consider bringing it into this form first by preprocessing (e.g. removing line-breaks, dashes) and using a sentencizer (e.g. https://spacy.io/api/sentencizer).
