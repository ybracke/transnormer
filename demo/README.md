# Demo

## Notebook `transnormer.ipynb`

The Jupyter Notebook exemplifies the basic usage for transnormer models.

## Script `process-full-text.py`

The script is a demo for processing a full text file with a transnormer model.
To run the demo, you must pass a text file. You can get an example text file with the bash script:

```bash
bash get-full-text.sh
```

Then run the demo like this:

```bash
python3 process-full-text.py data/1700-1799/robins_artillerie_1745.txt
```

Note:  The current transnormer models have been trained with entire sentences as input and thus work best when presented with complete sentences.
The file processing in the script (#L125ff) assumes that each line in the input file contains a sentence.
If this is not the case for your data, consider bringing your input data into this shape first by preprocessing (e.g. removing line-breaks, dashes) and using a sentencizer (e.g. https://spacy.io/api/sentencizer).
