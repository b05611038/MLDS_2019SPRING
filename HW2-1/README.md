# HW2-1 Video Capture Generalization

MLDS HW2-1

## Training Video Caption Generation seq2seq Model

Sample command, please use -h for further information

### Preprocess

Please run the command to build the needed package for text encoding/decoding.<br />
The process would also generate needed data training pair in the data folder.

```
python3 text_preprocessing.py one_hot 3 5 20
```

### Training Model

```
python3 train_video2text.py video2text 40 128 false -1
```

## Testing Video Caption Generation seq2seq Model

### Output testing file

Note that the beamsearch of the predicting process will exploring all word space.

```
python3 predict.py video2text.pkl word2vec.pkl 2 -1
```

### Evaluate bleu score

```
python3 bleu_eval.py output.txt 
```
