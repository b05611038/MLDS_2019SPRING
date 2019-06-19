# Chinese Chatbot Implemeatation by pyotrch

MLDS HW2-2

## Training Chinese Chatbot

Sample command, please use -h for further information

### Preprocess

Please run the command to build the needed package for text encoding/decoding.<br />
The process would also generate needed data training pair in the data folder.

```
python3 text_preprocessing.py one_hot true 3
```

### Training Chinese Chatbot

```
python3 train_seq2seq.py seq2seq_got 256 True dot 40 128 -1
```

## Testing Chinese Chatbot

### Output testing file

Output the file for responding input file

```
python3 predict.py test_input.txt seq2seq_dot.pkl word2vec.pkl 2 -1
```

### Evaluate model performance

The process would output preplexity and correlation score.<br />
Please change directory to ./evaluation

```
python3 main.py input.txt output.txt
```

### Plot training history

```
python3 plot_his.py history [csv file 1] [csv file 2] ...
```
