# Generalization

MLDS HW1-3

## Can network fit random labels?

### Training

dataset: Cifar-10 <br />
model: ResNet18

```
python3 train_RL.py random_label_0.2 0.2 100 128 -1
```

### Plot results

```
python3 plot_his_RL.py [image name] [csv file 1] [csv file 2] ...
```

## Number of parameters v.s. Generalization

### Training

dataset: Cifar-10 <br />
model: stack conv network

```
python3 train_Gpara.py conv_16 16 100 128 -1
```

### Plot results

```
python3 plot_Gpara.py Gpara [csv file 1]  [csv file 2] ...
```

## Flatness v.s. Generalization

dataset: Cifar-10 <br />
model: stack conv network

```
python3 train_Gsen.py conv_lr0.001_bs128 100 0.001 128 -1
```

### Plot results

```
python3 plot_Gsen.py [image name] [seperation letter] [hyperparameters name] [csv file] ...
```
