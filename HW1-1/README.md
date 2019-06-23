# Deep v.s. Shallow

MLDS HW1-1

## Stimulate a function

Program can provided the trained neural network with different depth but same amount of parameters to fit a function

### Training

```
python3 train_FT.py 2 132 func_fit_NN_d2n128 -1
python3 train_FT.py 4 27 func_fit_NN_d4n27 -1
python3 train_FT.py 8 16 func_fit_NN_d8c16 -1
```

### Contruct the Image

```
python3 plot_his_FT.py func_fit_plot func_fit_NN_d2n128.csv func_fit_NN_d4n27.csv func_fit_NN_d8n16.csv
```

## Train on actual tasks

Program can provided the trained neural network with different depth but same amount of parameters to fit cifar-10 dataset.

### Training

```
python3 train_AT.py CNN 2 132 actual_task_NN_d2k128 100 128 -1
python3 train_AT.py CNN 4 27 actual_task_NN_d4k27 100 128 -1
python3 train_AT.py CNN 8 16 actual_task_NN_d8k16 100 128 -1
```

### Contruct the Image

```
python3 plott_his_AT.py actual_task_plot actual_task_NN_d2k128.csv actual_task_NN_d4k27.csv actual_task_NN_d8k16.csv
```
