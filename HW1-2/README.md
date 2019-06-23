# Optimization

MLDS HW1-2

## Visualize the optimization process

The code will provide needed model for visualizing model optimization process.

### Training

```
python3 train_AT.py visualized_cifar10_NN 100 10 128 -1
python3 train_FT.py visualized_func_fit_NN 100000 10000 -1
```

### Constructe the Image

first can visualize the weight state in first layer of NN, while whole is visualizing whole model weight.

```
python3 show_optimiztion.py optimized_process_cifar10_whole visualized_cifar10_NN whole
python3 show_optimiztion.py optimized_process_cifar10_first visualized_cifar10_NN first
python3 show_optimiztion.py optimized_process_func_fit_whole visualized_func_fit_NN whole
python3 show_optimiztion.py optimized_process_func_fit_first visualized_func_fit_NN first
```

## Observe gradient norm during training

The code can provide differnet grad norm value on differnet task.

### Training

```
python3 train_AT.py grad_norm_cifar10_NN 100 101 128 -1
python3 train_FT.py grad_norm_func_fit_NN 100000 100001 -1 
```

### Constructe the Image

The norm value plot would be recored in csv file having the same name with model. <br />
The program cam plot several models on the plot with put more csv files on args.

```
python3 plot_his_AT.py grad_norm_cifar10_NN.csv ...
python3 plot_his_FT.py grad_norm_func_fit_NN.csv ...
```

## What happens when gradient is almost zero

The code will provide the model for being trained to saddle point, but the epoch numbers should be controlled by user. <br />
Note that the training program will save many models, therefore, be awared of the disk volumn.

### Training

```
python3 train_NZ.py grad_zero_NN grad_zero_NN 10000 2000 1000 -1
```

### Constructe the Image

The program needs to sample many neighbor NN between to check that the model is indeed located at saddle point.  <br />
In such sampling and calculation process, it will take lots of time to generate the image.

```
plot_minimum_ratio.py minimum_ratio_plot grad_zero_NN 1000 100 -1
```
