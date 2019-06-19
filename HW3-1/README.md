# Generative Adversarial Network Implementation by pytorch

MLDS HW3-1

## Training GAN

Sample command, please use -h for further information <br />
In training process, image would auto save for recording <br />

```
python3 train_GAN.py GAN GAN -1
```

## Testing

Output images format with  course testing output

```
python3 GAN.pkl -1
```
Detect face completion by another model

```
python3 baseline.py --input output_GAN.png
```

### Training history plot

```
python3 plot_his_GAN.py history [csv file 1] [csv file 2] ...
```

### Reference of Testing model

```
https://github.com/nagadomi/lbpcascade_animeface
```
