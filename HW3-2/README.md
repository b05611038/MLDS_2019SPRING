# Text to Image Generation Implementation by pytorch

MLDS HW3-2

## Training Text2ImageGAN

Sample command, please use -h for further information <br />
In training process, image would auto save for recording <br />

```
python3 train_Text2ImageGAN.py GAN Text2ImageGAN -1
```

## Testing

Output images format with  course testing output

```
python3 Text2ImageGAN.pkl -1
```
<br />
Detect face completion by another model

```
python3 baseline.py --input output_Text2ImageGAN.png 
```

### Reference of Testing model

```
https://github.com/nagadomi/lbpcascade_animeface
```
