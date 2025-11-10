## Segmentation NDS

Training and inference scripts for nutrient mask.
The model uses ConvNeXt-tiny backbone for Unet encoder. I chose the tiny one to fit on the limited gpu power.
It takes images of timestamps t-2, t-1 and t to predict nutrient mask for timestamp t.
Each image is passed through encoder with shared weights.
A single decoder is used to decode the images and to output probabilities for mask.
Encoder images are concatenated and passed through Conv2d 1x1 -> LayerNorm -> Activation from 3 x ch -> ch.
The same block is used for skip connections. Decoder has resblocks of ConvNeXt.
The input size is 512x512 from random crop. The training lasts 200 epoch but from the epoch 61 there were
no improvements.

#### Best score:

[Epoch 061] train_loss=0.7385 val_loss=0.7438 val_dice=0.3119, val_f1=0.3350, val_iou=0.2374

#### Weights and Logs:

The full tensorboard logs and best weights can be found here:

https://drive.google.com/drive/folders/1E-5vNw6KhcyXPy4RzqO5YBPUbBVyFSqB

### Installation

```sh
pip install -r requirements.txt
```

### Train

Validation and train datasets should have separate folder for each sample and
each sample folder should contain files starting with:

- image_i0
- image_i1
- image_i2
- boundary_mask
- nutrient_mask_g0

Example of the terminal run from the project root path.

```sh
python train.py --train-root PATH_TO_TRAIN_DATA --val-root PATH_TO_VAL_DATA
```

### Inference

The datasets should have separate folder for each sample with the images named
Each item folder should contain files starting with:

- image_i0
- image_i1
- image_i2
- boundary_mask

Example of the terminal run from the project root path.

```sh
python inference.py --input-root PATH_TO_DATA --output-root PATH_TO_OUTPUT --model_path CHECKPOINT_PATH
```
