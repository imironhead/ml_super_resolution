# Replicate "Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network"

[arXiv:
Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network](https://arxiv.org/abs/1609.05158)

## Requirements

* numpy
* pillow
* scikit-image
* tensorflow

## Dataset Preprocessing

Build sub-pixel convolved ground-truth.

* **source_dir_path**: path to a directory which contains images for training. the script pre-process those images to tfrecord format for training.
* **result_dir_path**: path to a directory for keeping pre-processed results.
* **upscaling_factor**: upscaling factor for training (affect depth of the last layer).
* **lr_patch_size**: size of low resolution patches for training.

## Training

* **data_path**: path to training data (tfrecord) directory.
* **ckpt_path**: path to a directory for keeping the checkpoint.
* **logs_path**: path to a directory for keeping log.
* **batch_size**: size of each batch during training, default is 64.
* **scaling_factor**: scaling factor for training, default is 3. It should conform to the dataset.
* **lr_patch_size**: size of low resolution patches as training data, default is 17. It should conform to the dataset.
* **initial_learning_rate**: initial value of learning rate, default is 0.1.
* **learning_rate_decay_factor**: decay factor of the learning rate. applied to learning rate periodically.
* **learning_rate_decay_steps**: period of learning rate in steps, default is 2560.
* **stop_training_at_k_step**: stop training once reach the specified step, default is 10000.

## Testing

* **data_path**: path to the test data directory to do PSNR/SSIM evaluation. If the path is not a directory, treat it as an image and do super-resolution.
* **ckpt_path**: path to the checkpoint.
* **result_path**: path for the super-resolved image if **data_path** is not a directory.
* **score_space**: do evaluation on y([Y]UV) or rgb([RGB]).

## Results

### 3x - Set5

| image \ metric   | PSNR (Y) | PSNR (RGB) | SSIM (Y) | SSIM (RGB) |
|------------------|----------|------------|----------|------------|
| baby_GT.bmp      | 33.7472  | 33.5818    | 0.9395   | 0.9521     |
| bird_GT.bmp      | 34.0798  | 33.3488    | 0.9655   | 0.9748     |
| butterfly_GT.bmp | 26.4748  | 26.4421    | 0.9550   | 0.9668     |
| head_GT.bmp      | 32.3847  | 30.5316    | 0.8586   | 0.8727     |
| woman_GT.bmp     | 30.0418  | 29.9995    | 0.9490   | 0.9650     |
| average          | 31.3457  | 30.7808    | 0.9335   | 0.9463     |

### 3x - Set14

| image \ metric | PSNR (Y) | PSNR (RGB) | SSIM (Y) | SSIM (RGB) |
|----------------|----------|------------|----------|------------|
| baboon.bmp     | 22.0401  | 21.3238    | 0.6558   | 0.6994     |
| barbara.bmp    | 24.8522  | 24.6806    | 0.7889   | 0.8258     |
| bridge.bmp     | 24.9163  | 24.9158    | 0.7930   | 0.8423     |
| coastguard.bmp | 25.7097  | 25.7028    | 0.7140   | 0.7217     |
| comic.bmp      | 23.1227  | 22.9184    | 0.8459   | 0.8822     |
| face.bmp       | 32.3426  | 30.4955    | 0.8576   | 0.8722     |
| flowers.bmp    | 27.7271  | 26.9823    | 0.8843   | 0.9058     |
| foreman.bmp    | 29.9510  | 29.8202    | 0.9581   | 0.9695     |
| lenna.bmp      | 32.2138  | 31.2232    | 0.8879   | 0.8937     |
| man.bmp        | 26.9681  | 26.9633    | 0.8323   | 0.8828     |
| monarch.bmp    | 31.3954  | 31.3330    | 0.9608   | 0.9691     |
| pepper.bmp     | 31.1340  | 29.6980    | 0.8996   | 0.8976     |
| ppt3.bmp       | 24.9575  | 24.6559    | 0.9385   | 0.9477     |
| zebra.bmp      | 27.8160  | 27.7703    | 0.8918   | 0.9232     |
| average        | 27.5105  | 27.0345    | 0.8506   | 0.8738     |

### 4x - Set5

| image \ metric   | PSNR (Y) | PSNR (RGB) | SSIM (Y) | SSIM (RGB) |
|------------------|----------|------------|----------|------------|
| baby_GT.bmp      | 30.3603  | 30.2519    | 0.9049   | 0.9241     |
| bird_GT.bmp      | 28.9599  | 28.2401    | 0.9181   | 0.9420     |
| butterfly_GT.bmp | 21.9552  | 21.9966    | 0.9057   | 0.9276     |
| head_GT.bmp      | 30.4144  | 28.9813    | 0.8317   | 0.8593     |
| woman_GT.bmp     | 25.6424  | 25.6243    | 0.9055   | 0.9321     |
| average          | 27.4665  | 27.0188    | 0.8932   | 0.9170     |

### 4x - Set14

| image \ metric | PSNR (Y) | PSNR (RGB) | SSIM (Y) | SSIM (RGB) |
|----------------|----------|------------|----------|------------|
| baboon.bmp     | 20.9618  | 20.3128    | 0.5669   | 0.6395     |
| barbara.bmp    | 23.7583  | 23.5458    | 0.7468   | 0.8010     |
| bridge.bmp     | 23.0620  | 23.0563    | 0.7264   | 0.7872     |
| coastguard.bmp | 24.1602  | 24.1617    | 0.6414   | 0.6487     |
| comic.bmp      | 20.6292  | 20.4506    | 0.7612   | 0.8160     |
| face.bmp       | 30.3858  | 28.9513    | 0.8319   | 0.8600     |
| flowers.bmp    | 24.6085  | 24.0219    | 0.8254   | 0.8618     |
| foreman.bmp    | 26.2309  | 26.1344    | 0.9310   | 0.9553     |
| lenna.bmp      | 28.6900  | 28.3952    | 0.8490   | 0.8721     |
| man.bmp        | 24.6420  | 24.6367    | 0.7740   | 0.8432     |
| monarch.bmp    | 27.0734  | 27.0245    | 0.9306   | 0.9441     |
| pepper.bmp     | 28.0834  | 27.2095    | 0.8759   | 0.8897     |
| ppt3.bmp       | 21.5625  | 21.2276    | 0.8810   | 0.8945     |
| zebra.bmp      | 23.3518  | 23.3285    | 0.8093   | 0.8596     |
| average        | 24.8000  | 24.4612    | 0.7965   | 0.8338     |

### Butterfly 3X

![Butterfly 3X](../assets/espcn_3x_butterfly.png)

### Butterfly 4X

![Butterfly 4X](../assets/espcn_4x_butterfly.png)

## Note

* Only tanh & 91 model are evaluated.
* Different optimizer. I did not find the optimizer on the paper.
* The difference between the experiments' & papers' may be due to:
    - The blur factor
    - The 3.1 section of the paper mentioned YCbCr. I am not sure if the model should work with YCbCr or RGB. Also the PSNR value might be applied on Y only.
