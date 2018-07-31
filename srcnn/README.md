# Replicated "Image Super-Resolution Using Deep Convolutional Networks"

I could not get ImageNet dataset.

* training set: LSUN bridge train.
* test set: LSUN bridge val.

![SRCNN](../assets/srcnn_000.jpg)

![SRCNN](../assets/srcnn_001.jpg)

## FLags

* **ckpt-dir-path** : path to the directory of checkpoints.
* **logs-dir-path** : path to the directory of logs.
* **training-images-path** : path to the directory of images. all images under this directory will be used for training.
* **sr-source-path** : path to the source image to do super resolution.
* **sr-target-path** : path to the result of super resolution.
* **train** : if this session is for training.
* **batch-size** : batch size for training. ignored if the the session is not for training.
* **upscaling-factor** : upscaling factor to train / do super resolution.
* **crop-image-size** : size of cropped images.
* **crop-image-side** : size of border of cropped images. pixels on the border are ignored.
* **srcnn-fsub** : ~~I don't know how to explain XD.~~
* **srcnn-f1** : size of the first filter.
* **srcnn-f2** : size of the second filter.
* **srcnn-f3** : size of the third filter.
* **srcnn-n1** : depth of the first hidden layer.
* **srcnn-n2** : depth of the second hidden layer.
