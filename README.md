# stability_of_cnns
In this project I want to test and try to improve performance of chosen state-of-the-art convolutional neural networks applying Gaussian process to the training set.
This reposatory is part of the Bachelor's project for the university with the name "Geometric prior for image processing with convolutional neural networks".
.
In the project Julia is used to deform images whereas Python to train model on them.

### Getting started

I suggest using docker environment:
* [Docker Data Science Stack](https://hub.docker.com/r/jupyter/datascience-notebook)

Additional dependencies for python files:
* [PyTorch](https://hub.docker.com/r/jupyter/datascience-notebook)
* [PyTorch Torchvision](https://hub.docker.com/r/jupyter/datascience-notebook)

### Installation

1. Clone the repo

   ```sh
   git clone https://github.com/sukhovd/stability_of_cnns
   ```

### Usage
import first image from the database
```sh
image = data_batch_1[b'data'][1]
```
prepare for transformation
```sh
image = image.reshape(3,32,32)
image = image.transpose(1,2,0)
```
transform data with imported Julia function
```sh
dimage = Julia_function.deform(image,0.1)
```
Display the image
```sh
plt.imshow(dimage)
```
deformed image compared to the original with magnitude (m) of 0.1
![m1]
m = 0.2
![m2]
m = 0.3
![m3]
m = 0.4
![m4]

Note that in this example 'nice' deformations are chosen. Typically with the increasing magnitude the image is becoming less and less recognisable by convolutional neural networks.

[m1]: images/m1.png
[m2]: images/m2.png
[m3]: images/m3.png
[m4]: images/m4.png
