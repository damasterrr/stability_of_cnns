# stability_of_cnns
In this project I want to test and try to improve performance of chosen state-of-the-art convolutional neural networks applying Gaussian process to the training set.
This reposatory is part of my Bachelor's project for the university. The name of the project is "Geometric prior for image processing with convolutional neural networks".
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
import image from the database
```sh
image = data_batch_1['data'][1]
```
prepare for transformation
```sh
image = image.reshape(3,32,32)
image = image.transpose(1,2,0)
```
transform data with imported Julia function
```sh
dimage = deform.deform(image,0.001)
```
Display the image
```sh
plt.imshow(dimage)
```
deformed image compared to the original with magnitude (m) of 0.0005 :

![m1]

m = 0.001 :

![m2]

m = 0.0015 :

![m3]

m = 0.002 :

![m4]

Note that in this example 'nice' deformations are chosen. Typically with the increasing magnitude the image is becoming less and less recognisable by convolutional neural networks.

[m1]: images/00005.png
[m2]: images/0001.png
[m3]: images/00015.png
[m4]: images/0002.png
