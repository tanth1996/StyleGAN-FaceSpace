# StyleGAN FaceSpace
This repository contains code utilising a PyTorch port of Nvidia's StyleGAN to construct a framework to generate blends of faces based on several face images, as well as experiments on style blending.

# Synthetic Face Generation
With this framework, an encoder can be trained to map orthogonal (one-hot) encoded vectors that represent selected training faces from a target category to a latent vector (512 dimensional) in the latent input space of StyleGAN's synthesis network to generate the training faces.

The input to the encoder can then be modified to blend features of the training faces to generate new faces.

Using a few training images of a target category:

![Target faces](https://github.com/tanth1996/StyleGAN-FaceSpace/blob/master/Example%20Images/Idol%20FaceSpace%20Target%20Faces.png "Japanese Idol Target Faces")

We can blend the features of the training faces to make new faces (bar graph illustrates weighting of features of training faces in the generated face):

![Generated faces](https://github.com/tanth1996/StyleGAN-FaceSpace/blob/master/Example%20Images/Idol%20FaceSpace%20Synthetic%20Faces.png "Japanese Idol Generated Faces")

And also do face morphing:

![Face morphing](https://github.com/tanth1996/StyleGAN-FaceSpace/blob/master/Example%20Images/Idol%20FaceSpace%20Face%20Morphing.png "Japanese Idol Face Morphing")

# FaceSpace API Usage
1.	Collect several images (6-10) of the target distribution
2.	Crop and align the images similar to FFHQ facial image specifications
3.	Construct a FaceData (or FaceLatents) object from a folder containing the cropped and aligned images
4.	Construct a model consisting of a multilayer perceptron encoder network and the StyleGAN synthesis network
5.	Train the encoder network for a prescribed number of epochs for a prescribed number of cycles with annealed learning rates
6.	Once the network is trained, synthetic face generation can be performed by varying the encoder input, and face morphing can be performed by interpolating between face vectors

# Limitations
Face blending/morphing is not alwayys stable, but with some manual tuning of the input vectors, a good quality face can be obtained by eye. 
Selecting images with similar lighting/angles can improve the stability of the blending/morphing.

# Due Credit
## Original StyleGAN by Nvidia
Nvidia's groundbreaking [StyleGAN](https://github.com/NVlabs/stylegan) forms the heart of this project.

## PyTorch port of StyleGAN
Code for constructing the StyleGAN model is based on a PyTorch port of Nvidia's StyleGAN from the [Lernapparat repository](https://github.com/lernapparat/lernapparat/tree/541b6b1f21cbce602c4981cb3fb73f75b42227c8).

## UsideU
This project was conceived of as part of one of UsideU's projects during my internship. This work would not have been possible without the supportive members of UsideU, especially Alireza Goudarzi who generously and patiently mentored me throughtout my undertaking of this project.
