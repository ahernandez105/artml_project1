# 10-615 Machine Learning and Art Project1

the following repo is dedicated to the project1 submission in 10-615 at Carnegie Mellon University.
Our team explored the applications of style transfer on hand doodle drawings with hope of 
showing, with the help of ML, anyone can be an artist. 


## Running Jupyter Notebook
The following notebook was taken from the following AMI on aws: **Art_ML_02_01_2019 - ami-0d9c0acbc057a2acd.** Modifications 
were made to enable seamless hyper parameter tuning while running 
style transfer iterations. A style and content image is provided where a 
loss function, using hidden layers from the VGG19 model, attempts to transfer
an optimal amount of style to the content image. <br/> 

Running the notebook is governed by the following configurations and hyperparameters:
* trial_name: name of the trial being run 
* content_file_name: content image (.png or .jpeg)
* style_file_name: style image (.png or .jpeg)
* iterations: the number of epochs before terminating style transfer run
* total_variation_weight: ??
* style weight: the amount of weight in the loss function allocated to the style image
* content_weight: the amount of weight in the loss function allocated to the content image
* content_features: hidden layer in VGG19 used to create content features
* style_features: hidden layers in VGG19 used to create style features

Fine tuning these parameters can be achieved through trial error and in the directory there is 
a file  _vgg19_layers.txt_ that list the layers you can substitute in an out for generating 
features. All hyperparameter tuning is done in cell 61 (second cell). <br/>

Ensure your style image and content image are in the same directory as the notebook.
After the model is finished running a directory taking the name _trial_name_ will be created 
where images from all iterations are stored, including a time elapsed .gif of the iteration
images. **NOTE**, creating the .gif requires the package `imageio`. Using the Art AMI through `aws` throws
errors and this packaged needs to be installed locally with the following command: <br/>
`pip install imageio` <br/>
This should do the trick and you can run the notebook localy, assuming you have, [`anaconda`](https://docs.anaconda.com/anaconda/install/), [`tensorflow`](https://www.tensorflow.org/install/pip) and [`keras`](https://pypi.org/project/Keras/) installed locally, as well. <br/>

If you run into issues or want to run the notebook within AWS, you will need to comment out the 
following lines of code:

* Cell 38 (first cell) line 12
* Cell 76 (last cell) the entire cell <br\>

Furthermore, navigating to scripts/ in this repo is a script similar to this notebook but can be ran over
a batch of images and randomly selects hidden units to use for feature extraction. 





