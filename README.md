
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
file  _vgg19_layers.txt_ that list the layers you can substitute in an out for generating 
features. <br/>

After the model is finished running a directory taking the name _trial_name_ will be created 
where images from all iterations are stored. 



