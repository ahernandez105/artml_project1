The following script will run style transfer over 
a given set of input images, style images and 
hyperparameters. This difference between this script 
and the notebook is:
* this script can be run in the foreground or background of your processes over a defined set of images
* the hidden layers in the VGG19 network which are used to generate features across the style and content images are randomly sampled over a uniform distribution.
<br/>
Our experiments yielded some interesting images, without any injected noise, 
while use this process of hyperparameter tuning. 

Set `size` in line 41 to the number of output images you want to generate and edit
the configurations of those images in line 58. The script can be run be keying in the following in the terminal: <br/>
`python style_transfer_keras_project_randsearch.py`