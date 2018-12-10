# COMS4731-Final-Project-Colorization
This is the final project for Fall 2018 COMS4731 Computer Vision course in Columbia University
# Colorize your own greyscale image
* Requirements

  Python 2.7 or 3.6
  
  Pytorch 0.4

* Using the pre-trained model

  We provide two pre-trained models. You can download the pre-trained models in realease tab. `colornet.pth` is our baseline model, in which we combines the classification output with the low-level image features. `colornet_global.pth` is our best model, in which we combines the global-level image features with the low-level image features. These two pretrained models can be downloaded under the 1.0 release version.
  
  If you want to use `colornet.pth`: in `test.py`, make sure file's name in the line 17 is "colornet.pth". Run `python test.py`. The result will be in the result folder.
  
  If you want to use `colornet_global.pth`: in `test_global.py`, make sure file's name in the line 17 is "colornet_global.pth". Run `python test_global.py`. The result will be in the result_global folder.
  
# Train your own model
If you want to train your own model using our baseline method, you can specify the dataset directory in `train.py` line 27. If you want to train your own model using our best method, you can specify the dataset directory in `train_global.py` line 27. 
