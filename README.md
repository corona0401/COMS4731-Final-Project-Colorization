# COMS4731-Final-Project-Colorization
This is the final project for Fall 2018 COMS4731 Computer Vision course in Columbia University
# Colorize your own greyscale image
* Requirements

  Python 2.7 or 3.6
  
  Pytorch 0.4

* Using the pre-trained model

  We provide two pre-trained models. `color_net.pth` is our baseline model, in which we combines the classification output with the low-level image features. `color_net_global.pth` is our best model, in which we combines the global-level image features with the low-level image features. 
  
  If you want to use `color_net.pth`: in `test.py`, change the model name in line 17 as `color_net.pth`, then run `python test.py`. The result will be in the result folder.
  
  If you want to use `color_net_global.pth`: in `test_global.py`, change the model name in line 17 as `color_net_global.pth`, then run `python test_global.py`. The result will be in the result folder.
  
# Train your own model
