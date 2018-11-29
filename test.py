from dataset import *
from train import *
from skimage.io import imsave
import torch.utils.data as data
import torch

test_dataset = PlaceDataset(image_dir = 'places_test/', transform=transform)
test_loader = data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

# get the pre-trained model
color_net = complete_net(*args, **kwargs)
color_net.load_state_dict(torch.load('/colornet_v1.pt'))
color_net.eval()

# Test model
data = test_loader[0]
inputs, embeds, labels = data['image'], data['embedding'], data['label']
output = color_net(inputs, embeds)
output = output * 128

# Output colorizations
for i in range(len(output)):
    cur = np.zeros((256, 256, 3))
    cur[:,:,0] = inputs[i][:,:,0]
    cur[:,:,1:] = output[i]
    imsave("result/img_"+str(i)+".png", lab2rgb(cur))