from dataset import *
from train import complete_net 
from skimage.io import imsave
import torch.utils.data as data
import torch
import torchvision
from skimage.color import lab2rgb

transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
test_dataset = PlaceDataset(image_dir = 'places_test/', transform=transform)
test_loader = data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

# get the pre-trained model
color_net = complete_net()
color_net.load_state_dict(torch.load('colornet_v1.pth'))
color_net.eval()

# Test model
dataiter = iter(test_loader)
data = dataiter.next()
inputs, embeds, labels = data['image'], data['embedding'], data['label']
with torch.no_grad():
    output = color_net(inputs.float(), embeds.float())
output = output * 128

# Output colorizations
for i in range(len(output)):
    cur = np.zeros((256, 256, 3))
    cur[:,:,0] = inputs[i][:,:,0]
    cur[:,:,1:] = output[i].transpose(0,2)
    imsave("result/img_"+str(i)+".png", lab2rgb(cur))
