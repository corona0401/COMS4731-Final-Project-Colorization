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
test_loader = data.DataLoader(test_dataset, batch_size=5, shuffle=False, num_workers=2)

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

# input: 1 x H x W
# output: 2 x H x W
# rgb: H x W x 3
def recon_rgb(_input,_output):
    lab = np.zeros((256, 256, 3))
    lab[:,:,0] = _input[0,:,:]*100
    lab[:,:,1:3] = np.transpose(_output,(1,2,0))*128
    rgb = (lab2rgb(lab)*256).astype(np.uint8)
    return rgb

# Output colorizations
for i in range(len(output)):
    cur = np.zeros((256, 256, 3))
    
    imsave("result/img_"+str(i)+"_input.png", inputs[i][0])
    
    #cur[:,:,0] = inputs[i][0,:,:]*100
    #cur[:,:,1:] = np.transpose(output[i],(1,2,0))*128
    #imsave("result/img_"+str(i)+"_pred.png", lab2rgb(cur))
    imsave("result/img_"+str(i)+"_pred.png", recon_rgb(inputs[i],output[i]))

    #cur[:,:,0] = inputs[i][0,:,:]*100
    #cur[:,:,1:] = np.transpose(labels[i],(1,2,0))*128
    #imsave("result/img_"+str(i)+"_gt.png", lab2rgb(cur))
    imsave("result/img_"+str(i)+"_gt.png", recon_rgb(inputs[i],labels[i]))
    
