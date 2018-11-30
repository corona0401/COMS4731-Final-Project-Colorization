from models import *
from dataset import *
from utils import *
import torchvision
import torch
import torch.optim as optim
import torch.utils.data as data
import torch.nn as nn

class complete_net(nn.Module):
    def __init__(self):
        super(complete_net, self).__init__()
        self.encoder = encoder_net()
        self.conv1 = nn.Conv2d(1256, 256, kernel_size=1, stride=1, padding=1, bias=False)
        self.decoder = decoder_net()

    def forward(self, x, emd):
        end = self.encoder(x)
        print("encoder result shape", end.shape)
        print("emd shape", emd.shape)
        # concate end and emd to mix
        emd = emd.expand(1, 32, 32, 1000)
        emd = emd.transpose(1, 3)
        mix = torch.cat((end, emd), 1)
        print("mix shape", mix.shape)
        mix = F.relu(self.conv1(mix))
        res = self.decoder(mix)
        return res

color_net = complete_net()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
criterion = nn.MSELoss()
optimizer = optim.SGD(color_net.parameters(), lr=0.001, momentum=0.9)

transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
place_dataset = PlaceDataset(image_dir = 'places_train/', transform=transform)
dataset_len = len(place_dataset)
train_size = int(0.9*dataset_len)
val_size = int(dataset_len-train_size)
train_dataset, val_dataset = data.random_split(place_dataset, [train_size, val_size])
train_loader = data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=2)
val_loader = data.DataLoader(val_dataset, batch_size=100, shuffle=False, num_workers=2)

for epoch in range(1):
  
    running_loss = 0.0
    val_loss = 0.0
  
    for i, data in enumerate(train_loader, 0):

      inputs, embeds, labels = data['image'], data['embedding'], data['label']
      inputs, embeds, labels = inputs.to(device), embeds.to(device), labels.to(device)

      optimizer.zero_grad()

      outputs = color_net(inputs.float(), embeds.float())
     
      print("output shape", outputs.shape)
      print("label shape", labels.shape)
      loss = criterion(outputs, labels.float())
      loss.backward()
      optimizer.step()

      running_loss += loss.item()
      print(running_loss)

    
      if i % 100 == 99:
        print('[%d, %5d] training loss: %.3f validation accurancy: %.3f'
       	% (epoch + 1, i + 1, running_loss / 100, val_loss / 100))
        running_loss = 0.0
        val_loss = 0.0

print('Finished Training')

torch.save(color_net.state_dict(), '/colornet_v1.pt')
