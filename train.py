import models
from dataset import *
import torchvision
import torch
import torch.optim as optim
import torch.utils.data as data
import torch.nn as nn

class complete_net(nn.Module):
    def __init__(self):
        super(complete_net, self).__init__()
        self.low_feature = models.low_feature_net()
        self.mid_feature = models.mid_feature_net()
        self.global_feature = models.global_feature_net()
        self.upsample = models.upsample_color_net()

    def forward(self, x):
        low = self.low_feature(x)
        mid = self.mid_feature(low)
        glb = self.global_feature(low)
        glb = glb.unsqueeze(2)
        glb = glb.unsqueeze(2)
        glb = glb.expand(1, 256, 32, 32)
        mix = torch.cat((mid, glb), 1)
        res = self.upsample(mix)
        return res


color_net = complete_net()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(color_net.parameters(), lr=0.001, momentum=0.9)

transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
place_dataset = PlaceDataset(image_dir = 'places_train/', transform=transform)
dataset_len = len(place_dataset)
train_size = int(0.9*dataset_len)
val_size = int(0.1*dataset_len)
train_dataset, val_dataset = data.random_split(place_dataset, [train_size, val_size])
test_dataset = PlaceDataset(image_dir = 'places_test/', transform=transform)
# train_loader = data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=2)
# val_loader = data.DataLoader(val_dataset, batch_size=100, shuffle=False, num_workers=2)
# test_loader = data.DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)

for epoch in range(10):
  
    running_loss = 0.0
    val_acc = 0.0
  
    for i in range(len(train_dataset)):

      data = train_dataset[i]
      inputs, labels = data['image'], data['label']
      inputs, labels = inputs.to(device), labels.to(device)

      optimizer.zero_grad()

      outputs = color_net(inputs)
      print(outputs.shape)
      print(labels.shape)
      loss = criterion(outputs, labels.long())
      loss.backward()
      optimizer.step()

      running_loss += loss.item()
      val_acc += val_accurancy(cnn_net, val_loader)

    
      if i % 50 == 49:
        print('[%d, %5d] training loss: %.3f validation accurancy: %.3f'
        	% (epoch + 1, i + 1, running_loss / 50, val_acc / 50))
        running_loss = 0.0
        val_acc = 0.0

print('Finished Training')
