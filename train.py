
import models
import torchvision
import torch.optim as optim

class complete_net(nn.Module):
    def __init__(self):
        super(complete_net, self).__init__()
        self.low_feature = torchvision.models.vgg16()
        self.mid_feature = models.mid_feature_net()
        self.global_feature = models.global_feature_net()
        self.upsample = models.upsample_color_net()

    def forward(self, x):
        low = self.low_feature(x)
        mid = self.mid_feature(low)
        glb = self.global_feature(low)
        mix = torch.cat((mid, glb), 0)
        res = self.upsample(mix)
        return res


color_net = complete_net()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(cnn_net.parameters(), lr=0.001, momentum=0.9)

# To do: define train_loader, val_loader
for epoch in range(10):
  
    running_loss = 0.0
    val_acc = 0.0
  
    for i, data in enumerate(train_loader, 0):
    
      inputs, labels = data
      inputs, labels = inputs.to(device), labels.to(device)
    
      optimizer.zero_grad()

      outputs = color_net(inputs)
      loss = criterion(outputs, labels)
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
